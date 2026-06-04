# parsers/mineru_parser.py
# MinerU 文档解析器 — MinerU v2.7+ (mineru package, pipeline backend)
#
# 从 middle_json 的 para_blocks 重建 content_list，保证每个元素都有
# 像素级 bbox + 文本，下游代码无需任何修改。

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from .base_parser import BaseParser, ParseResult


class MinerUParser(BaseParser):
    """MinerU v2.7+ 解析器 (pipeline backend)"""
    
    def __init__(self, formula_enable=None, table_enable=None, lang="en"):
        super().__init__(name="mineru")
        self._formula_enable = formula_enable if formula_enable is not None else True
        self._table_enable = table_enable if table_enable is not None else True
        self._lang = lang
        self._initialized = False
        self._init_mineru()
    
    def _init_mineru(self):
        """延迟初始化 MinerU v2.7 组件"""
        try:
            from mineru.data.data_reader_writer.filebase import FileBasedDataWriter
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze
            from mineru.backend.pipeline.model_json_to_middle_json import (
                result_to_middle_json,
            )
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
                union_make,
            )
            from mineru.utils.enum_class import MakeMode

            self._FileBasedDataWriter = FileBasedDataWriter
            self._doc_analyze = doc_analyze
            self._result_to_middle_json = result_to_middle_json
            self._union_make = union_make
            self._MakeMode = MakeMode
            self._initialized = True
        except ImportError as e:
            print(f"Warning: MinerU (mineru) not installed. Error: {e}")
            self._initialized = False
    
    def parse(self, image_path: str) -> ParseResult:
        """解析文档图像或 PDF 文件。"""
        if not self._initialized:
            raise RuntimeError("MinerU not initialized. Please install mineru package.")
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")
        
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
            pdf_bytes = self._image_to_pdf(image_path)
        else:
            with open(image_path, 'rb') as f:
                pdf_bytes = f.read()
        
        return self._parse_pdf_bytes(pdf_bytes, path.stem)
    
    def parse_image(self, image: np.ndarray) -> ParseResult:
        """解析 numpy 数组格式的图像 (BGR 或 RGB)。"""
        if not self._initialized:
            raise RuntimeError("MinerU not initialized. Please install mineru package.")
        
        pdf_bytes = self._numpy_to_pdf(image)
        return self._parse_pdf_bytes(pdf_bytes, "image")
    
    def _image_to_pdf(self, image_path: str) -> bytes:
        """将图像文件转换为 PDF bytes"""
        from PIL import Image, ImageOps
        import io
        
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img) or img
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pdf_buffer = io.BytesIO()
        img.save(pdf_buffer, format='PDF')
        return pdf_buffer.getvalue()
    
    def _numpy_to_pdf(self, image: np.ndarray) -> bytes:
        """将 numpy 数组转换为 PDF bytes"""
        from PIL import Image
        import io
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        img = Image.fromarray(image_rgb)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pdf_buffer = io.BytesIO()
        img.save(pdf_buffer, format='PDF')
        return pdf_buffer.getvalue()
    
    def _parse_pdf_bytes(self, pdf_bytes: bytes, name: str) -> ParseResult:
        """解析 PDF bytes via MinerU v2.7 pipeline backend."""
        import tempfile
        import os
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            
            image_writer = self._FileBasedDataWriter(image_dir)
            
            # v2.7 batch API: lists of pdf_bytes and langs.
            # Workaround: some MinerU versions may throw ZeroDivisionError
            # in pipeline_analyze when load_images_time rounds to 0.
            last_exc = None
            for attempt in range(3):
                try:
                    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
                        self._doc_analyze(
                            [pdf_bytes],
                            [self._lang],
                            parse_method="auto",
                            formula_enable=self._formula_enable,
                            table_enable=self._table_enable,
                        )
                    )
                    break
                except ZeroDivisionError as exc:
                    last_exc = exc
                    if attempt == 2:
                        raise
                    time.sleep(0.2 * (attempt + 1))
            else:
                if last_exc is not None:
                    raise last_exc
            
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            ocr_enable = ocr_enabled_list[0]
            
            middle_json = self._result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer,
                self._lang, ocr_enable, self._formula_enable,
            )
            
            pdf_info = middle_json.get("pdf_info", [])
            
            image_subdir = os.path.basename(image_dir)
            markdown_text = self._union_make(
                pdf_info, self._MakeMode.MM_MD, image_subdir,
            )
            if isinstance(markdown_text, list):
                markdown_text = "\n".join(str(item) for item in markdown_text)
            
            content_list = self._build_content_list_from_middle_json(middle_json)
            
            layout = self._extract_layout(content_list)
            tables = self._extract_tables(content_list)
            reading_order = list(range(len(layout)))
            
            return ParseResult(
                text=markdown_text,
                layout=layout,
                tables=tables,
                reading_order=reading_order,
                raw_output={
                    "content_list": content_list,
                    "middle_json": middle_json,
                },
            )
    
    @staticmethod
    def _build_content_list_from_middle_json(middle_json: dict) -> List[Dict]:
        """从 middle_json 的 para_blocks 构建 content_list (带像素级 bbox)。"""
        content_list = []
        for page_info in middle_json.get('pdf_info', []):
            page_idx = page_info.get('page_idx', 0)
            for pb in page_info.get('para_blocks', []):
                bbox = pb.get('bbox', [])
                if not bbox or len(bbox) < 4:
                    continue
                
                text_parts = []
                for line in pb.get('lines', []):
                    for span in line.get('spans', []):
                        content = span.get('content', '')
                        if content:
                            text_parts.append(content)
                text = ' '.join(text_parts)
                
                block_type = pb.get('type', 'text')
                
                content_list.append({
                    'type': block_type,
                    'bbox': [float(bbox[0]), float(bbox[1]),
                             float(bbox[2]), float(bbox[3])],
                    'text': text,
                    'page_idx': page_idx,
                })
        return content_list
    
    def _extract_layout(self, content_list: List) -> List[Dict]:
        """从 content_list 提取布局信息"""
        layout = []
        for item in content_list:
            if isinstance(item, dict):
                layout_item = {
                    "bbox": item.get("bbox", []),
                    "type": item.get("type", "text"),
                    "text": item.get("text", ""),
                }
                layout.append(layout_item)
        return layout
    
    def _extract_tables(self, content_list: List) -> List[Dict]:
        """从 content_list 提取表格信息"""
        tables = []
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "table":
                tables.append(item)
        return tables
