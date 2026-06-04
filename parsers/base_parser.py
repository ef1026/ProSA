# parsers/base_parser.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class ParseResult:
    """所有解析器的统一输出格式"""
    text: str = ""                                    # 全文文本
    layout: List[Dict] = field(default_factory=list)  # [{"bbox": [x1,y1,x2,y2], "type": str, "text": str}]
    tables: List[Dict] = field(default_factory=list)  # 表格结构
    reading_order: List[int] = field(default_factory=list)  # 文本块阅读顺序
    raw_output: Optional[Dict] = None                 # 原始输出（调试用）


class BaseParser(ABC):
    """解析器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def parse(self, image_path: str) -> ParseResult:
        """
        输入: 文档图像路径
        输出: 统一格式的解析结果
        """
        pass
    
    @abstractmethod
    def parse_image(self, image: np.ndarray) -> ParseResult:
        """
        输入: numpy 数组格式的图像
        输出: 统一格式的解析结果
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"