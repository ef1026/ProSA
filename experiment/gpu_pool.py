from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Any, Optional

import numpy as np
from tqdm import tqdm


def setup_cuda_dll_paths() -> None:
    import sys
    if sys.platform != "win32":
        return
    nvidia_path = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    for subdir in ["cublas", "cuda_runtime", "cudnn", "cusparse", "cusolver", "cufft", "curand"]:
        bin_path = os.path.join(nvidia_path, subdir, "bin")
        if os.path.exists(bin_path):
            try:
                os.add_dll_directory(bin_path)
            except Exception:
                continue


class GPUParserPool:
    def __init__(self, num_workers: int = 1, parser_kwargs: Optional[dict] = None, parser_name: str = "mineru"):
        setup_cuda_dll_paths()

        self.num_workers = num_workers
        self.parser_name = parser_name
        self._queue: Queue = Queue()
        parser_kwargs = parser_kwargs or {}

        if parser_name == "ppstructure":
            from parsers.ppstructure_parser import PPStructureParser
            for _ in range(num_workers):
                parser = PPStructureParser(**parser_kwargs)
                self._queue.put(parser)
        elif parser_name == "mineru":
            from parsers.mineru_parser import MinerUParser
            # Disable table processing by default — rapid_table crashes on
            # many PubLayNet images (TypeError in format_ocr_results) and the
            # silent exception causes all spans to be lost.
            parser_kwargs.setdefault("table_enable", False)
            for _ in range(num_workers):
                parser = MinerUParser(**parser_kwargs)
                self._queue.put(parser)
        else:
            raise ValueError(f"Unsupported parser: {parser_name}. Use 'mineru' or 'ppstructure'.")

    def warmup(self) -> None:
        blank = np.zeros((1024, 1024, 3), dtype=np.uint8)
        try:
            _ = self.parse_one(blank)
        except Exception as exc:
            # Warmup is only best-effort; keep pipeline runnable.
            print(f"[GPUParserPool] warmup skipped due to error: {exc}")

    def parse_one(self, image_or_path: Any, strict: bool = False) -> Optional[object]:
        """Parse one image or path.

        ``strict=False`` (default): swallow every exception, print a
        traceback, and return ``None`` so an occasional bad page doesn't kill
        a long attack sweep. Safe for phase 1/2 where a few dropped rows are
        tolerable.

        ``strict=True``: re-raise the exception. Use this in the baseline
        build, where a silently-returned ``None`` poisons *every* downstream
        metric (every ``B_SLR`` / ``EIR`` / ``SLR_miss`` for that image will
        collapse to 0 because ``extract_elements_from_result(None) == []``).
        Failing loud here lets the operator fix the parser before hours of
        compute are wasted.
        """
        parser = self._queue.get()
        try:
            if isinstance(image_or_path, np.ndarray):
                return parser.parse_image(image_or_path)
            return parser.parse(str(image_or_path))
        except Exception as exc:
            import traceback
            print(f"[GPUParserPool] parse_one failed: {exc}")
            traceback.print_exc()
            if strict:
                raise
            return None
        finally:
            self._queue.put(parser)

    def parse_batch(self, items: list[Any], desc: str = "GPU parse") -> list[Optional[object]]:
        if not items:
            return []
        results: list[Optional[object]] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_map = {executor.submit(self.parse_one, item): idx for idx, item in enumerate(items)}
            for future in tqdm(as_completed(future_map), total=len(items), desc=desc):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
        return results
