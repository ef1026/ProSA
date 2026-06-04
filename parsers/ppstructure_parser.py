# parsers/ppstructure_parser.py
# PP-Structure parser — second multi-modal pipeline for cross-architecture validation.
#
# Uses PaddleOCR's PPStructureV3 (PaddleOCR 3.x) or PPStructure (PaddleOCR 2.x)
# which is an entirely different architecture from MinerU (LayoutLMv3 + DiT).
# Both are multi-modal pipelines with OCR text output, enabling fair
# cross-architecture comparison using the same B-TLR matching standard
# (IoU + TextSim).
#
# Architecture: PaddleDetection (layout) + PaddleOCR (text recognition)
# Output: bounding boxes + document-element categories + OCR text (span-level)
# B-TLR matching: IoU + TextSim, identical to MinerU (no iou_only special case)
#
# IMPLEMENTATION NOTE (Windows CUDA DLL conflict):
#   torch (cu128) and paddlepaddle-gpu cannot coexist in the same
#   Python process on Windows — both bundle their own NVIDIA DLLs that clash.
#   This parser launches a *persistent subprocess* running in the separate
#   'paddle' (preferred) or 'ppocr' conda environment (GPU), communicating
#   over stdin/stdout JSON.  The subprocess worker lives in
#   parsers/_ppstructure_worker.py and auto-detects PaddleOCR 2.x vs 3.x.

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .base_parser import BaseParser, ParseResult

log = logging.getLogger(__name__)

# Path to the worker script (same directory)
_WORKER_SCRIPT = str(Path(__file__).with_name("_ppstructure_worker.py"))

# Auto-detect paddle/ppocr env Python path
# Prefer 'paddle' env (PaddlePaddle 3.x + PaddleOCR 3.x / PPStructureV3) over
# legacy 'ppocr' env (PaddlePaddle 2.x + PaddleOCR 2.x / PPStructure).
_PPOCR_PYTHON_CANDIDATES = [
    # ── 'paddle' env (preferred, PaddlePaddle 3.x + Blackwell GPU support) ──
    Path(sys.prefix).parent / "paddle" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "Anaconda3" / "envs" / "paddle" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "anaconda3" / "envs" / "paddle" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "miniconda3" / "envs" / "paddle" / ("python.exe" if os.name == "nt" else "bin/python"),
    # ── Legacy 'ppocr' env (PaddlePaddle 2.x) ──
    Path(sys.prefix).parent / "ppocr" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "Anaconda3" / "envs" / "ppocr" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "anaconda3" / "envs" / "ppocr" / ("python.exe" if os.name == "nt" else "bin/python"),
    Path.home() / "miniconda3" / "envs" / "ppocr" / ("python.exe" if os.name == "nt" else "bin/python"),
]


def _find_ppocr_python() -> str:
    """Locate the Python executable in the ppocr conda environment."""
    env_val = os.environ.get("PPOCR_PYTHON")
    if env_val and Path(env_val).exists():
        return env_val
    for cand in _PPOCR_PYTHON_CANDIDATES:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "Cannot find 'paddle' or 'ppocr' conda environment. "
        "Create it with:\n"
        "  conda create -n paddle python=3.10 -y\n"
        "  conda run -n paddle pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/\n"
        "  conda run -n paddle pip install -U 'paddleocr[doc-parser]'\n"
        "Or set PPOCR_PYTHON env var to point to the Python executable."
    )


class PPStructureParser(BaseParser):
    """Multi-modal document layout parser based on PP-StructureV3.

    Uses a persistent subprocess in the 'paddle' conda env for GPU inference,
    avoiding CUDA DLL conflicts between torch and paddle on Windows.
    Auto-detects PaddleOCR version: 3.x uses PPStructureV3, 2.x falls back to V2.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        show_log: bool = False,
        layout_model_dir: Optional[str] = None,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
    ):
        super().__init__(name="ppstructure")
        self.lang = lang
        self.use_gpu = use_gpu
        self.show_log = show_log
        self._proc: Optional[subprocess.Popen] = None
        self._initialized = False
        self._init_engine()

    def _init_engine(self) -> None:
        """Launch the worker subprocess in the ppocr environment."""
        try:
            ppocr_python = _find_ppocr_python()
            log.info("PPStructure: using Python at %s", ppocr_python)
        except FileNotFoundError as exc:
            log.warning("PPStructure init skipped: %s", exc)
            print(f"Warning: {exc}")
            return

        try:
            # Build env with NVIDIA DLLs on PATH for paddle GPU
            env = os.environ.copy()
            ppocr_root = os.path.dirname(ppocr_python)  # envs/paddle or envs/ppocr
            # conda cuDNN installs to Library/bin
            conda_lib_bin = os.path.join(ppocr_root, "Library", "bin")
            # pip nvidia-* packages install to site-packages/nvidia/*/bin
            sp_nvidia = os.path.join(ppocr_root, "Lib", "site-packages", "nvidia")
            cudnn_bin = os.path.join(sp_nvidia, "cudnn", "bin")
            cublas_bin = os.path.join(sp_nvidia, "cublas", "bin")
            cuda_runtime_bin = os.path.join(sp_nvidia, "cuda_runtime", "bin")
            nvjitlink_bin = os.path.join(sp_nvidia, "nvjitlink", "bin")
            cufft_bin = os.path.join(sp_nvidia, "cufft", "bin")
            cusolver_bin = os.path.join(sp_nvidia, "cusolver", "bin")
            cusparse_bin = os.path.join(sp_nvidia, "cusparse", "bin")
            extra_paths = [p for p in [
                conda_lib_bin, cudnn_bin, cublas_bin, cuda_runtime_bin,
                nvjitlink_bin, cufft_bin, cusolver_bin, cusparse_bin,
            ] if os.path.isdir(p)]
            if extra_paths:
                env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")
            # Suppress model source check for PaddleOCR 3.x. The worker also
            # self-protects by setting these at the top of its module, but we
            # inject them here too so the subprocess environment is
            # deterministic regardless of the worker's own import ordering
            # and so operators can override via the parent shell env.
            env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            env["PADDLE_PDX_MODEL_SOURCE"] = os.environ.get("PADDLE_PDX_MODEL_SOURCE", "BOS")
            env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
            env["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

            self._proc = subprocess.Popen(
                [ppocr_python, _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
                env=env,
            )

            # Send init config
            init_cfg = json.dumps({
                "lang": self.lang,
                "use_gpu": self.use_gpu,
                "show_log": self.show_log,
            })
            self._proc.stdin.write(init_cfg + "\n")
            self._proc.stdin.flush()

            # Wait for ready signal (up to 2 min for first-time model download)
            deadline = time.monotonic() + 120
            resp_line = ""
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    stderr_out = self._proc.stderr.read()
                    raise RuntimeError(
                        f"Worker process died during init (exit={self._proc.returncode}).\n"
                        f"stderr:\n{stderr_out[:3000]}"
                    )
                resp_line = self._proc.stdout.readline()
                if resp_line.strip():
                    break

            if not resp_line.strip():
                self._kill()
                raise RuntimeError("Worker did not respond within 120s timeout")

            resp = json.loads(resp_line.strip())
            if not resp.get("ok"):
                self._kill()
                raise RuntimeError(f"Worker init failed: {resp.get('error', 'unknown')}")

            self._initialized = True
            log.info("PPStructure GPU worker ready (pid=%d)", self._proc.pid)
            print(f"PPStructure GPU worker ready (pid={self._proc.pid})")

        except Exception as exc:
            log.warning("Failed to start PPStructure worker: %s", exc)
            print(f"Warning: PPStructure worker failed to start: {exc}")
            self._kill()

    def _kill(self):
        """Kill the worker subprocess."""
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def _send_image(self, image: np.ndarray) -> dict:
        """Encode image, send to worker, return parsed result dict."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("PP-StructureV3 worker not running.")

        # Encode image as PNG → base64
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image to PNG")
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        req = json.dumps({"image_b64": img_b64})
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()

        # Read response
        resp_line = self._proc.stdout.readline()
        if not resp_line.strip():
            stderr_tail = ""
            try:
                stderr_tail = self._proc.stderr.read(3000)
            except Exception:
                pass
            raise RuntimeError(f"Worker returned empty response. stderr: {stderr_tail}")

        resp = json.loads(resp_line.strip())
        if not resp.get("ok"):
            raise RuntimeError(f"Worker parse error: {resp.get('error', 'unknown')}")

        return resp["result"]

    def parse(self, image_path: str) -> ParseResult:
        if not self._initialized:
            raise RuntimeError("PP-StructureV3 not initialized.")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.parse_image(image)

    def parse_image(self, image: np.ndarray) -> ParseResult:
        if not self._initialized:
            raise RuntimeError("PP-StructureV3 not initialized.")

        result = self._send_image(image)

        layout = result["layout"]
        content_list = result["content_list"]
        middle_json = result["middle_json"]
        full_text = result["text"]

        return ParseResult(
            text=full_text,
            layout=layout,
            tables=result.get("tables", []),
            reading_order=result.get("reading_order", list(range(len(layout)))),
            raw_output={
                "content_list": content_list,
                "middle_json": middle_json,
            },
        )

    def __del__(self):
        """Gracefully shut down the worker on garbage collection."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("__EXIT__\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._kill()
