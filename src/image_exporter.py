import cv2
import numpy as np
import os
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "results"
OUTPUT_EXTENSION = ".png"
OUTPUT_PREFIX = "scan_result"


class ImageExporter:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self._output_dir = output_dir

    def _ensure_dir(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_path(self, filename: str) -> Path:
        if not filename.endswith(OUTPUT_EXTENSION):
            filename = filename + OUTPUT_EXTENSION
        return self._output_dir / filename

    def export(self, image: np.ndarray, filename: str = OUTPUT_PREFIX) -> Path:
        self._ensure_dir()
        out_path = self._build_path(filename)
        cv2.imwrite(str(out_path), image)
        return out_path

    def get_output_dir(self) -> Path:
        return self._output_dir
