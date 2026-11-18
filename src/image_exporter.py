import cv2
import numpy as np
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "results"
OUTPUT_EXTENSION = ".png"
OUTPUT_PREFIX = "scan_result"
MAX_INDEX_SEARCH = 10000


class ImageExporter:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self._output_dir = output_dir

    def _ensure_dir(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_indexed_path(self, prefix: str) -> Path:
        base_name = prefix[:-4] if prefix.endswith(OUTPUT_EXTENSION) else prefix
        for i in range(1, MAX_INDEX_SEARCH):
            filename = f"{base_name}_{i:02d}{OUTPUT_EXTENSION}"
            candidate = self._output_dir / filename
            if not candidate.exists():
                return candidate
        return self._output_dir / f"{base_name}_fallback{OUTPUT_EXTENSION}"

    def export(self, image: np.ndarray, filename: str = OUTPUT_PREFIX, auto_increment: bool = True) -> Path:
        self._ensure_dir()
        if auto_increment:
            out_path = self._get_next_indexed_path(filename)
        else:
            name = filename if filename.endswith(OUTPUT_EXTENSION) else f"{filename}{OUTPUT_EXTENSION}"
            out_path = self._output_dir / name

        cv2.imwrite(str(out_path), image)
        return out_path

    def get_output_dir(self) -> Path:
        return self._output_dir
