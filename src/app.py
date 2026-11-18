import numpy as np
from image_source import ImageSourceFactory
from image_processor import ImageProcessor
from image_exporter import ImageExporter


MENU_PROMPT = "\nTextify - CamScanner\n[1] Choose Picture\n[2] Take Picture\nSelect: "
MENU_CHOICE_FILE = "1"
MENU_CHOICE_CAMERA = "2"
ERROR_NO_IMAGE = "[ERROR] No image acquired."
ERROR_INVALID_CHOICE = "[ERROR] Invalid choice. Please enter 1 or 2."
SUCCESS_SAVED = "[OK] Scan saved to: "


class TextifyApp:
    def __init__(self):
        self._processor = ImageProcessor()
        self._exporter = ImageExporter()

    def _prompt_mode(self) -> str:
        return input(MENU_PROMPT).strip()

    def _acquire_image(self, mode: str) -> np.ndarray | None:
        source = ImageSourceFactory.create(mode)
        return source.acquire()

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        return self._processor.process(image)

    def _export_result(self, result: np.ndarray) -> str:
        path = self._exporter.export(result)
        return str(path)

    def run(self) -> int:
        choice = self._prompt_mode()
        if choice == MENU_CHOICE_FILE:
            mode = ImageSourceFactory.MODE_FILE
        elif choice == MENU_CHOICE_CAMERA:
            mode = ImageSourceFactory.MODE_CAMERA
        else:
            print(ERROR_INVALID_CHOICE)
            return 1

        image = self._acquire_image(mode)
        if image is None:
            print(ERROR_NO_IMAGE)
            return 1

        result = self._process_image(image)
        path = self._export_result(result)
        print(SUCCESS_SAVED + path)
        return 0
