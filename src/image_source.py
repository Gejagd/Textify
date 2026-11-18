import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from abc import ABC, abstractmethod
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


MAX_CAMERA_ATTEMPTS = 5
CAMERA_INDEX_RANGE = range(MAX_CAMERA_ATTEMPTS)
CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
CAPTURE_KEY_ENTER = 13
CAPTURE_KEY_ESC = 27
WINDOW_PREVIEW = "Camera Preview - Press ENTER to capture, ESC to cancel"


class ImageSource(ABC):
    @abstractmethod
    def acquire(self) -> np.ndarray | None:
        pass


class FileImageSource(ImageSource):
    SUPPORTED_TYPES = [("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]

    def acquire(self) -> np.ndarray | None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=self.SUPPORTED_TYPES
        )
        root.destroy()
        if not path:
            return None
        image = cv2.imread(path)
        return image


class CameraImageSource(ImageSource):
    def _open_camera(self) -> cv2.VideoCapture | None:
        for idx in CAMERA_INDEX_RANGE:
            for backend in CAMERA_BACKENDS:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    return cap
                cap.release()
        return None

    def acquire(self) -> np.ndarray | None:
        cap = self._open_camera()
        if cap is None:
            return None
        captured = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(WINDOW_PREVIEW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == CAPTURE_KEY_ENTER:
                captured = frame.copy()
                break
            if key == CAPTURE_KEY_ESC:
                break
        cap.release()
        cv2.destroyAllWindows()
        return captured


class ImageSourceFactory:
    MODE_FILE = "file"
    MODE_CAMERA = "camera"

    @staticmethod
    def create(mode: str) -> ImageSource:
        if mode == ImageSourceFactory.MODE_CAMERA:
            return CameraImageSource()
        return FileImageSource()
