import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from abc import ABC, abstractmethod
import os
from document_detector import DocumentDetector

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

MAX_CAMERA_ATTEMPTS = 3
CAMERA_INDEX_RANGE = range(MAX_CAMERA_ATTEMPTS)
CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
CAPTURE_KEY_ENTER = 13
CAPTURE_KEY_ESC = 27
WINDOW_PREVIEW = "Textify CamScanner - ENTER: Capture | ESC: Cancel"
COLOR_CONTOUR = (0, 255, 0)
COLOR_GUIDE_TEXT = (0, 255, 0)
COLOR_SEARCHING_TEXT = (0, 165, 255)
THICKNESS_CONTOUR = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    def __init__(self):
        self._detector = DocumentDetector()

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
            if not ret or frame is None:
                break

            display_frame = frame.copy()
            contour_pts = self._detector.find_document_contour(frame)

            if contour_pts is not None:
                pts_int = contour_pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts_int], True, COLOR_CONTOUR, THICKNESS_CONTOUR)
                cv2.putText(
                    display_frame,
                    "Document Detected! Press ENTER",
                    (20, 40),
                    TEXT_FONT,
                    0.8,
                    COLOR_GUIDE_TEXT,
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    "Align document with camera...",
                    (20, 40),
                    TEXT_FONT,
                    0.7,
                    COLOR_SEARCHING_TEXT,
                    2
                )

            cv2.imshow(WINDOW_PREVIEW, display_frame)
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
