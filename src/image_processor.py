import cv2
import numpy as np
from document_detector import DocumentDetector
from finger_remover import FingerRemover


GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C_CONSTANT = 2
MORPH_KERNEL_SIZE = (3, 3)
MORPH_ITERATIONS = 1


class ImageProcessor:
    def __init__(self):
        self._detector = DocumentDetector()
        self._finger_remover = FingerRemover()

    def crop_document(self, image: np.ndarray) -> np.ndarray:
        contour_pts = self._detector.find_document_contour(image)
        if contour_pts is None:
            return image.copy()
        return self._detector.warp_perspective(image, contour_pts)

    def remove_fingers(self, image: np.ndarray) -> np.ndarray:
        return self._finger_remover.remove_fingers(image)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    def binarize(self, blurred: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C_CONSTANT
        )

    def clean_noise(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, MORPH_KERNEL_SIZE
        )
        opened = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS
        )
        return opened

    def process(self, image: np.ndarray) -> np.ndarray:
        cropped = self.crop_document(image)
        finger_cleaned = self.remove_fingers(cropped)
        gray = self.to_grayscale(finger_cleaned)
        blurred = self.denoise(gray)
        binary = self.binarize(blurred)
        result = self.clean_noise(binary)
        return result
