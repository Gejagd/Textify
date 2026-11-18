import cv2
import numpy as np


GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C_CONSTANT = 2
MORPH_KERNEL_SIZE = (3, 3)
MORPH_ITERATIONS = 1


class ImageProcessor:
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
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
        gray = self.to_grayscale(image)
        blurred = self.denoise(gray)
        binary = self.binarize(blurred)
        result = self.clean_noise(binary)
        return result
