import cv2
import numpy as np


LOWER_SKIN_HSV = np.array([0, 20, 70], dtype=np.uint8)
UPPER_SKIN_HSV = np.array([25, 255, 255], dtype=np.uint8)
MORPH_KERNEL_SIZE = (5, 5)
INPAINT_RADIUS = 3


class FingerRemover:
    def detect_skin_mask(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_SKIN_HSV, UPPER_SKIN_HSV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def remove_fingers(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3:
            return image.copy()

        mask = self.detect_skin_mask(image)
        if cv2.countNonZero(mask) == 0:
            return image.copy()

        cleaned = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        return cleaned
