import cv2
import numpy as np


CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_KERNEL = (5, 5)
APPROX_EPSILON_RATIO = 0.02
MIN_CONTOUR_AREA_RATIO = 0.05
MAX_SEARCH_CONTOURS = 5
BORDER_POINTS_COUNT = 4


class DocumentDetector:
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def find_document_contour(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        total_area = float(height * width)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        checked_count = min(len(sorted_contours), MAX_SEARCH_CONTOURS)

        for i in range(checked_count):
            c = sorted_contours[i]
            area = cv2.contourArea(c)
            if area < (total_area * MIN_CONTOUR_AREA_RATIO):
                break

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, APPROX_EPSILON_RATIO * peri, True)

            if len(approx) == BORDER_POINTS_COUNT:
                pts = approx.reshape(4, 2)
                return self.order_points(pts)

        return None

    def warp_perspective(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
        width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        max_width = max(int(width_a), int(width_b))

        height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
        height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
        max_height = max(int(height_a), int(height_b))

        if max_width <= 0 or max_height <= 0:
            return image.copy()

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped
