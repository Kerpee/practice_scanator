import cv2
import numpy as np


class Binarizer:
    def __init__(self, tophat_ksize: int = 9, thresh_value: int = 15,
                 min_area: float = 30.0, max_area: float = 1200.0):
        self.tophat_ksize = tophat_ksize
        self.thresh_value = thresh_value
        self.min_area = min_area
        self.max_area = max_area
    def _crop_to_black_plate(self, img: np.ndarray) -> tuple:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh_plate = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, 0, 0
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x_start = max(0, x + padding)
        y_start = max(0, y + padding)
        x_end = min(img.shape[1], x + w - padding)
        y_end = min(img.shape[0], y + h - padding)
        crop_img = img[y_start:y_end, x_start:x_end]
        return crop_img, x_start, y_start
    def process(self, img: np.ndarray) -> np.ndarray:
        cropped_img, offset_x, offset_y = self._crop_to_black_plate(img)
        gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )
        equalized = clahe.apply(gray_cropped)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.tophat_ksize, self.tophat_ksize)
        )
        tophat = cv2.morphologyEx(
            equalized,
            cv2.MORPH_TOPHAT,
            kernel
        )
        _, binary_cropped = cv2.threshold(
            tophat,
            self.thresh_value,
            255,
            cv2.THRESH_BINARY
        )
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
        binary_cropped = cv2.morphologyEx(
            binary_cropped,
            cv2.MORPH_OPEN,
            morph_kernel
        )
        binary_cropped = cv2.morphologyEx(
            binary_cropped,
            cv2.MORPH_CLOSE,
            morph_kernel
        )
        margin = 30
        binary_cropped[:margin, :] = 0
        binary_cropped[-margin:, :] = 0
        binary_cropped[:, :margin] = 0
        binary_cropped[:, -margin:] = 0
        full_binary_mask = np.zeros(
            img.shape[:2],
            dtype=np.uint8
        )
        full_binary_mask[
            offset_y:offset_y + binary_cropped.shape[0],
            offset_x:offset_x + binary_cropped.shape[1]
        ] = binary_cropped
        filtered_mask = np.zeros_like(full_binary_mask)
        contours, _ = cv2.findContours(
            full_binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if area < self.min_area or area > self.max_area:
                continue
            if not (0.8 <= aspect_ratio <= 1.3):
                continue
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                -1
            )
        return filtered_mask