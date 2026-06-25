import cv2
import numpy as np


class Binarizer:

    def __init__(self):
        pass

    def _normalize_and_enhance(self, gray_img: np.ndarray) -> np.ndarray:
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_img)
        if max_val - min_val > 2:
            gray_img = cv2.normalize(
                gray_img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        return clahe.apply(gray_img)
    def _crop_to_black_plate(self, img: np.ndarray) -> tuple:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh_plate = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img, 0, 0, img.shape[0] * img.shape[1]
        largest_contour = max(contours, key=cv2.contourArea)
        plate_area = cv2.contourArea(largest_contour)
        if plate_area < (img.shape[0] * img.shape[1] * 0.15):
            return img, 0, 0, img.shape[0] * img.shape[1]
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = max(2, int(w * 0.01))
        x_start = max(0, x + padding)
        y_start = max(0, y + padding)
        x_end = min(img.shape[1], x + w - padding)
        y_end = min(img.shape[0], y + h - padding)
        return img[y_start:y_end, x_start:x_end], x_start, y_start, plate_area
    def process(self, img: np.ndarray) -> np.ndarray:
        cropped_img, offset_x, offset_y, plate_area = self._crop_to_black_plate(img)
        gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        equalized = self._normalize_and_enhance(gray_cropped)
        img_diagonal = np.sqrt(cropped_img.shape[0] ** 2 + cropped_img.shape[1] ** 2)
        tophat_ksize = int(img_diagonal * 0.018)
        if tophat_ksize % 2 == 0:
            tophat_ksize += 1
        tophat_ksize = max(9, tophat_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_ksize, tophat_ksize))
        tophat = cv2.morphologyEx(equalized, cv2.MORPH_TOPHAT, kernel)
        _, binary_cropped = cv2.threshold(
            tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Размер очищающего ядра зависит от масштаба: 0.2% от диагонали
        morph_size = max(3, int(img_diagonal * 0.002) | 1)  # гарантируем нечетность через побитовое ИЛИ
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        binary_cropped = cv2.dilate(binary_cropped, morph_kernel, iterations=1)
        binary_cropped = cv2.morphologyEx(binary_cropped, cv2.MORPH_OPEN, morph_kernel)
        full_binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_binary_mask[
            offset_y: offset_y + binary_cropped.shape[0],
            offset_x: offset_x + binary_cropped.shape[1],
        ] = binary_cropped
        filtered_mask = np.zeros_like(full_binary_mask)
        contours, _ = cv2.findContours(
            full_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = plate_area * 0.00005
        max_area = plate_area * 0.015
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if not (0.4 <= aspect_ratio <= 2.5):
                continue
            extent = area / float(w * h)
            if extent > 0.85:
                continue
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        return filtered_mask