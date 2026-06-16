import cv2
import numpy as np


class PcaDetector:

    def __init__(self, roi_size: int = 15, min_corner_ratio: float = 0.12):
        self.roi_size = roi_size
        self.min_corner_ratio = min_corner_ratio

    def _refine_subpixel(self, gray_img: np.ndarray, point: tuple) -> tuple:
        corner = np.array([[point]], dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        try:
            cv2.cornerSubPix(
                gray_img,
                corner,
                (5, 5),
                (-1, -1),
                criteria,
            )
            return tuple(corner[0, 0])
        except cv2.error:
            return point
    def _is_cross_like(self, gray_img: np.ndarray, point: tuple) -> bool:
        px, py = point
        half = self.roi_size // 2
        x1 = max(0, int(px) - half)
        y1 = max(0, int(py) - half)
        x2 = min(gray_img.shape[1], int(px) + half + 1)
        y2 = min(gray_img.shape[0], int(py) + half + 1)
        roi = gray_img[y1:y2, x1:x2]
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            return False
        roi = roi.astype(np.float32)
        dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        H = np.array(
            [
                [np.sum(dx * dx), np.sum(dx * dy)],
                [np.sum(dx * dy), np.sum(dy * dy)],
            ],
            dtype=np.float32,
        )
        eigenvalues = np.linalg.eigvalsh(H)
        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[1]
        if lambda_max < 1e-6:
            return False
        return (lambda_min / lambda_max) >= self.min_corner_ratio
    def detect_points(
        self,
        binary_mask: np.ndarray,
        gray_img: np.ndarray = None,
    ) -> list:

        if gray_img is None:
            gray_img = binary_mask

        h_img, w_img = binary_mask.shape[:2]

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        detected_points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (
                x <= 2
                or y <= 2
                or x + w >= w_img - 2
                or y + h >= h_img - 2
            ):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            refined_point = self._refine_subpixel(gray_img, (cx, cy))
            if not self._is_cross_like(gray_img, refined_point):
                continue
            detected_points.append(
                (
                    float(refined_point[0]),
                    float(refined_point[1]),
                )
            )
        return detected_points