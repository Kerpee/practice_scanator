import cv2
import numpy as np


class PcaDetector:

    def __init__(self, roi_padding: int = 5):
        self.roi_padding = roi_padding
    def _compute_center_pca(
        self, roi_mask: np.ndarray, offset_x: int, offset_y: int
    ) -> tuple:
        indices = np.argwhere(roi_mask == 255)
        if len(indices) == 0:
            return None
        pts = indices[:, [1, 0]].astype(np.float64)
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        mean, _, _ = cv2.PCACompute2(pts, mean=np.empty((0)))
        return mean[0, 0], mean[0, 1]
    def detect_points(self, binary_mask: np.ndarray) -> list:
        detected_centers = []
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h_img, w_img = binary_mask.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(0, x - self.roi_padding)
            y1 = max(0, y - self.roi_padding)
            x2 = min(w_img, x + w + self.roi_padding)
            y2 = min(h_img, y + h + self.roi_padding)

            roi = binary_mask[y1:y2, x1:x2]
            center = self._compute_center_pca(roi, x1, y1)
            if center is not None:
                detected_centers.append(center)
        return detected_centers