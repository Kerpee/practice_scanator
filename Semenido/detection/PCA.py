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

    def detect_single_cross_center(self, gray_img: np.ndarray):
        if gray_img is None:
            return None
        img = gray_img.astype(np.float32)
        dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx * dx + dy * dy)
        if np.max(mag) < 1e-6:
            return None
        mask = mag > (0.6 * np.max(mag))
        if np.sum(mask) < 50:
            return None
        ys, xs = np.where(mask)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        if len(pts) < 20:
            return None
        dx_v = dx[mask]
        dy_v = dy[mask]
        vert_mask = np.abs(dx_v) > np.abs(dy_v)
        horiz_mask = ~vert_mask
        vert_pts = pts[vert_mask]
        horiz_pts = pts[horiz_mask]
        if len(vert_pts) < 10 or len(horiz_pts) < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            vert_pts, cv2.DIST_L2, 0, 0.01, 0.01
        )
        hx, hy, x1, y1 = cv2.fitLine(
            horiz_pts, cv2.DIST_L2, 0, 0.01, 0.01
        )
        vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
        hx, hy, x1, y1 = hx.item(), hy.item(), x1.item(), y1.item()
        A1, B1 = -vy, vx
        C1 = -(A1 * x0 + B1 * y0)
        A2, B2 = -hy, hx
        C2 = -(A2 * x1 + B2 * y1)
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-6:
            return None
        cx = (B1 * C2 - B2 * C1) / det
        cy = (C1 * A2 - C2 * A1) / det
        return float(cx), float(cy)