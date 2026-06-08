import cv2
import numpy as np


class PcaDetector:

    def __init__(self, roi_padding: int = 4):
        self.roi_padding = roi_padding

    def _find_cross_intersection(self, gray_img: np.ndarray, initial_center: tuple) -> tuple:
        cx, cy = initial_center
        h_img, w_img = gray_img.shape[:2]
        win_size = 4
        x_start = max(0, int(round(cx)) - win_size)
        y_start = max(0, int(round(cy)) - win_size)
        x_end = min(w_img, int(round(cx)) + win_size + 1)
        y_end = min(h_img, int(round(cy)) + win_size + 1)
        roi = gray_img[y_start:y_end, x_start:x_end]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return initial_center
        _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = np.argwhere(roi_bin == 255)
        if len(pts) < 5:
            return initial_center
        pts_global = np.zeros((len(pts), 2), dtype=np.float32)
        pts_global[:, 0] = pts[:, 1] + x_start
        pts_global[:, 1] = pts[:, 0] + y_start
        vert_pixels = pts_global[np.abs(pts_global[:, 0] - cx) < np.abs(pts_global[:, 1] - cy)]
        horiz_pixels = pts_global[np.abs(pts_global[:, 0] - cx) >= np.abs(pts_global[:, 1] - cy)]
        if len(vert_pixels) < 2 or len(horiz_pixels) < 2:
            return initial_center
        # Аппроксимируем вертикальную линию: x = k1 * y + b1
        [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(vert_pixels, cv2.DIST_L2, 0, 0.01, 0.01)
        # Аппроксимируем горизонтальную линию: y = k2 * x + b2
        [vx_h, vy_h, x0_h, y0_h] = cv2.fitLine(horiz_pixels, cv2.DIST_L2, 0, 0.01, 0.01)
        v_xv, v_yv, v_x0, v_y0 = vx_v.item(), vy_v.item(), x0_v.item(), y0_v.item()
        h_xv, h_yv, h_x0, h_y0 = vx_h.item(), vy_h.item(), x0_h.item(), y0_h.item()
        A1 = -v_yv
        B1 = v_xv
        C1 = v_yv * v_x0 - v_xv * v_y0
        A2 = -h_yv
        B2 = h_xv
        C2 = h_yv * h_x0 - h_xv * h_y0
        D = A1 * B2 - B1 * A2
        if np.abs(D) > 1e-5:
            intersect_x = (B1 * C2 - C1 * B2) / D
            intersect_y = (C1 * A2 - A1 * C2) / D
            if np.hypot(intersect_x - cx, intersect_y - cy) < 4.0:
                return float(intersect_x), float(intersect_y)

        return initial_center
    def detect_points(self, binary_mask: np.ndarray, gray_img: np.ndarray = None) -> list:
        if gray_img is None:
            gray_img = binary_mask

        h_img, w_img = binary_mask.shape[:2]
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rough_centers = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bx <= 3 or by <= 3 or (bx + bw) >= w_img - 3 or (by + bh) >= h_img - 3:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            rough_centers.append((cx, cy))
        precise_centers = []
        for rc in rough_centers:
            precise_pt = self._find_cross_intersection(gray_img, rc)
            precise_centers.append(precise_pt)
        detected_centers = []
        for px, py in precise_centers:
            padding = 6
            x_start = max(0, int(round(px)) - padding)
            y_start = max(0, int(round(py)) - padding)
            x_end = min(w_img, int(round(px)) + padding + 1)
            y_end = min(h_img, int(round(py)) + padding + 1)
            roi = gray_img[y_start:y_end, x_start:x_end]
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                continue
            dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
            Ixx = np.sum(dx ** 2)
            Iyy = np.sum(dy ** 2)
            Ixy = np.sum(dx * dy)
            H = np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=np.float32)
            eigenvalues = np.linalg.eigvals(H)
            lambda1, lambda2 = max(eigenvalues), min(eigenvalues)
            if lambda1 == 0 or (lambda2 / lambda1) < 0.10:
                continue
            detected_centers.append((px, py))
        print(f"\n[DEBUG] Найдено точек методом пересечения осей: {len(detected_centers)}")
        for idx, (px, py) in enumerate(detected_centers):
            crop_w, crop_h = 24, 24
            x_start = max(0, int(round(px)) - crop_w // 2)
            y_start = max(0, int(round(py)) - crop_h // 2)
            x_end = min(w_img, x_start + crop_w)
            y_end = min(h_img, y_start + crop_h)

            roi_gray = gray_img[y_start:y_end, x_start:x_end]
            if roi_gray is None or roi_gray.size == 0 or roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
                continue

            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            local_cx = px - x_start
            local_cy = py - y_start

            display_size = 480
            roi_resized = cv2.resize(roi_color, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

            scale = display_size / float(roi_gray.shape[1])
            view_cx = int(round(local_cx * scale))
            view_cy = int(round(local_cy * scale))

            cv2.circle(roi_resized, (view_cx, view_cy), 2, (0, 0, 255), -1)
            cv2.line(roi_resized, (view_cx - 8, view_cy), (view_cx + 8, view_cy), (0, 0, 255), 1)
            cv2.line(roi_resized, (view_cx, view_cy - 8), (view_cx, view_cy + 8), (0, 0, 255), 1)

            cv2.imshow(f"Line Intersect - {idx + 1}", roi_resized)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                break
        return detected_centers