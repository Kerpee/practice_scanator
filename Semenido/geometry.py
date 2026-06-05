import cv2
import numpy as np


class PerspectiveCorrector:

    def __init__(self, grid_size: tuple = (7, 7), actual_step_mm: float = 35.0):
        self.grid_size = grid_size
        self.actual_step_mm = actual_step_mm

    def process_and_restore_grid(self, detected_points: list) -> tuple:
        total_expected = self.grid_size[0] * self.grid_size[1]
        if len(detected_points) < 8:
            return detected_points, []

        pts = np.array(detected_points, dtype=np.float32)
        ideal_mm_grid = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                ideal_mm_grid.append(
                    [c * self.actual_step_mm, r * self.actual_step_mm]
                )
        ideal_mm_grid = np.array(ideal_mm_grid, dtype=np.float32)
        best_h = None
        best_inliers = -1
        best_mapping = {}
        iterations = 1000
        np.random.seed(42)
        center_img = np.mean(pts, axis=0)
        dists_to_center = np.linalg.norm(pts - center_img, axis=1)
        central_indices = np.argsort(dists_to_center)[: len(pts) // 2]
        if len(central_indices) < 4:
            central_indices = np.arange(len(pts))
        for _ in range(iterations):
            idx = np.random.choice(central_indices, 4, replace=False)
            sample_pts = pts[idx]
            sum_pts = sample_pts.sum(axis=1)
            diff_pts = np.diff(sample_pts, axis=1).flatten()
            src_quad = np.array(
                [
                    sample_pts[np.argmin(sum_pts)],
                    sample_pts[np.argmin(diff_pts)],
                    sample_pts[np.argmax(sum_pts)],
                    sample_pts[np.argmax(diff_pts)],
                ],
                dtype=np.float32,
            )
            # Пробуем сопоставить эту четверку со случайным внутренним квадрантом идеальной сетки
            # (так как мы не знаем, реальные ли это углы всей платы)
            for r_offset in range(self.grid_size[0] - 1):
                for c_offset in range(self.grid_size[1] - 1):
                    dst_quad = np.array(
                        [
                            [
                                c_offset * self.actual_step_mm,
                                r_offset * self.actual_step_mm,
                            ],
                            [
                                (c_offset + 1) * self.actual_step_mm,
                                r_offset * self.actual_step_mm,
                            ],
                            [
                                (c_offset + 1) * self.actual_step_mm,
                                (r_offset + 1) * self.actual_step_mm,
                            ],
                            [
                                c_offset * self.actual_step_mm,
                                (r_offset + 1) * self.actual_step_mm,
                            ],
                        ],
                        dtype=np.float32,
                    )
                    h_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
                    if np.abs(np.linalg.det(h_matrix)) < 1e-6:
                        continue
                    pts_transformed = cv2.perspectiveTransform(
                        np.array([pts]), h_matrix
                    )[0]
                    current_inliers = 0
                    current_mapping = {}
                    threshold_mm = self.actual_step_mm * 0.3
                    for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
                        dists = np.linalg.norm(
                            pts_transformed - ideal_pt, axis=1
                        )
                        min_idx = np.argmin(dists)
                        if dists[min_idx] <= threshold_mm:
                            current_inliers += 1
                            current_mapping[ideal_idx] = min_idx
                    if current_inliers > best_inliers:
                        best_inliers = current_inliers
                        best_h = h_matrix
                        best_mapping = current_mapping
        if best_h is None or best_inliers < 6:
            return detected_points, []
        best_h_inv = np.linalg.inv(best_h)
        final_pixel_points = []
        final_mm_points = []
        for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
            if ideal_idx in best_mapping:
                real_idx = best_mapping[ideal_idx]
                final_pixel_points.append(detected_points[real_idx])
                final_mm_points.append(ideal_pt.tolist())
            else:
                ideal_pt_reshaped = np.array([[ideal_pt]], dtype=np.float32)
                restored_pixel_pt = cv2.perspectiveTransform(
                    ideal_pt_reshaped, best_h_inv
                )[0][0]
                final_pixel_points.append(
                    (float(restored_pixel_pt[0]), float(restored_pixel_pt[1]))
                )
                final_mm_points.append(ideal_pt.tolist())
        return final_pixel_points, final_mm_points