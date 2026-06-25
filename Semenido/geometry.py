import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


class PerspectiveCorrector:
    def __init__(
        self,
        grid_size=(7, 7),
        actual_step_mm=35.0,
        ransac_iterations=500,
    ):
        self.grid_size = grid_size
        self.actual_step_mm = actual_step_mm
        self.ransac_iterations = ransac_iterations
        self.ideal_mm_grid = self._generate_grid()
    def _generate_grid(self):
        rows, cols = self.grid_size
        x = np.arange(cols) * self.actual_step_mm
        y = np.arange(rows) * self.actual_step_mm
        xx, yy = np.meshgrid(x, y)
        return np.column_stack(
            [xx.ravel(), yy.ravel()]
        ).astype(np.float32)
    def _build_mapping(
        self,
        transformed_points,
        threshold_mm,
    ):
        cost_matrix = np.linalg.norm(
            self.ideal_mm_grid[:, None, :]
            - transformed_points[None, :, :],
            axis=2,
        )
        rows, cols = linear_sum_assignment(cost_matrix)
        mapping = {}
        for grid_idx, point_idx in zip(rows, cols):
            if cost_matrix[grid_idx, point_idx] <= threshold_mm:
                mapping[grid_idx] = point_idx
        return mapping
    def _estimate_homography_robust(
        self,
        points,
    ):
        if len(points) < 6:
            return None, self.ideal_mm_grid, {}
        threshold_mm = self.actual_step_mm * 0.45
        best_h = None
        best_mapping = {}
        best_inliers = 0
        rng = np.random.default_rng(42)
        tree = cKDTree(points)
        for _ in range(self.ransac_iterations):
            base_idx = rng.integers(len(points))
            _, neighbor_ids = tree.query(
                points[base_idx],
                k=min(12, len(points)),
            )
            if len(np.atleast_1d(neighbor_ids)) < 4:
                continue
            sample_ids = rng.choice(
                np.atleast_1d(neighbor_ids),
                4,
                replace=False,
            )
            sample_pts = points[sample_ids]
            sums = sample_pts.sum(axis=1)
            diffs = np.diff(sample_pts, axis=1).flatten()
            src_quad = np.array(
                [
                    sample_pts[np.argmin(sums)],
                    sample_pts[np.argmin(diffs)],
                    sample_pts[np.argmax(sums)],
                    sample_pts[np.argmax(diffs)],
                ],
                dtype=np.float32,
            )
            for row in range(self.grid_size[0] - 1):
                for col in range(self.grid_size[1] - 1):
                    dst_quad = np.array(
                        [
                            [
                                col * self.actual_step_mm,
                                row * self.actual_step_mm,
                            ],
                            [
                                (col + 1) * self.actual_step_mm,
                                row * self.actual_step_mm,
                            ],
                            [
                                (col + 1) * self.actual_step_mm,
                                (row + 1) * self.actual_step_mm,
                            ],
                            [
                                col * self.actual_step_mm,
                                (row + 1) * self.actual_step_mm,
                            ],
                        ],
                        dtype=np.float32,
                    )
                    h = cv2.getPerspectiveTransform(
                        src_quad,
                        dst_quad,
                    )
                    if abs(np.linalg.det(h)) < 1e-8:
                        continue
                    transformed = cv2.perspectiveTransform(
                        np.array([points]),
                        h,
                    )[0]
                    mapping = self._build_mapping(
                        transformed,
                        threshold_mm,
                    )
                    inliers = len(mapping)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_h = h
                        best_mapping = mapping
        if best_h is None or best_inliers < 12:
            return None, self.ideal_mm_grid, {}
        src_refine = np.array(
            [
                points[idx]
                for idx in best_mapping.values()
            ],
            dtype=np.float32,
        )
        dst_refine = np.array(
            [
                self.ideal_mm_grid[idx]
                for idx in best_mapping.keys()
            ],
            dtype=np.float32,
        )
        refined_h, mask = cv2.findHomography(
            src_refine,
            dst_refine,
            cv2.RANSAC,
            ransacReprojThreshold=self.actual_step_mm * 0.3,
        )
        if refined_h is None:
            return best_h, self.ideal_mm_grid, best_mapping
        transformed = cv2.perspectiveTransform(
            np.array([points]),
            refined_h,
        )[0]
        final_mapping = self._build_mapping(
            transformed,
            threshold_mm,
        )
        return (
            refined_h,
            self.ideal_mm_grid,
            final_mapping,
        )
    def process_and_restore_grid(
        self,
        detected_points,
    ):
        if len(detected_points) < 4:
            return detected_points, []
        points = np.asarray(
            detected_points,
            dtype=np.float32,
        )
        (h_matrix,ideal_grid,mapping) = self._estimate_homography_robust(points)
        if h_matrix is None:
            return detected_points, []
        try:
            h_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError:
            return detected_points, []
        final_pixel_points = []
        final_mm_points = []
        for grid_idx, mm_pt in enumerate(ideal_grid):
            if grid_idx in mapping:
                point_idx = mapping[grid_idx]
                final_pixel_points.append(
                    tuple(
                        map(
                            float,
                            detected_points[point_idx],
                        )
                    )
                )
            else:
                restored = cv2.perspectiveTransform(
                    np.array([[mm_pt]], dtype=np.float32),
                    h_inv,
                )[0][0]
                final_pixel_points.append(
                    (
                        float(restored[0]),
                        float(restored[1]),
                    )
                )
            final_mm_points.append(mm_pt.tolist())
        return (final_pixel_points,final_mm_points,)