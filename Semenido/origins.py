import numpy as np
from scipy.spatial import cKDTree


class CoordinateSystemDetector:
    def __init__(self, angle_thresh=25.0):
        self.angle_thresh = angle_thresh
    def _build_frame(self, O, X, Y):
        x_axis = X - O
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
        y_axis = Y - O
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        return {
            "origin": O,
            "x_axis": x_axis,
            "y_axis": y_axis,
        }
    def detect(self, detected_points, circled_points, grid_rows=7, grid_cols=7):
        if len(detected_points) == 0 or len(circled_points) == 0:
            return []
        detected_points = np.array(detected_points, dtype=np.float32)
        circled_points = np.array(circled_points, dtype=np.float32)
        grid_tree = cKDTree(detected_points)
        marked_indices = []
        for cp in circled_points:
            dist, idx = grid_tree.query(cp)
            if dist < 30.0:
                row = idx // grid_cols
                col = idx % grid_cols
                marked_indices.append((row, col, idx))
        frames = []
        for o_row, o_col, o_idx in marked_indices:
            O = detected_points[o_idx]
            x_idx = None
            y_idx = None
            for r, c, idx in marked_indices:
                if idx == o_idx:
                    continue
                if abs(r - o_row) + abs(c - o_col) == 1:
                    x_idx = idx
                    x_dir = (r - o_row, c - o_col)
                    break
            if x_idx is None:
                continue
            y_dir_target = (-x_dir[1], x_dir[0])
            y_dir_target_inv = (x_dir[1], -x_dir[0])
            for r, c, idx in marked_indices:
                if idx == o_idx or idx == x_idx:
                    continue
                current_dir = (r - o_row, c - o_col)
                if current_dir == (y_dir_target[0] * 2, y_dir_target[1] * 2) or \
                        current_dir == (y_dir_target_inv[0] * 2, y_dir_target_inv[1] * 2):
                    y_idx = idx
                    break
            if y_idx is None:
                continue
            X = detected_points[x_idx]
            Y = detected_points[y_idx]
            frames.append(self._build_frame(O, X, Y))
        return frames