import cv2
import numpy as np
from scipy.spatial import cKDTree


def detect_circled_points(binary_mask, grid_points):
    if len(grid_points) == 0:
        return []
    grid_points = np.array(grid_points, dtype=np.float32)
    gray = binary_mask.copy()

    contours, _ = cv2.findContours(
        gray,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    debug_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    grid_tree = cKDTree(grid_points)
    matched_grid_centers = {}
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist, idx = grid_tree.query([cx, cy])
        if dist < 30.0:
            grid_pt = tuple(grid_points[idx])
            matched_grid_centers[grid_pt] = area
            cv2.drawContours(debug_vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(debug_vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        else:
            cv2.drawContours(debug_vis, [cnt], -1, (255, 0, 255), 2)
    unique_centers = list(matched_grid_centers.keys())
    print(f"DEBUG: Итоговых уникальных маркеров для СК: {len(unique_centers)}")
    return unique_centers