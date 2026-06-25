import cv2
import numpy as np


class Visualizer:

    def __init__(
        self,
        marker_color: tuple = (0, 255, 0),
        radius: int = 5,
        thickness: int = 2,
    ):
        self.marker_color = marker_color
        self.radius = radius
        self.thickness = thickness
        self.dot_color = (0, 0, 255)

    def draw_detected_points(self, img: np.ndarray, points: list) -> np.ndarray:
        vis_img = img.copy()

        for pt in points:
            center_x = int(round(pt[0]))
            center_y = int(round(pt[1]))
            cv2.circle(
                vis_img,
                (center_x, center_y),
                max(1, int(self.radius * 0.3)),
                self.dot_color,
                -1
            )

        return vis_img