import cv2
import numpy as np


class PerspectiveCorrector:

    def __init__(self, grid_size: tuple = (7, 7), actual_step_mm: float = 35.0):
        self.grid_size = grid_size
        self.actual_step_mm = actual_step_mm

    def _estimate_homography_robust(self, points: np.ndarray) -> tuple:
        total_points = self.grid_size[0] * self.grid_size[1]
        # 1. Генерируем эталонную физическую сетку в мм (49 точек)
        ideal_mm_grid = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                ideal_mm_grid.append(
                    [c * self.actual_step_mm, r * self.actual_step_mm]
                )
        ideal_mm_grid = np.array(ideal_mm_grid, dtype=np.float32)
        if len(points) < 6:
            return None, ideal_mm_grid, {}
        best_h = None
        best_inliers_count = -1
        best_mapping = {}
        threshold_mm = self.actual_step_mm * 0.45
        # Нам нужно выбрать 4 точки из кадра и предположить, какому внутреннему
        # четырехугольнику в миллиметрах они соответствуют.
        # Чтобы не перебирать все 49 точек, мы отсортируем их по близости,
        # выбирая локальные группы соседних точек.
        np.random.seed(42)
        iterations = 300
        for _ in range(iterations):
            # Выбираем случайную базовую точку
            base_idx = np.random.choice(len(points))
            base_pt = points[base_idx]
            # Находим ближайших соседей к этой точке
            dists_to_base = np.linalg.norm(points - base_pt, axis=1)
            closest_indices = np.argsort(dists_to_base)[:12]  # Берем локальный кластер из 12 точек
            if len(closest_indices) < 4:
                continue
            # Выбираем 4 случайные точки из этого локального кластера
            idx = np.random.choice(closest_indices, 4, replace=False)
            sample_pts = points[idx]
            # Упорядочиваем выбранные 4 пиксельные точки
            sum_pts = sample_pts.sum(axis=1)
            diff_pts = np.diff(sample_pts, axis=1).flatten()
            src_quad = np.array([
                sample_pts[np.argmin(sum_pts)],
                sample_pts[np.argmin(diff_pts)],
                sample_pts[np.argmax(sum_pts)],
                sample_pts[np.argmax(diff_pts)]
            ], dtype=np.float32)
            # Пробуем сопоставить этот квадрант со ВСЕМИ возможными дискретными
            # позициями на физической доске 7х7
            for r_offset in range(self.grid_size[0] - 1):
                for c_offset in range(self.grid_size[1] - 1):
                    dst_quad = np.array([
                        [c_offset * self.actual_step_mm, r_offset * self.actual_step_mm],
                        [(c_offset + 1) * self.actual_step_mm, r_offset * self.actual_step_mm],
                        [(c_offset + 1) * self.actual_step_mm, (r_offset + 1) * self.actual_step_mm],
                        [c_offset * self.actual_step_mm, (r_offset + 1) * self.actual_step_mm]
                    ], dtype=np.float32)
                    # Вычисляем пробную гипотезу гомографии
                    h_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
                    if np.abs(np.linalg.det(h_matrix)) < 1e-6:
                        continue
                    # Проецируем все сырые точки кадра в мм по этой гипотезе
                    pts_transformed = cv2.perspectiveTransform(np.array([points]), h_matrix)[0]
                    # Считаем, сколько точек попало в структуру сетки
                    current_inliers = 0
                    current_mapping = {}
                    for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
                        dists = np.linalg.norm(pts_transformed - ideal_pt, axis=1)
                        min_idx = np.argmin(dists)
                        if dists[min_idx] <= threshold_mm:
                            current_inliers += 1
                            current_mapping[ideal_idx] = min_idx
                    # Если гипотеза объясняет больше точек, чем предыдущие — сохраняем её
                    if current_inliers > best_inliers_count:
                        best_inliers_count = current_inliers
                        best_h = h_matrix
                        best_mapping = current_mapping
        # Если мы не нашли устойчивую структуру (хотя бы треть сетки должна совпасть)
        if best_h is None or best_inliers_count < 12:
            return None, ideal_mm_grid, {}
        # На основе лучшей гипотезы мы собираем все совпавшие пары и пересчитываем
        # финальную гомографию через точный метод наименьших квадратов
        src_refine = []
        dst_refine = []
        for ideal_idx, real_idx in best_mapping.items():
            src_refine.append(points[real_idx])
            dst_refine.append(ideal_mm_grid[ideal_idx])
        src_refine = np.array(src_refine, dtype=np.float32)
        dst_refine = np.array(dst_refine, dtype=np.float32)
        refined_h, _ = cv2.findHomography(src_refine, dst_refine, 0)

        if refined_h is not None and np.abs(np.linalg.det(refined_h)) > 1e-6:
            final_mapping = {}
            pts_transformed = cv2.perspectiveTransform(np.array([points]), refined_h)[0]
            for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
                dists = np.linalg.norm(pts_transformed - ideal_pt, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] <= threshold_mm:
                    final_mapping[ideal_idx] = min_idx
            return refined_h, ideal_mm_grid, final_mapping

        return best_h, ideal_mm_grid, best_mapping


    def process_and_restore_grid(self, detected_points: list) -> tuple:
        if len(detected_points) < 4:
            return detected_points, []
        pts = np.array(detected_points, dtype=np.float32)
        h_matrix, ideal_mm_grid, mapping = self._estimate_homography_robust(pts)
        if h_matrix is None:
            return detected_points, []
        try:
            h_matrix_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError:
            return detected_points, []
        final_pixel_points = []
        final_mm_points = []
        for ideal_idx, ideal_pt in enumerate(ideal_mm_grid):
            if ideal_idx in mapping:
                real_idx = mapping[ideal_idx]
                final_pixel_points.append(detected_points[real_idx])
                final_mm_points.append(ideal_pt.tolist())
            else:
                ideal_pt_reshaped = np.array([[ideal_pt]], dtype=np.float32)
                restored_pixel_pt = cv2.perspectiveTransform(
                    ideal_pt_reshaped, h_matrix_inv
                )[0][0]

                final_pixel_points.append(
                    (float(restored_pixel_pt[0]), float(restored_pixel_pt[1]))
                )
                final_mm_points.append(ideal_pt.tolist())
        return final_pixel_points, final_mm_points