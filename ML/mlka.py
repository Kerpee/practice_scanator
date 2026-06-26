"""
Скрипт mlka.py
--------------
Основной модуль логики машинного обучения. Содержит:
- HybridYoloDetector: класс для первичного поиска крестов (YOLO) и уточнения их центра (OpenCV Moments).
- ResultVisualizer: класс для отрисовки найденных точек и рамок на изображении.
- LocalizationEvaluator: класс для вычисления метрики ошибки (стандартного отклонения расстояний).
- ScannerCalibrationApp: основное приложение, объединяющее пайплайн распознавания и сохранения.
"""

import cv2
import numpy as np
import time
import os
import csv
from ultralytics import YOLO
from typing import List, Tuple

class HybridYoloDetector:
    """
    Гибридный детектор:
    1. YOLOv8 для грубой локализации (находит рамку крестика).
    2. OpenCV Moments для нахождения точного субпиксельного центра внутри рамки.
    """
    def __init__(self, model_path: str = 'model.pt', conf: float = 0.25):
        if model_path == 'model.pt':
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель YOLO не найдена по пути: {model_path}. Сначала дождитесь окончания обучения!")
        
        # Отключаем лишний вывод от YOLO в консоль
        self.model = YOLO(model_path)
        self.conf = conf

    def _find_exact_center(self, patch: np.ndarray, x1: int, y1: int) -> Tuple[float, float]:
        """Классический CV алгоритм (Шаг 4) внутри квадратика для поиска точного центра масс."""
        # 1. Переводим вырезанный квадратик в ЧБ
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # 2. Применяем пороговое преобразование (Threshold)
        # Так как кресты могут быть и светлые, и темные, используем адаптивный порог.
        # THRESH_BINARY_INV сделает кресты белыми на черном фоне.
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 3. Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: геометрический центр рамки, если контур почему-то не найден
            h, w = patch.shape[:2]
            return x1 + w / 2.0, y1 + h / 2.0
            
        # Берем самый большой контур (скорее всего это и есть крест)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 4. Вычисляем центр масс (моменты)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx_local = M["m10"] / M["m00"]
            cy_local = M["m01"] / M["m00"]
        else:
            h, w = patch.shape[:2]
            cx_local, cy_local = w / 2.0, h / 2.0
            
        # 5. Перевод из локальных координат квадратика в глобальные координаты большой картинки
        return float(x1 + cx_local), float(y1 + cy_local)

    def detect(self, img: np.ndarray) -> List[Tuple[float, float]]:
        # Шаг 1: Прогон через YOLO
        max_size = max(img.shape[0], img.shape[1])
        results = self.model.predict(img, conf=self.conf, imgsz=max_size, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy() # уверенность модели
        
        # Сортируем по уверенности и берем топ-49 самых "четких" крестов
        if len(boxes) > 49:
            sorted_indices = np.argsort(confs)[::-1]
            boxes = boxes[sorted_indices[:49]]
            
        points = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Немного расширяем рамку для уверенности (чтобы крест не оказался на краю квадратика)
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.shape[1], x2 + pad)
            y2 = min(img.shape[0], y2 + pad)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            patch = img[y1:y2, x1:x2]
            
            # Шаг 2: Алгоритм внутри квадратика (CV)
            exact_point = self._find_exact_center(patch, x1, y1)
            points.append(exact_point)
            
        return points

class ResultVisualizer:
    @staticmethod
    def draw_results(img: np.ndarray, points: List[Tuple[float, float]], output_path: str):
        result_img = img.copy()
        for x, y in points:
            # Рисуем субпиксельный центр
            cv2.circle(result_img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
            # Рисуем зеленую рамку (показывая зону)
            size = 15
            top_left = (int(round(x - size)), int(round(y - size)))
            bottom_right = (int(round(x + size)), int(round(y + size)))
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 1)
            
        is_success, im_buf_arr = cv2.imencode(".jpg", result_img)
        if is_success:
            im_buf_arr.tofile(output_path)

class LocalizationEvaluator:
    @staticmethod
    def calculate_error(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        pts = np.array(points)
        errors = []
        for i, p1 in enumerate(pts):
            distances = []
            for j, p2 in enumerate(pts):
                if i != j:
                    distances.append(np.linalg.norm(p1 - p2))
            if distances:
                distances.sort()
                nearest_4 = distances[:4]
                std_dist = np.std(nearest_4)
                errors.append(std_dist)
        return float(np.mean(errors))

class ScannerCalibrationApp:
    def __init__(self):
        try:
            self.detector = HybridYoloDetector()
        except FileNotFoundError as e:
            print(f"Ошибка инициализации: {e}")
            self.detector = None
            
        self.visualizer = ResultVisualizer()
        self.evaluator = LocalizationEvaluator()
        
    def run(self, image_path: str, output_img: str = None, output_csv: str = None, output_json: str = None) -> dict:
        if not self.detector:
            raise RuntimeError("YOLO Модель не загружена. Дождитесь конца обучения!")
            
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        start_time = time.time()
        
        # Получаем точные субпиксельные точки (YOLO + Moments)
        points = self.detector.detect(img)
        
        process_time = time.time() - start_time
        error = self.evaluator.calculate_error(points)
        
        base_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        
        if output_img is None:
            term_dir = os.path.join(base_dir, "term")
            os.makedirs(term_dir, exist_ok=True)
            output_img = os.path.join(term_dir, base_name.replace(".jpg", f"_hybrid.jpg"))
            
        if output_csv is None:
            cords_dir = os.path.join(base_dir, "cords")
            os.makedirs(cords_dir, exist_ok=True)
            output_csv = os.path.join(cords_dir, base_name.replace(".jpg", f"_coords_hybrid.csv"))
            
        if output_json is None:
            output_json = output_csv.replace('.csv', '.json')
            
        self.visualizer.draw_results(img, points, output_img)
        
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            # Сохраняем с точностью до 4 знаков после запятой (субпиксельная точность!)
            for p in points:
                writer.writerow([f"{p[0]:.4f}", f"{p[1]:.4f}"])
                
        import datetime
        import json
        json_data = {
            "image": base_name,
            "mode": "hybrid",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "points_found": len(points),
            "points": [[round(p[0], 4), round(p[1], 4)] for p in points],
            "processing_time": round(process_time, 4),
            "error_std": round(error, 4)
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
                
        return {
            'found': len(points) == 49,
            'points_count': len(points),
            'time': process_time,
            'error': error
        }

if __name__ == "__main__":
    app = ScannerCalibrationApp()
    # Тестовый запуск
    # app.run("data/photo.jpg")
