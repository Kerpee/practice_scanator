import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple
from sklearn.cluster import DBSCAN
import os
import csv
import json

class ImagePreprocessor:
    """Модуль препроцессинга изображений калибровочных карт."""
    def __init__(self, clahe_clip: float = 2.0, clahe_grid: Tuple[int, int] = (8, 8),
                 thresh_block_size: int = 21, thresh_c: int = -10):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.thresh_block_size = thresh_block_size
        self.thresh_c = thresh_c

    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает изображение и применяет пайплайн препроцессинга.
        Возвращает оригинальное изображение и бинарную маску меток.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        # 1. Перевод в градации серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Выравнивание гистограммы (CLAHE) для устранения бликов
        enhanced = self.clahe.apply(gray)
        
        # 3. Фильтрация шумов
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 4. Адаптивная бинаризация для выделения светлых крестов
        # Кресты светлые на темном фоне
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, self.thresh_block_size, self.thresh_c
        )
        
        # 5. Морфологические операции для очистки шума
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return img, morph

class BaseDetector(ABC):
    """Абстрактный базовый класс для алгоритмов детектирования меток."""
    
    @abstractmethod
    def detect(self, img: np.ndarray, preprocessed: np.ndarray) -> List[Tuple[float, float]]:
        """
        Метод должен возвращать список координат (x, y) найденных центров меток.
        """
        pass

class ContourGridDetector(BaseDetector):
    """
    Находит метки с помощью контурного анализа (проверка формы) 
    и фильтрует сетку с помощью DBSCAN.
    """
    def __init__(self, eps: float = 15, min_samples: int = 5):
        self.eps = eps
        # min_samples здесь больше используется для параметров старого пайплайна,
        # оставляем для совместимости с json.
        self.min_samples = min_samples

    def detect(self, img: np.ndarray, preprocessed: np.ndarray) -> List[Tuple[float, float]]:
        # Находим контуры на бинарном изображении
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clusters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Базовый фильтр по площади (слишком мелкий шум или огромные засветы убираем)
            if area < 5 or area > 1500:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 1. Проверяем Aspect Ratio (соотношение сторон)
            # У креста оно близко к 1.0. Дадим запас от 0.3 до 3.0
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
                
            # 2. Проверяем Extent (Заполненность)
            # Крест внутри своего описывающего прямоугольника занимает мало места (пустота по углам)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Сплошной квадрат это 1.0. Крест обычно 0.3-0.6.
            # Если это сплошной блик (пятно), extent будет > 0.8.
            if extent > 0.8:
                continue
                
            # Вычисляем центр масс контура (субпиксельная точность)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
                
            clusters.append((cx, cy, area, w, h))

        # --- NMS (Удаление близких дубликатов) ---
        # Оцениваем шаг сетки (35 мм) в пикселях для перевода 2.5 мм в пиксели
        grid_step_pixels = []
        for c in clusters:
            cx, cy, area, w, h = c
            min_dist = float('inf')
            for ec in clusters:
                if c is ec: continue
                ecx, ecy, _, ew, eh = ec
                dist = np.sqrt((cx - ecx)**2 + (cy - ecy)**2)
                # Игнорируем точки, которые лежат внутри/очень близко (скорее всего дубликаты)
                if dist > max(w, h): 
                    if dist < min_dist:
                        min_dist = dist
            if min_dist != float('inf'):
                grid_step_pixels.append(min_dist)
                
        median_step_px = self.eps * 3.0  # Значение по умолчанию
        if grid_step_pixels:
            median_step_px = np.median(grid_step_pixels)
            # Если два креста ближе, чем 0.4 от шага сетки, один из них явно лишний.
            # Ограничиваем снизу безопасным значением, если фото сильно зашумлено.
            min_dist_pixels = max(median_step_px * 0.4, self.eps * 0.8)
        else:
            min_dist_pixels = self.eps * 0.8

        # Сортируем кластеры по близости к медианной площади (идеальным крестам)
        if clusters:
            median_area = np.median([c[2] for c in clusters])
            clusters.sort(key=lambda c: abs(c[2] - median_area))
        
        nms_clusters = []
        for c in clusters:
            cx, cy, area, w, h = c
            too_close = False
            for ec in nms_clusters:
                ecx, ecy, earea, ew, eh = ec
                dist = np.sqrt((cx - ecx)**2 + (cy - ecy)**2)
                if dist < min_dist_pixels:
                    too_close = True
                    break
            if not too_close:
                nms_clusters.append(c)
                
        clusters = nms_clusters
        # ----------------------------------------
        # ----------------------------------------

        from sklearn.cluster import DBSCAN
        from collections import Counter
        
        # Запускаем кластеризацию по строкам и столбцам (DBSCAN), если точек достаточно для построения хотя бы части сетки.
        if len(clusters) > 10:
            points_arr = np.array([[c[0], c[1]] for c in clusters])
            
            # Делаем сетку более щадящей к перспективе: радиус поиска строки равен 75% от шага сетки
            dbscan_eps = max(self.eps * 3.0, median_step_px * 0.75)
            
            # Кластеризуем координаты по оси Y (строки) и X (столбцы)
            dbscan_y = DBSCAN(eps=dbscan_eps, min_samples=2).fit(points_arr[:, 1].reshape(-1, 1))
            dbscan_x = DBSCAN(eps=dbscan_eps, min_samples=2).fit(points_arr[:, 0].reshape(-1, 1))
            
            y_counts = Counter(dbscan_y.labels_)
            x_counts = Counter(dbscan_x.labels_)
            
            # Убираем шум (-1)
            y_counts.pop(-1, None)
            x_counts.pop(-1, None)
            
            # Оставляем только 7 самых заполненных строк и столбцов
            top_y_labels = set([item[0] for item in y_counts.most_common(7)])
            top_x_labels = set([item[0] for item in x_counts.most_common(7)])
            
            # Если сетка реально нашлась (как минимум 4 строки и столбца), 
            # оставляем строго одну точку на каждую ячейку сетки.
            # Это решает проблему разорванных крестов, которые не слились по дистанции.
            if len(top_y_labels) >= 4 and len(top_x_labels) >= 4:
                grid_cells = {}
                for i, c in enumerate(clusters):
                    ly = dbscan_y.labels_[i]
                    lx = dbscan_x.labels_[i]
                    if ly in top_y_labels and lx in top_x_labels:
                        cell = (ly, lx)
                        if cell not in grid_cells:
                            grid_cells[cell] = []
                        grid_cells[cell].append(c)
                
                filtered_clusters = []
                for cell, pts in grid_cells.items():
                    if len(pts) == 1:
                        filtered_clusters.append(pts[0])
                    else:
                        # Если в одну ячейку попало несколько точек (разорванный крест),
                        # оставляем ту, чья площадь ближе к медианной.
                        best_pt = min(pts, key=lambda p: abs(p[2] - median_area))
                        filtered_clusters.append(best_pt)
            else:
                # Если сетка не нашлась (сплошной шум), просто берем точки
                filtered_clusters = []
                for i, c in enumerate(clusters):
                    if dbscan_y.labels_[i] in top_y_labels and dbscan_x.labels_[i] in top_x_labels:
                        filtered_clusters.append(c)
            
            # Если все равно осталось больше, отсекаем по площади (ближе к медианной)
            if len(filtered_clusters) > 49:
                filtered_clusters.sort(key=lambda c: abs(c[2] - median_area))
                filtered_clusters = filtered_clusters[:49]
                
            clusters = filtered_clusters

        # Сортируем итоговые кресты сверху-вниз (по Y), а внутри строк - слева-направо (по X)
        clusters.sort(key=lambda c: c[1])
        
        final_centers = []
        if len(clusters) > 0:
            current_row = [clusters[0]]
            for c in clusters[1:]:
                # Если отклонение по Y небольшое, значит это та же самая строка
                if abs(c[1] - current_row[-1][1]) < self.eps * 3.0:
                    current_row.append(c)
                else:
                    # Строка закончилась, сортируем внутри нее по X
                    current_row.sort(key=lambda p: p[0])
                    final_centers.extend([(p[0], p[1]) for p in current_row])
                    current_row = [c]
            
            current_row.sort(key=lambda p: p[0])
            final_centers.extend([(p[0], p[1]) for p in current_row])
            
        return final_centers

class LocalizationEvaluator:
    """Модуль оценки точности локализации."""
    @staticmethod
    def calculate_error(points: List[Tuple[float, float]]) -> float:
        """
        Упрощенная оценка: среднеквадратичное отклонение дистанций
        до ближайших соседей (оцениваем равномерность сетки).
        """
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

class ResultVisualizer:
    """Модуль визуализации результатов."""
    @staticmethod
    def draw_results(img: np.ndarray, points: List[Tuple[float, float]], output_path: str):
        result_img = img.copy()
        for x, y in points:
            cv2.circle(result_img, (int(x), int(y)), 2, (0, 0, 255), -1)
            size = 15
            top_left = (int(x - size), int(y - size))
            bottom_right = (int(x + size), int(y + size))
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 1)
            
        is_success, im_buf_arr = cv2.imencode(".jpg", result_img)
        if is_success:
            im_buf_arr.tofile(output_path)
            print(f"Результат сохранен в {output_path}")
        else:
            print(f"Ошибка сохранения в {output_path}")



class DataExporter:
    """Модуль экспорта результатов."""
    @staticmethod
    def export_to_csv(points: List[Tuple[float, float]], output_path: str):
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for p in points:
                writer.writerow([f"{p[0]:.2f}", f"{p[1]:.2f}"])
        print(f"Координаты сохранены в {output_path}")

class ScannerCalibrationApp:
    """Главный класс приложения."""
    def __init__(self, params_file: str = "best_params.json"):
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.params = {}
        if os.path.exists(params_file):
            with open(params_file, 'r', encoding='utf-8') as f:
                self.params = json.load(f)
                print(f"Загружены оптимизированные параметры: {self.params}")
                
        # Настраиваем препроцессор на основе загруженных параметров
        prep_params = {}
        if 'thresh_block_size' in self.params:
            prep_params['thresh_block_size'] = self.params['thresh_block_size']
        if 'thresh_c' in self.params:
            prep_params['thresh_c'] = self.params['thresh_c']
            
        self.preprocessor = ImagePreprocessor(**prep_params)
        self.visualizer = ResultVisualizer()
        self.evaluator = LocalizationEvaluator()
        
    def run(self, image_path: str):
        print(f"Начало обработки: {image_path}")
        start_time = time.time()
        
        img, preprocessed = self.preprocessor.process(image_path)
        
        ml_params = {}
        if 'eps' in self.params: ml_params['eps'] = self.params['eps']
        if 'min_samples' in self.params: ml_params['min_samples'] = self.params['min_samples']
        detector = ContourGridDetector(**ml_params)
        print("Используется Contour метод (Контуры + Фильтрация формы)")
            
        points_ml = detector.detect(img, preprocessed)
        
        process_time = time.time() - start_time
        
        print(f"Найдено меток: {len(points_ml)} (Ожидается: 49 / 7x7)")
        if len(points_ml) != 49:
            print("ВНИМАНИЕ: Количество найденных меток отличается от ожидаемого (49)!")
            
        print(f"Время обработки: {process_time:.3f} сек")
        
        error = self.evaluator.calculate_error(points_ml)
        print(f"Оценка ошибки локализации (STD расстояний сетки): {error:.4f} пикс.")
        
        base_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        
        term_dir = os.path.join(base_dir, "term", self.timestamp)
        os.makedirs(term_dir, exist_ok=True)
        output_path = os.path.join(term_dir, base_name.replace(".jpg", "_result_contour.jpg"))
        
        self.visualizer.draw_results(img, points_ml, output_path)
        
        cords_dir = os.path.join(base_dir, "cords", self.timestamp)
        os.makedirs(cords_dir, exist_ok=True)
        csv_path = os.path.join(cords_dir, base_name.replace(".jpg", "_coords_ml.csv"))
        
        DataExporter.export_to_csv(points_ml, csv_path)

if __name__ == "__main__":
    import glob
    app = ScannerCalibrationApp()
    
    # use dynamic path from __file__
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    images = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not images:
        print(f"Не найдено изображений в папке {data_dir}")
        
    for image_file in images:
        print(f"\n{'=' * 50}\nОбработка файла: {os.path.basename(image_file)}\n{'=' * 50}")
        app.run(image_file)
