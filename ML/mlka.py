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

class ClusteringMLDetector(BaseDetector):
    """
    ML метод (Unsupervised Learning) на основе кластеризации DBSCAN.
    Кластеризует пиксели бинарной маски и находит их центроиды.
    Улучшен фильтрацией по размеру кластера, чтобы отсеивать шум и блики.
    """
    def __init__(self, eps: float = 15, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def detect(self, img: np.ndarray, preprocessed: np.ndarray) -> List[Tuple[float, float]]:
        # Получаем координаты всех белых пикселей (y, x)
        y_coords, x_coords = np.where(preprocessed == 255)
        
        if len(y_coords) == 0:
            return []
            
        # Формируем массив признаков для кластеризации
        pixels = np.column_stack((x_coords, y_coords))
        
        # Применяем DBSCAN
        labels = self.model.fit_predict(pixels)
        
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                # Игнорируем шум
                continue
                
            # Выбираем пиксели текущего кластера
            cluster_pixels = pixels[labels == label]
            
            # Фильтрация по размеру кластера (площадь в пикселях)
            area = len(cluster_pixels)
            # Крест не может состоять из 3 пикселей (скорее всего шум) 
            # и не может занимать 1000 пикселей (скорее всего большой блик)
            if area < 5 or area > 500:
                continue
            
            # Вычисляем центроид (субпиксельная точность)
            cx = np.mean(cluster_pixels[:, 0])
            cy = np.mean(cluster_pixels[:, 1])
            clusters.append((cx, cy, area))
        # Настоящие метки образуют ровную сетку.
        # Поэтому мы отфильтруем весь "мусор", оставив 49 точек, которые
        # геометрически образуют наиболее правильную сетку (с одинаковым шагом).
        if len(clusters) > 49:
            points_arr = np.array([[c[0], c[1]] for c in clusters])
            diff = points_arr[:, np.newaxis, :] - points_arr[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
            
            dist_mat.sort(axis=1)
            # Берем среднее расстояние до 4-х ближайших соседей
            avg_nearest_dists = np.mean(dist_mat[:, 1:5], axis=1)
            
            median_step = np.median(avg_nearest_dists)
            median_area = np.median([c[2] for c in clusters])
            
            # Ошибка: насколько шаг и площадь отличаются от идеальных
            area_errors = np.array([abs(c[2] - median_area) / median_area if median_area > 0 else 0 for c in clusters])
            step_errors = np.abs(avg_nearest_dists - median_step) / median_step if median_step > 0 else 0
            
            # Геометрия сетки намного важнее площади, поэтому коэффициент 2.0
            total_errors = area_errors + step_errors * 2.0
            
            scored_clusters = list(zip(clusters, total_errors))
            scored_clusters.sort(key=lambda x: x[1])
            clusters = [x[0] for x in scored_clusters[:49]]
            
        centers = [(c[0], c[1]) for c in clusters]
        return centers

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
        detector = ClusteringMLDetector(**ml_params)
        print("Используется ML метод (DBSCAN Кластеризация)")
            
        points = detector.detect(img, preprocessed)
        
        process_time = time.time() - start_time
        
        print(f"Найдено меток: {len(points)} (Ожидается: 49 / 7x7)")
        if len(points) != 49:
            print("ВНИМАНИЕ: Количество найденных меток отличается от ожидаемого (49)!")
            
        print(f"Время обработки: {process_time:.3f} сек")
        
        error = self.evaluator.calculate_error(points)
        print(f"Оценка ошибки локализации (STD расстояний сетки): {error:.4f} пикс.")
        
        base_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        
        term_dir = os.path.join(base_dir, "term", self.timestamp)
        os.makedirs(term_dir, exist_ok=True)
        output_path = os.path.join(term_dir, base_name.replace(".jpg", "_result_ml.jpg"))
        
        self.visualizer.draw_results(img, points, output_path)
        
        cords_dir = os.path.join(base_dir, "cords", self.timestamp)
        os.makedirs(cords_dir, exist_ok=True)
        csv_path = os.path.join(cords_dir, base_name.replace(".jpg", "_coords_ml.csv"))
        
        DataExporter.export_to_csv(points, csv_path)

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
