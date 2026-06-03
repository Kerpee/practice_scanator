import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple
from sklearn.cluster import DBSCAN
import os
import csv

class ImagePreprocessor:
    """Модуль препроцессинга изображений калибровочных карт."""
    def __init__(self, clahe_clip: float = 2.0, clahe_grid: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

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
            cv2.THRESH_BINARY, 21, -10  # C < 0 чтобы выделить светлые участки
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
        
        centers = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                # Игнорируем шум
                continue
                
            # Выбираем пиксели текущего кластера
            cluster_pixels = pixels[labels == label]
            
            # Вычисляем центроид (субпиксельная точность)
            cx = np.mean(cluster_pixels[:, 0])
            cy = np.mean(cluster_pixels[:, 1])
            centers.append((cx, cy))
            
        return centers

class TemplateMatchingDetector(BaseDetector):
    """
    CV метод на основе сопоставления с шаблоном (Template Matching).
    """
    def __init__(self, template_size: int = 21, threshold: float = 0.6):
        self.template_size = template_size
        self.threshold = threshold
        self.template = self._create_cross_template()

    def _create_cross_template(self) -> np.ndarray:
        """Создает синтетический шаблон креста."""
        template = np.zeros((self.template_size, self.template_size), dtype=np.uint8)
        center = self.template_size // 2
        thickness = 3
        # Горизонтальная линия
        template[center-thickness//2:center+thickness//2+1, 2:-2] = 255
        # Вертикальная линия
        template[2:-2, center-thickness//2:center+thickness//2+1] = 255
        return template

    def detect(self, img: np.ndarray, preprocessed: np.ndarray) -> List[Tuple[float, float]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        res = cv2.matchTemplate(enhanced, self.template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.threshold)
        
        # Немаксимальное подавление (упрощенное)
        points = []
        for pt in zip(*loc[::-1]):  # Смена порядка на (x, y)
            cx = pt[0] + self.template_size / 2
            cy = pt[1] + self.template_size / 2
            points.append((cx, cy))
            
        filtered_points = []
        for p in points:
            if not any(np.hypot(p[0]-fp[0], p[1]-fp[1]) < 15 for fp in filtered_points):
                filtered_points.append(p)
                
        return filtered_points

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
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.visualizer = ResultVisualizer()
        self.evaluator = LocalizationEvaluator()
        
    def run(self, image_path: str, method: str = 'ml'):
        print(f"Начало обработки: {image_path}")
        start_time = time.time()
        
        img, preprocessed = self.preprocessor.process(image_path)
        
        if method == 'ml':
            detector = ClusteringMLDetector()
            print("Используется ML метод (DBSCAN Кластеризация)")
        elif method == 'cv':
            detector = TemplateMatchingDetector()
            print("Используется CV метод (Template Matching)")
        else:
            raise ValueError("Неизвестный метод")
            
        points = detector.detect(img, preprocessed)
        
        process_time = time.time() - start_time
        
        print(f"Найдено меток: {len(points)} (Ожидается: 49 / 7x7)")
        if len(points) != 49:
            print("ВНИМАНИЕ: Количество найденных меток отличается от ожидаемого (49)!")
            
        print(f"Время обработки: {process_time:.3f} сек")
        
        error = self.evaluator.calculate_error(points)
        print(f"Оценка ошибки локализации (STD расстояний сетки): {error:.4f} пикс.")
        
        # Создаем папки для вывода
        base_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        
        term_dir = os.path.join(base_dir, "term")
        os.makedirs(term_dir, exist_ok=True)
        output_path = os.path.join(term_dir, base_name.replace(".jpg", f"_result_{method}.jpg"))
        
        self.visualizer.draw_results(img, points, output_path)
        
        cords_dir = os.path.join(base_dir, "cords")
        os.makedirs(cords_dir, exist_ok=True)
        csv_path = os.path.join(cords_dir, base_name.replace(".jpg", f"_coords_{method}.csv"))
        
        DataExporter.export_to_csv(points, csv_path)

if __name__ == "__main__":
    app = ScannerCalibrationApp()
    image_file = r"c:\Users\kheso\Documents\практика\3 курс\practice_scanator\data\photo_2026-06-03_14-59-52.jpg"
    
    print("=" * 40)
    app.run(image_file, method='ml')
    print("=" * 40)
    app.run(image_file, method='cv')
    print("=" * 40)
