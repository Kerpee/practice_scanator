"""
Скрипт optimizer.py
-------------------
Выполняет автоматический подбор (Grid Search) наилучших параметров (thresh_block_size, thresh_c, eps, min_samples) 
для алгоритма кластеризации (если используется старый пайплайн без YOLO).
Сохраняет найденные лучшие параметры в файл best_params.json.
"""

import os
import glob
import itertools
import json
import time
import numpy as np

# Импортируем классы из основного скрипта
from mlka import ImagePreprocessor, ClusteringMLDetector, LocalizationEvaluator

def evaluate_params(images, prep_params, ml_params):
    """
    Оценивает набор параметров на всех переданных изображениях.
    Возвращает:
        success_count: количество изображений, на которых найдено ровно 49 меток
        avg_error: средняя ошибка (STD сетки) по всем успешным изображениям
    """
    success_count = 0
    total_error = 0.0
    
    preprocessor = ImagePreprocessor(**prep_params)
    detector = ClusteringMLDetector(**ml_params)
    evaluator = LocalizationEvaluator()
    
    for img_path in images:
        try:
            img, preprocessed = preprocessor.process(img_path)
            points = detector.detect(img, preprocessed)
            
            if len(points) == 49:
                success_count += 1
                error = evaluator.calculate_error(points)
                total_error += error
        except Exception as e:
            # Игнорируем ошибки при неудачных параметрах
            pass
            
    avg_error = total_error / success_count if success_count > 0 else float('inf')
    return success_count, avg_error

def run_optimization(data_dir: str, output_json: str = "best_params.json"):
    images = glob.glob(os.path.join(data_dir, "*.jpg"))
    # Фильтруем результаты, чтобы не брать файлы `_result_`
    images = [img for img in images if "_result_" not in img]
    
    if not images:
        print(f"В папке {data_dir} не найдено изображений для обучения.")
        return

    print(f"Найдено {len(images)} изображений для авто-тюнинга.")
    print("Запуск оптимизации (Grid Search). Пожалуйста, подождите...")
    
    # Увеличенная сетка параметров для перебора
    param_grid = {
        'thresh_block_size': [11, 15, 21, 27, 35, 45],  # Должны быть нечетными
        'thresh_c': [0, -5, -10, -15, -20, -25],
        'eps': [10, 15, 20, 25],
        'min_samples': [2, 3, 5, 7]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_params = None
    best_success_count = -1
    best_error = float('inf')
    
    start_time = time.time()
    
    for i, params in enumerate(combinations):
        prep_params = {
            'thresh_block_size': params['thresh_block_size'],
            'thresh_c': params['thresh_c']
        }
        ml_params = {
            'eps': params['eps'],
            'min_samples': params['min_samples']
        }
        
        success_count, avg_error = evaluate_params(images, prep_params, ml_params)
        
        # Логика выбора лучших параметров:
        # 1. Приоритет - максимальное количество фото с 49 найденными метками
        # 2. При равном успехе - выбираем с минимальной ошибкой (самая ровная сетка)
        if success_count > best_success_count or (success_count == best_success_count and avg_error < best_error):
            best_success_count = success_count
            best_error = avg_error
            best_params = params
            print(f"[{i}/{len(combinations)}] Новые лучшие параметры: {params} | Успешных фото: {success_count}/{len(images)} | Ошибка: {avg_error:.4f}")

    print("-" * 50)
    print(f"Оптимизация завершена за {time.time() - start_time:.1f} сек.")
    print(f"Лучший результат: {best_success_count}/{len(images)} фото распознано идеально. Ошибка сетки: {best_error:.4f}")
    print(f"Сохранение в {output_json}...")
    
    if best_params:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4)
        print("Готово!")
    else:
        print("Не удалось найти параметры, которые распознают хотя бы одно фото корректно.")

if __name__ == "__main__":
    # Папка с датасетом (по умолчанию на уровень выше, если мы в ML/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(base_dir, "data")
    output_json = os.path.join(os.path.dirname(__file__), "best_params.json")
    run_optimization(data_folder, output_json=output_json)
