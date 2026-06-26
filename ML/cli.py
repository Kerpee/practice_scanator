#!/usr/bin/env python3
"""
Скрипт cli.py
-------------
Интерфейс командной строки для запуска гибридного детектора кросс-меток (YOLO + OpenCV).
Позволяет передать путь к изображению или папке (--input) и указать папку для результатов (--output).
Сохраняет визуализацию и CSV-файлы с координатами меток.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlka import ScannerCalibrationApp

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid YOLO+CV Cross Marker Detector.")
    parser.add_argument('--input', '-i', type=str, required=True, help='Путь к картинке или папке с картинками.')
    parser.add_argument('--output', '-o', type=str, default='results', help='Папка для результатов.')
    return parser.parse_args()

def get_image_files(input_path):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        return sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)])
    return []

def main():
    args = parse_args()
    image_files = get_image_files(args.input)
    
    if not image_files:
        print("[ОШИБКА] Изображения не найдены.")
        sys.exit(1)
        
    import datetime
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(args.output, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    
    try:
        app = ScannerCalibrationApp()
    except Exception as e:
        print(f"[ОШИБКА] Не удалось инициализировать приложение: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Файлов:      {len(image_files)}")
    print(f"  Алгоритм:    HYBRID (YOLO + OpenCV Moments)")
    print(f"  Папка:       {run_output_dir}")
    print(f"{'='*60}\n")

    for idx, path in enumerate(image_files, 1):
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_img = os.path.join(run_output_dir, f"{base_name}_result.jpg")
        out_csv = os.path.join(run_output_dir, f"{base_name}_result.csv")
        out_json = os.path.join(run_output_dir, f"{base_name}_result.json")
        
        result = app.run(path, output_img=out_img, output_csv=out_csv, output_json=out_json)
        
        print(f"[{idx}/{len(image_files)}] {base_name} | Меток: {result['points_count']} | Ошибка (STD): {result['error']:.3f} | {result['time']:.2f} сек")

if __name__ == "__main__":
    main()
