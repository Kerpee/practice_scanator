"""
Скрипт generate_dataset.py
--------------------------
Генерирует синтетический датасет с нарисованными крестиками на градиентном фоне.
Формирует структуру папок (images/train, images/val, labels/train, labels/val)
и создает файл конфигурации dataset.yaml для обучения модели YOLO.
"""

import cv2
import numpy as np
import os
import random
import yaml
from pathlib import Path

def create_gradient_background(width, height):
    c1 = np.array([random.randint(0, 255) for _ in range(3)])
    c2 = np.array([random.randint(0, 255) for _ in range(3)])
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        alpha = y / height
        color = c1 * (1 - alpha) + c2 * alpha
        img[y, :] = color
        
    # Добавляем шум
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Размытие
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), random.uniform(0.5, 2.0))
        
    return img

def generate_dataset(base_dir="dataset", num_train=300, num_val=50, img_size=640):
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    def generate_split(split, num_images):
        print(f"Генерация {split} ({num_images} фото)...")
        for i in range(num_images):
            img = create_gradient_background(img_size, img_size)
            
            num_crosses = random.randint(10, 30)
            labels = []
            
            for _ in range(num_crosses):
                cx = random.randint(50, img_size - 50)
                cy = random.randint(50, img_size - 50)
                size = random.randint(15, 35)
                thickness = random.randint(2, 5)
                
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                # Рисуем крест
                cv2.line(img, (cx - size//2, cy), (cx + size//2, cy), color, thickness)
                cv2.line(img, (cx, cy - size//2), (cx, cy + size//2), color, thickness)
                
                # Координаты для YOLO (нормированные)
                bw = size / img_size
                bh = size / img_size
                nx = cx / img_size
                ny = cy / img_size
                
                labels.append(f"0 {nx:.6f} {ny:.6f} {bw:.6f} {bh:.6f}")
                
            img_path = f"{base_dir}/images/{split}/synth_{i:04d}.jpg"
            label_path = f"{base_dir}/labels/{split}/synth_{i:04d}.txt"
            
            cv2.imwrite(img_path, img)
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
                
    generate_split("train", num_train)
    generate_split("val", num_val)
    
    # Создаем dataset.yaml
    yaml_content = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['cross']
    }
    
    with open(f"{base_dir}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Датасет успешно сгенерирован в папке {base_dir}!")
    print(f"Конфиг сохранен в {os.path.abspath(base_dir)}/dataset.yaml")

if __name__ == "__main__":
    generate_dataset()
