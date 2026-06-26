"""
Скрипт generate_realistic_dataset.py
------------------------------------
Генерирует более реалистичный датасет: берет реальные фоновые фотографии
и накладывает на них случайным образом шаблоны крестов (с вращением, масштабированием 
и изменением яркости). Формирует конфигурацию для YOLO (dataset_realistic/dataset.yaml).
"""

import cv2
import numpy as np
import os
import random
import yaml
import glob

def get_random_crop(image, crop_size=640):
    h, w = image.shape[:2]
    if h <= crop_size or w <= crop_size:
        return cv2.resize(image, (crop_size, crop_size))
    
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    return image[y:y+crop_size, x:x+crop_size].copy()

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def blend_patch(bg, patch, x, y):
    h, w = patch.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    
    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return bg # Out of bounds
        
    roi = bg[y:y+h, x:x+w]
    
    # Создаем маску для смешивания (края прозрачные, центр плотный)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w//2, h//2), min(w, h)//2 - 2, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 2.0)
    mask = np.stack([mask]*3, axis=-1)
    
    # Смешиваем
    blended = roi * (1 - mask) + patch * mask
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    return bg

def generate_dataset():
    base_dir = "dataset_realistic"
    templates_dir = "templates"
    backgrounds_dir = "../data"
    
    if not os.path.exists(templates_dir):
        print(f"Ошибка: Папка с шаблонами крестов {templates_dir} не найдена!")
        return
        
    template_files = glob.glob(f"{templates_dir}/*.jpg") + glob.glob(f"{templates_dir}/*.png")
    if not template_files:
        print(f"Ошибка: В папке {templates_dir} нет картинок!")
        return
        
    bg_files = [f for f in glob.glob(f"{backgrounds_dir}/*.jpg") if "photo" in f]
    
    def read_image_unicode(path):
        with open(path, 'rb') as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return img
        
    templates = [read_image_unicode(f) for f in template_files]
    # Отфильтруем None если какой-то файл битый
    templates = [t for t in templates if t is not None]
    if not templates:
        print("Ошибка: Не удалось прочитать ни один шаблон!")
        return
        
    backgrounds = [read_image_unicode(f) for f in bg_files if read_image_unicode(f) is not None]
    
    dirs = [
        f"{base_dir}/images/train", f"{base_dir}/images/val",
        f"{base_dir}/labels/train", f"{base_dir}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    num_train = 300
    num_val = 50
    img_size = 640
    
    def generate_split(split, num_images):
        print(f"Генерация {split} ({num_images} фото)...")
        for i in range(num_images):
            # Берем случайный фон из реальных фото
            bg = get_random_crop(random.choice(backgrounds), img_size)
            
            num_crosses = random.randint(5, 20)
            labels = []
            
            for _ in range(num_crosses):
                template = random.choice(templates)
                
                # Случайные трансформации шаблона
                angle = random.uniform(-45, 45)
                scale = random.uniform(0.8, 1.2)
                
                h, w = template.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(template, (new_w, new_h))
                rotated = rotate_image(resized, angle)
                
                # Добавляем случайное изменение яркости
                brightness_factor = random.uniform(0.7, 1.3)
                rotated = cv2.convertScaleAbs(rotated, alpha=brightness_factor, beta=0)
                
                # Случайные координаты
                cx = random.randint(50, img_size - 50)
                cy = random.randint(50, img_size - 50)
                
                px = cx - new_w // 2
                py = cy - new_h // 2
                
                bg = blend_patch(bg, rotated, px, py)
                
                # Координаты для YOLO
                bw = new_w / img_size
                bh = new_h / img_size
                nx = cx / img_size
                ny = cy / img_size
                
                labels.append(f"0 {nx:.6f} {ny:.6f} {bw:.6f} {bh:.6f}")
                
            img_path = f"{base_dir}/images/{split}/synth_{i:04d}.jpg"
            label_path = f"{base_dir}/labels/{split}/synth_{i:04d}.txt"
            
            cv2.imwrite(img_path, bg)
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
                
    generate_split("train", num_train)
    generate_split("val", num_val)
    
    yaml_content = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['cross']
    }
    
    with open(f"{base_dir}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("Реалистичный датасет успешно сгенерирован!")

if __name__ == "__main__":
    generate_dataset()
