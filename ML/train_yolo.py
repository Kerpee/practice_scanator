from ultralytics import YOLO
import os

def main():
    dataset_yaml = os.path.abspath('dataset_realistic/dataset.yaml')
    if not os.path.exists(dataset_yaml):
        print("Датасет не найден. Сначала запустите generate_realistic_dataset.py")
        return

    print("Загрузка базовой модели yolov8n.pt...")
    model = YOLO('yolov8n.pt')

    print("Начало обучения на реалистичном датасете...")
    results = model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        batch=16,
        name='cross_detector_realistic',
        device='cpu' 
    )

    print("\nОбучение завершено!")
    print(f"Лучшая модель сохранена в: runs/detect/cross_detector/weights/best.pt")

if __name__ == "__main__":
    main()
