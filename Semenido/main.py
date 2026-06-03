import time
import cv2
from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from utils.visual import Visualizer
def main():
    image_path = "data/photo_1.jpg"
    binarizer = Binarizer(
        tophat_ksize=9,
        thresh_value=15,
        min_area=30.0,
        max_area=1200.0
    )
    detector = PcaDetector(roi_padding=5)
    visualizer = Visualizer(marker_color=(0, 255, 0), radius=6, thickness=2)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить файл {image_path}!")
        return
    start_time = time.time()
    mask = binarizer.process(img)
    pixel_points = detector.detect_points(mask)
    execution_time = time.time() - start_time
    print("-" * 50)
    print(f"Время обработки: {execution_time:.4f} сек.")
    print(f"Количество найденных меток: {len(pixel_points)}")
    print("-" * 50)
    result_img = visualizer.draw_detected_points(img, pixel_points)
    h, w = result_img.shape[:2]
    display_img = cv2.resize(result_img, (w // 2, h // 2)) if w > 1920 else result_img
    cv2.imshow("Calibration Result - PCA Centers", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()