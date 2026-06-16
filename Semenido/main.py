import time
import cv2
from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from geometry import PerspectiveCorrector
from utils.visual import Visualizer


def main():
    grid_rows = 7
    grid_cols = 7
    marker_step_mm = 35.0
    binarizer = Binarizer()
    detector = PcaDetector(roi_size=15)
    corrector = PerspectiveCorrector(
        grid_size=(grid_rows, grid_cols), actual_step_mm=marker_step_mm
    )
    visualizer = Visualizer(marker_color=(0, 255, 0), radius=1, thickness=2)
    for i in range(1, 10):
        image_path = f"data/photo_{i}.jpg"
        img = cv2.imread(image_path)
        if img is None:
            continue
        print(f"\nОбработка кадра: {image_path}")
        start_time = time.time()
        mask = binarizer.process(img)
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_pixel_points = detector.detect_points(mask, gray_orig)
        cv2.imshow(f"Фото {i}", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pixel_points, mm_points = corrector.process_and_restore_grid(
            raw_pixel_points
        )
        execution_time = time.time() - start_time
        print("-" * 50)
        print(f"Сырых точек найдено: {len(raw_pixel_points)}")
        print(f"После восстановления структуры: {len(pixel_points)}")
        print(f"Время: {execution_time:.4f} сек.")
        print("-" * 50)
        result_img = visualizer.draw_detected_points(img, pixel_points)
        h, w = result_img.shape[:2]
        display_img = (
            cv2.resize(result_img, (w // 2, h // 2)) if w > 1920 else result_img
        )
        cv2.imshow(f"Фото {i}", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()