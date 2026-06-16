import time
import cv2
from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from geometry import PerspectiveCorrector
from utils.visual import Visualizer
from detection.detect_circles import detect_circled_points
from origins import CoordinateSystemDetector
def main():
    grid_rows = 9
    grid_cols = 11
    marker_step_mm = 35.0
    binarizer = Binarizer()
    detector = PcaDetector(roi_size=15)
    corrector = PerspectiveCorrector(
        grid_size=(grid_rows, grid_cols), actual_step_mm=marker_step_mm
    )
    visualizer = Visualizer(marker_color=(0, 255, 0), radius=1, thickness=2)
    for i in range(10,12):
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
        circled_points = detect_circled_points(mask, pixel_points)
        print(f"DEBUG: Найдено округлых контуров: {len(circled_points)}")
        coord_detector = CoordinateSystemDetector()
        frames = coord_detector.detect(
            pixel_points,
            circled_points,
            grid_rows=grid_rows,
            grid_cols=grid_cols
        )
        print("Coordinate systems found:", len(frames))
        for idx, cs in enumerate(frames):
            print(f"\nSystem {idx}")
            print("origin:", cs["origin"])
            print("x_axis:", cs["x_axis"])
            print("y_axis:", cs["y_axis"])
        execution_time = time.time() - start_time
        print("-" * 50)
        print(f"Сырых точек найдено: {len(raw_pixel_points)}")
        print(f"После восстановления структуры: {len(pixel_points)}")
        print(f"Время: {execution_time:.4f} сек.")
        print("-" * 50)
        result_img = visualizer.draw_detected_points(img, pixel_points)
        for idx, cs in enumerate(frames):
            O_pixel = (int(cs["origin"][0]), int(cs["origin"][1]))
            axis_length = 80
            X_end = (
                int(cs["origin"][0] + cs["x_axis"][0] * axis_length),
                int(cs["origin"][1] + cs["x_axis"][1] * axis_length)
            )
            Y_end = (
                int(cs["origin"][0] + cs["y_axis"][0] * axis_length),
                int(cs["origin"][1] + cs["y_axis"][1] * axis_length)
            )
            cv2.arrowedLine(result_img, O_pixel, X_end, (0, 0, 255), 3, tipLength=0.2)
            cv2.arrowedLine(result_img, O_pixel, Y_end, (255, 0, 0), 3, tipLength=0.2)
            cv2.circle(result_img, O_pixel, 1, (255, 0, 255), -1)
            cv2.putText(
                result_img,
                f'{idx}',
                (O_pixel[0] - 20, O_pixel[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        h, w = result_img.shape[:2]
        display_img = (
            cv2.resize(result_img, (w // 2, h // 2)) if w > 1920 else result_img
        )
        cv2.imshow(f"Фото {i}", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()