import time
import cv2

from console import ConsoleInterface

from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from geometry import PerspectiveCorrector
from utils.visual import Visualizer
from detection.detect_circles import detect_circled_points
from origins import CoordinateSystemDetector


def main():

    ui = ConsoleInterface()
    ui.run()

    cfg = ui.config

    binarizer = Binarizer()

    detector = PcaDetector(
        roi_size=cfg.pca_roi_size,
        min_corner_ratio=cfg.pca_corner_ratio,
    )

    corrector = PerspectiveCorrector(
        grid_size=(cfg.grid_rows, cfg.grid_cols),
        actual_step_mm=cfg.marker_step_mm,
    )

    coord_detector = CoordinateSystemDetector(
        angle_thresh=cfg.angle_thresh,
    )

    visualizer = Visualizer(
        marker_color=(0, 255, 0),
        radius=1,
        thickness=2,
    )

    for i in range(cfg.start_image, cfg.end_image + 1):

        image_path = f"data/photo_{i}.jpg"
        img = cv2.imread(image_path)

        if img is None:
            print(f"[WARNING] Не удалось открыть {image_path}")
            continue

        if cfg.verbose:
            print(f"\nОбработка кадра: {image_path}")

        start_time = time.time()

        mask = binarizer.process(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        raw_points = detector.detect_points(mask, gray)

        pixel_points, _ = corrector.process_and_restore_grid(raw_points)

        circled_points = detect_circled_points(mask, pixel_points)

        coordinate_systems = coord_detector.detect(
            pixel_points,
            circled_points,
            grid_rows=cfg.grid_rows,
            grid_cols=cfg.grid_cols,
        )

        dt = time.time() - start_time

        if cfg.show_statistics:

            print("-" * 50)
            print(f"Сырых точек: {len(raw_points)}")
            print(f"Восстановленных: {len(pixel_points)}")
            print(f"Систем координат: {len(coordinate_systems)}")
            print(f"Время: {dt:.4f} сек.")
            print("-" * 50)

        if cfg.show_coordinate_systems:

            for idx, s in enumerate(coordinate_systems):

                print(f"\nSystem {idx}")
                print("origin:", s["origin"])
                print("x_axis:", s["x_axis"])
                print("y_axis:", s["y_axis"])

        if cfg.show_images:

            cv2.imshow(f"Mask {i}", mask)
            cv2.imshow(f"Result {i}",
                       visualizer.draw_detected_points(img, pixel_points))

            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()