import time
import cv2
import argparse
from console import ConsoleInterface
from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from geometry import PerspectiveCorrector
from utils.visual import Visualizer
from detection.detect_circles import detect_circled_points
from origins import CoordinateSystemDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-ui", action="store_true")
    return p.parse_args()


def main():

    args = parse_args()

    ui = ConsoleInterface()

    if not args.no_ui and not args.fast:
        ui.run()

    cfg = ui.config

    if args.fast:
        cfg.show_images = False
        cfg.show_statistics = True
        cfg.show_coordinate_systems = False
        cfg.verbose = False

    binarizer = Binarizer()

    detector = PcaDetector(
        roi_size=cfg.pca_roi_size,
        min_corner_ratio=cfg.pca_corner_ratio,
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

        image_path = f"data/{i}.jpg"
        img = cv2.imread(image_path)

        if img is None:
            print(f"[WARNING] Не удалось открыть {image_path}")
            continue

        if cfg.verbose:
            print(f"\nОбработка кадра: {image_path}")

        start_time = time.time()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cfg.cross_mode == "single":

            center = detector.detect_single_cross_center(gray)

            dt = time.time() - start_time

            if cfg.show_statistics:
                print("-" * 50)
                print("MODE: SINGLE")
                print("center:", center)
                print(f"time: {dt:.4f} sec")
                print("-" * 50)

            if cfg.show_images and center is not None:
                vis = img.copy()
                cv2.circle(
                    vis,
                    (int(center[0]), int(center[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )
                cv2.imshow("Single Cross", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            continue
        mask = binarizer.process(img)

        raw_points = detector.detect_points(mask, gray)

        pixel_points, _ = PerspectiveCorrector(
            grid_size=(cfg.grid_rows, cfg.grid_cols),
            actual_step_mm=cfg.marker_step_mm,
        ).process_and_restore_grid(raw_points)

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
            print(f"MODE: GRID")
            print(f"raw points: {len(raw_points)}")
            print(f"restored: {len(pixel_points)}")
            print(f"systems: {len(coordinate_systems)}")
            print(f"time: {dt:.4f} sec")
            print("-" * 50)

        if cfg.show_coordinate_systems:
            for idx, s in enumerate(coordinate_systems):
                print(f"\nSystem {idx}")
                print("origin:", s["origin"])
                print("x_axis:", s["x_axis"])
                print("y_axis:", s["y_axis"])

        if cfg.show_images:
            cv2.imshow(f"Mask {i}", mask)
            cv2.imshow(
                f"Result {i}",
                visualizer.draw_detected_points(img, pixel_points),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()