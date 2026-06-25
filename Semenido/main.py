import time
import cv2
import argparse
import json
import os
import numpy as np
from console import ConsoleInterface
from gui import AppGui
from preprocess.preprocess import Binarizer
from detection.PCA import PcaDetector
from geometry import PerspectiveCorrector
from utils.visual import Visualizer
from detection.detect_circles import detect_circled_points
from origins import CoordinateSystemDetector
from dataclasses import dataclass


@dataclass
class AppConfig:
    grid_rows: int = 7
    grid_cols: int = 7
    marker_step_mm: float = 35.0
    pca_roi_size: int = 15
    pca_corner_ratio: float = 0.12
    cluster_dist: float = 80.0
    angle_thresh: float = 25.0
    show_images: bool = True
    show_statistics: bool = True
    show_coordinate_systems: bool = True
    verbose: bool = False
    start_image: int = 1
    end_image: int = 10
    cross_mode: str = "grid"
    save_images: bool = False
    save_coords: bool = False
    output_dir: str = "output"
    debug: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--gui", action="store_true", help="Запустить оконный GUI интерфейс")
    p.add_argument("--config", type=str, help="Путь к JSON файлу с настройками")
    p.add_argument("--output-dir", type=str, default="output", help="Папка для сохранения результатов")
    p.add_argument("--save-images", action="store_true", help="Сохранять обработанные изображения")
    p.add_argument("--save-coords", action="store_true", help="Сохранять координаты точек в CSV")
    return p.parse_args()


def load_config_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"Файл {json_path} не найден, используются стандартные настройки")
        return AppConfig()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        cfg = AppConfig()
        for key, value in config_data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        print(f"Конфигурация загружена из {json_path}")
        return cfg
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return AppConfig()


def save_config_to_json(cfg, json_path="config.json"):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vars(cfg), f, indent=4, ensure_ascii=False)
        print(f"Конфигурация сохранена в {json_path}")
    except Exception as e:
        print(f"Ошибка сохранения конфигурации: {e}")


def save_coordinates_to_csv(coordinate_systems, pixel_points, filename, image_idx):
    import csv

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Point_Type', 'ID', 'X', 'Y'])

        # Проверяем тип pixel_points и преобразуем в список если нужно
        if isinstance(pixel_points, np.ndarray):
            points_list = pixel_points.tolist()
        else:
            points_list = list(pixel_points)

        for idx, point in enumerate(points_list):
            writer.writerow([image_idx, 'grid_point', idx, point[0], point[1]])

        for idx, system in enumerate(coordinate_systems):
            writer.writerow([image_idx, 'origin', idx, system['origin'][0], system['origin'][1]])
            writer.writerow([image_idx, 'x_axis', idx, system['x_axis'][0], system['x_axis'][1]])
            writer.writerow([image_idx, 'y_axis', idx, system['y_axis'][0], system['y_axis'][1]])


def run_pipeline(cfg, ui_image_callback=None, save_images=False, output_dir="output", save_coords=False):
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

    if save_images or save_coords:
        os.makedirs(output_dir, exist_ok=True)
        if save_images:
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
        if save_coords:
            coords_dir = os.path.join(output_dir, "coordinates")
            os.makedirs(coords_dir, exist_ok=True)

    all_results = []

    for i in range(cfg.start_image, cfg.end_image + 1):
        image_path = f"data/photo_{i}.jpg"
        img = cv2.imread(image_path)

        if img is None:
            print(f"Не удалось открыть {image_path}")
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
            if center is not None:
                vis = img.copy()
                cv2.circle(vis, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

                if save_images:
                    save_path = os.path.join(images_dir, f"single_{i}.jpg")
                    cv2.imwrite(save_path, vis)
                    print(f"Изображение сохранено: {save_path}")

                if ui_image_callback and cfg.show_images:
                    ui_image_callback(vis, f"Single Cross {i}")
                elif cfg.show_images:
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
        if pixel_points is None or len(pixel_points) == 0:
            print(f"Не удалось восстановить точки для кадра {i}")
            continue

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

        if save_coords:
            coords_filename = os.path.join(coords_dir, f"coordinates_{i}.csv")
            save_coordinates_to_csv(coordinate_systems, pixel_points, coords_filename, i)
            print(f"Координаты сохранены: {coords_filename}")

            # Преобразуем точки в список для JSON сериализации
            if isinstance(pixel_points, np.ndarray):
                points_list = pixel_points.tolist()
            else:
                points_list = list(pixel_points)

            all_results.append({
                'image': i,
                'pixel_points': points_list,
                'coordinate_systems': coordinate_systems
            })

        if cfg.show_images:
            res_img = visualizer.draw_detected_points(img, pixel_points)

            if save_images:
                save_path = os.path.join(images_dir, f"grid_{i}.jpg")
                cv2.imwrite(save_path, res_img)
                print(f"Изображение сохранено: {save_path}")

            if ui_image_callback:
                ui_image_callback(res_img, f"Result {i}")
            else:
                cv2.imshow(f"Mask {i}", mask)
                cv2.imshow(f"Result {i}", res_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if save_coords and all_results:
        all_results_path = os.path.join(coords_dir, "all_results.json")
        try:
            with open(all_results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Все результаты сохранены в: {all_results_path}")
        except Exception as e:
            print(f"Ошибка сохранения всех результатов: {e}")


def main():
    args = parse_args()

    cfg = AppConfig()

    if args.config:
        cfg = load_config_from_json(args.config)

    if args.save_images:
        cfg.save_images = True
    else:
        cfg.save_images = False

    if args.save_coords:
        cfg.save_coords = True
    else:
        cfg.save_coords = False

    if args.config and not os.path.exists(args.config):
        save_config_to_json(cfg, args.config)

    if args.gui:
        print("Запуск графического интерфейса...")
        gui = AppGui(run_pipeline_callback=run_pipeline)
        gui.mainloop()
        return

    ui = ConsoleInterface()
    if not args.no_ui and not args.fast:
        ui.run()
    cfg = ui.config
    if args.fast:
        cfg.show_images = False
        cfg.show_statistics = True
        cfg.show_coordinate_systems = False
        cfg.verbose = False

    run_pipeline(
        cfg,
        save_images=args.save_images,
        output_dir=args.output_dir,
        save_coords=args.save_coords
    )


if __name__ == "__main__":
    main()