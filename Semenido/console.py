import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


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


class ConsoleInterface:

    def __init__(self):
        self.config = AppConfig()
        self.running = True

    def run(self):
        print("\n" + "=" * 70)
        print("  CROSS MARKER DETECTOR - Консольный интерфейс")
        print("=" * 70)

        while self.running:
            self._print_main_menu()
            choice = input("\nВыберите пункт: ").strip()

            if choice == "1":
                self._run_pipeline()
            elif choice == "2":
                self._edit_config()
            elif choice == "3":
                self._load_config_from_json()
            elif choice == "4":
                self._save_config_to_json()
            elif choice == "5":
                self._show_config()
            elif choice == "6":
                self._batch_process()
            elif choice == "7":
                self._generate_example_config()
            elif choice == "0":
                self.running = False
                print("\nВыход из программы...")
            else:
                print("Неверный выбор. Попробуйте снова.")

    def _print_main_menu(self):
        print("\n" + "-" * 70)
        print("  ГЛАВНОЕ МЕНЮ")
        print("-" * 70)
        print("  1 - Запустить обработку")
        print("  2 - Редактировать конфигурацию")
        print("  3 - Загрузить конфигурацию из JSON")
        print("  4 - Сохранить конфигурацию в JSON")
        print("  5 - Показать текущую конфигурацию")
        print("  6 - Изменить режим работы")
        print("  7 - Сгенерировать пример конфигурации")
        print("  0 - Выход")
        print("-" * 70)

    def _print_edit_menu(self):
        print("\n" + "-" * 70)
        print("  РЕДАКТИРОВАНИЕ КОНФИГУРАЦИИ")
        print("-" * 70)
        print("  1 - Настройки сетки")
        print("  2 - Настройки детектора")
        print("  3 - Настройки координатных систем")
        print("  4 - Настройки отображения")
        print("  5 - Диапазон изображений")
        print("  6 - Режим работы")
        print("  7 - Настройки сохранения")
        print("  0 - Вернуться в главное меню")
        print("-" * 70)

    def _edit_config(self):
        while True:
            self._print_edit_menu()
            choice = input("\nВыберите пункт: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._edit_grid_settings()
            elif choice == "2":
                self._edit_detector_settings()
            elif choice == "3":
                self._edit_coordinate_settings()
            elif choice == "4":
                self._edit_display_settings()
            elif choice == "5":
                self._edit_image_range()
            elif choice == "6":
                self._edit_work_mode()
            elif choice == "7":
                self._edit_save_settings()
            else:
                print("Неверный выбор.")

    def _edit_grid_settings(self):
        cfg = self.config
        print("\nНастройки сетки")
        print("-" * 40)

        v = input(f"  Количество строк [{cfg.grid_rows}]: ").strip()
        if v:
            cfg.grid_rows = max(1, int(v))

        v = input(f"  Количество столбцов [{cfg.grid_cols}]: ").strip()
        if v:
            cfg.grid_cols = max(1, int(v))

        v = input(f"  Шаг маркеров (мм) [{cfg.marker_step_mm}]: ").strip()
        if v:
            cfg.marker_step_mm = max(0.1, float(v))

        print("Настройки сетки обновлены")

    def _edit_detector_settings(self):
        cfg = self.config
        print("\nНастройки детектора")
        print("-" * 40)

        v = input(f"  Размер ROI для PCA [{cfg.pca_roi_size}]: ").strip()
        if v:
            cfg.pca_roi_size = max(3, int(v))

        v = input(f"  Коэффициент углов [{cfg.pca_corner_ratio}]: ").strip()
        if v:
            cfg.pca_corner_ratio = max(0.01, min(1.0, float(v)))

        print("Настройки детектора обновлены")

    def _edit_coordinate_settings(self):
        cfg = self.config
        print("\nНастройки координатных систем")
        print("-" * 40)

        v = input(f"  Расстояние кластеризации [{cfg.cluster_dist}]: ").strip()
        if v:
            cfg.cluster_dist = max(1.0, float(v))

        v = input(f"  Порог угла (градусы) [{cfg.angle_thresh}]: ").strip()
        if v:
            cfg.angle_thresh = max(0.1, min(90.0, float(v)))

        print("Настройки координатных систем обновлены")

    def _edit_display_settings(self):
        cfg = self.config
        print("\nНастройки отображения")
        print("-" * 40)

        v = input(f"  Показывать изображения [{int(cfg.show_images)}]: ").strip()
        if v:
            cfg.show_images = bool(int(v))

        v = input(f"  Показывать статистику [{int(cfg.show_statistics)}]: ").strip()
        if v:
            cfg.show_statistics = bool(int(v))

        v = input(f"  Показывать координатные системы [{int(cfg.show_coordinate_systems)}]: ").strip()
        if v:
            cfg.show_coordinate_systems = bool(int(v))

        v = input(f"  Подробный вывод [{int(cfg.verbose)}]: ").strip()
        if v:
            cfg.verbose = bool(int(v))

        v = input(f"  Режим отладки [{int(cfg.debug)}]: ").strip()
        if v:
            cfg.debug = bool(int(v))

        print("Настройки отображения обновлены")

    def _edit_image_range(self):
        cfg = self.config
        print("\nДиапазон изображений")
        print("-" * 40)

        v = input(f"  Начальный номер [{cfg.start_image}]: ").strip()
        if v:
            cfg.start_image = max(0, int(v))

        v = input(f"  Конечный номер [{cfg.end_image}]: ").strip()
        if v:
            cfg.end_image = max(cfg.start_image, int(v))

        print(f"Диапазон обновлен: {cfg.start_image} - {cfg.end_image}")

    def _edit_work_mode(self):
        cfg = self.config
        print("\nРежим работы")
        print("-" * 40)
        print("  grid   - Обработка сетки маркеров")
        print("  single - Поиск одиночного крестика")
        print("-" * 40)

        v = input(f"  Режим [{cfg.cross_mode}]: ").strip()
        if v and v.lower() in ["grid", "single"]:
            cfg.cross_mode = v.lower()
            print(f"Режим обновлен: {cfg.cross_mode}")
        elif v:
            print("Неверный режим. Доступны: grid, single")

    def _edit_save_settings(self):
        cfg = self.config
        print("\nНастройки сохранения")
        print("-" * 40)

        v = input(f"  Сохранять изображения [{int(cfg.save_images)}]: ").strip()
        if v:
            cfg.save_images = bool(int(v))

        v = input(f"  Сохранять координаты [{int(cfg.save_coords)}]: ").strip()
        if v:
            cfg.save_coords = bool(int(v))

        v = input(f"  Папка для результатов [{cfg.output_dir}]: ").strip()
        if v:
            cfg.output_dir = v

        print("Настройки сохранения обновлены")

    def _save_config_to_json(self):
        print("\nСохранение конфигурации")
        print("-" * 40)

        default_name = "config.json"
        filename = input(f"  Имя файла [{default_name}]: ").strip()
        if not filename:
            filename = default_name

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=4, ensure_ascii=False)
            print(f"Конфигурация сохранена в {filename}")
        except Exception as e:
            print(f"Ошибка сохранения: {e}")

    def _load_config_from_json(self):
        print("\nЗагрузка конфигурации")
        print("-" * 40)

        default_name = "config.json"
        filename = input(f"  Имя файла [{default_name}]: ").strip()
        if not filename:
            filename = default_name

        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            cfg = self.config
            for key, value in config_data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

            print(f"Конфигурация загружена из {filename}")
            self._show_config()
        except Exception as e:
            print(f"Ошибка загрузки: {e}")

    def _show_config(self):
        print("\n" + "=" * 70)
        print("  ТЕКУЩАЯ КОНФИГУРАЦИЯ")
        print("=" * 70)

        cfg = self.config
        print("\nНАСТРОЙКИ СЕТКИ:")
        print(f"  Строки:            {cfg.grid_rows}")
        print(f"  Столбцы:           {cfg.grid_cols}")
        print(f"  Шаг (мм):          {cfg.marker_step_mm}")

        print("\nНАСТРОЙКИ ДЕТЕКТОРА:")
        print(f"  Размер ROI:        {cfg.pca_roi_size}")
        print(f"  Коэф. углов:       {cfg.pca_corner_ratio}")

        print("\nНАСТРОЙКИ КООРДИНАТНЫХ СИСТЕМ:")
        print(f"  Кластеризация:     {cfg.cluster_dist}")
        print(f"  Порог угла:        {cfg.angle_thresh} град")

        print("\nНАСТРОЙКИ ОТОБРАЖЕНИЯ:")
        print(f"  Показ. картинок:   {cfg.show_images}")
        print(f"  Статистика:        {cfg.show_statistics}")
        print(f"  Коорд. системы:    {cfg.show_coordinate_systems}")
        print(f"  Подробный вывод:   {cfg.verbose}")
        print(f"  Отладка:           {cfg.debug}")

        print("\nНАСТРОЙКИ ИЗОБРАЖЕНИЙ:")
        print(f"  Диапазон:          {cfg.start_image} - {cfg.end_image}")
        print(f"  Режим:             {cfg.cross_mode}")

        print("\nНАСТРОЙКИ СОХРАНЕНИЯ:")
        print(f"  Сохранять фото:    {cfg.save_images}")
        print(f"  Сохранять коорд:   {cfg.save_coords}")
        print(f"  Папка результатов: {cfg.output_dir}")

        print("\n" + "=" * 70)

    def _generate_example_config(self):
        print("\nГенерация примера конфигурации")
        print("-" * 40)

        filename = input("  Имя файла [example_config.json]: ").strip()
        if not filename:
            filename = "example_config.json"

        example_config = {
            "grid_rows": 7,
            "grid_cols": 7,
            "marker_step_mm": 35.0,
            "pca_roi_size": 15,
            "pca_corner_ratio": 0.12,
            "cluster_dist": 80.0,
            "angle_thresh": 25.0,
            "show_images": True,
            "show_statistics": True,
            "show_coordinate_systems": True,
            "verbose": False,
            "debug": False,
            "start_image": 1,
            "end_image": 10,
            "cross_mode": "grid",
            "save_images": False,
            "save_coords": False,
            "output_dir": "output"
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(example_config, f, indent=4, ensure_ascii=False)
            print(f"Пример конфигурации сохранен в {filename}")
            print("\nСодержимое примера:")
            print(json.dumps(example_config, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Ошибка сохранения: {e}")

    def _run_pipeline(self):
        print("\n" + "=" * 70)
        print("  ЗАПУСК ОБРАБОТКИ")
        print("=" * 70)

        self._show_config()

        confirm = input("\nЗапустить обработку с этими настройками? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Запуск отменен")
            return

        print("\nЗапуск обработки...")
        start_time = time.time()

        try:
            from main import run_pipeline

            run_pipeline(
                self.config,
                save_images=self.config.save_images,
                output_dir=self.config.output_dir,
                save_coords=self.config.save_coords
            )

            elapsed = time.time() - start_time
            print(f"\nОбработка завершена за {elapsed:.2f} секунд")

        except ImportError as e:
            print(f"Ошибка импорта: {e}")
            print("   Убедитесь, что файл main.py существует и содержит функцию run_pipeline")
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()

    def _batch_process(self):
        print("\nПАКЕТНАЯ ОБРАБОТКА")
        print("=" * 70)

        folder = input("Путь к папке с изображениями [data/]: ").strip()
        if not folder:
            folder = "data/"

        if not os.path.exists(folder):
            print(f"Папка {folder} не найдена")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(valid_extensions)
        ])

        if not images:
            print(f"В папке {folder} не найдено изображений")
            return

        print(f"\nНайдено изображений: {len(images)}")
        for i, img in enumerate(images, 1):
            print(f"  {i}. {img}")

        confirm = input("\nОбработать все изображения? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Отменено")
            return

        self.config.start_image = 1
        self.config.end_image = len(images)

        print(f"\nЗапуск пакетной обработки ({len(images)} изображений)...")
        start_time = time.time()

        try:
            from main import run_pipeline

            run_pipeline(
                self.config,
                save_images=self.config.save_images,
                output_dir=self.config.output_dir,
                save_coords=self.config.save_coords
            )

            elapsed = time.time() - start_time
            print(f"\nПакетная обработка завершена за {elapsed:.2f} секунд")

        except Exception as e:
            print(f"Ошибка пакетной обработки: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Cross Marker Detector - Консольный интерфейс",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python console.py
  python console.py --config config.json
  python console.py --mode grid --start 1 --end 10 --save-images
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Путь к JSON файлу конфигурации'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['grid', 'single'],
        help='Режим работы'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='Начальный номер изображения'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='Конечный номер изображения'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Сохранять обработанные изображения'
    )
    parser.add_argument(
        '--save-coords',
        action='store_true',
        help='Сохранять координаты точек'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Папка для сохранения результатов'
    )

    args = parser.parse_args()

    ui = ConsoleInterface()

    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(ui.config, key):
                    setattr(ui.config, key, value)
            print(f"Конфигурация загружена из {args.config}")
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")

    if args.mode:
        ui.config.cross_mode = args.mode

    if args.start:
        ui.config.start_image = args.start

    if args.end:
        ui.config.end_image = args.end

    if args.save_images:
        ui.config.save_images = True

    if args.save_coords:
        ui.config.save_coords = True

    if args.output:
        ui.config.output_dir = args.output

    ui.run()


if __name__ == "__main__":
    main()