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


class ConsoleInterface:

    def __init__(self):
        self.config = AppConfig()


    def run(self):
        while True:
            self._print_menu()
            choice = input("\nВыберите пункт: ").strip()
            if choice == "1":
                self._edit_grid()
            elif choice == "2":
                self._edit_detector()
            elif choice == "3":
                self._edit_coordinate_system()
            elif choice == "4":
                self._edit_visualization()
            elif choice == "5":
                self._edit_image_range()
            elif choice == "6":
                self._show_config()
            elif choice == "0":
                break


    def _print_menu(self):
        print("\n" + "=" * 60)
        print("НАСТРОЙКИ ПРОЕКТА")
        print("=" * 60)
        print("1 - Настройки размеров сетки")
        print("2 - Настройки детектора")
        print("3 - Настройки координатных систем")
        print("4 - Настройки отображения")
        print("5 - Диапазон изображений")
        print("6 - Показать текущую конфигурацию")
        print("0 - Продолжить запуск")


    def _edit_grid(self):
        cfg = self.config
        v = input(f"Grid rows [{cfg.grid_rows}]: ").strip()
        if v:
            cfg.grid_rows = int(v)
        v = input(f"Grid cols [{cfg.grid_cols}]: ").strip()
        if v:
            cfg.grid_cols = int(v)
        v = input(f"Marker step mm [{cfg.marker_step_mm}]: ").strip()
        if v:
            cfg.marker_step_mm = float(v)


    def _edit_detector(self):
        cfg = self.config
        v = input(f"PCA roi size [{cfg.pca_roi_size}]: ").strip()
        if v:
            cfg.pca_roi_size = int(v)
        v = input(f"PCA corner ratio [{cfg.pca_corner_ratio}]: ").strip()
        if v:
            cfg.pca_corner_ratio = float(v)


    def _edit_coordinate_system(self):
        cfg = self.config
        v = input(f"Cluster distance [{cfg.cluster_dist}]: ").strip()
        if v:
            cfg.cluster_dist = float(v)
        v = input(f"Angle threshold [{cfg.angle_thresh}]: ").strip()
        if v:
            cfg.angle_thresh = float(v)


    def _edit_visualization(self):
        cfg = self.config
        v = input(f"Show images [{int(cfg.show_images)}]: ").strip()
        if v:
            cfg.show_images = bool(int(v))
        v = input(f"Show statistics [{int(cfg.show_statistics)}]: ").strip()
        if v:
            cfg.show_statistics = bool(int(v))
        v = input(f"Show coordinate systems [{int(cfg.show_coordinate_systems)}]: ").strip()
        if v:
            cfg.show_coordinate_systems = bool(int(v))
        v = input(f"Verbose [{int(cfg.verbose)}]: ").strip()
        if v:
            cfg.verbose = bool(int(v))
    def _edit_image_range(self):
        cfg = self.config
        v = input(f"Start image [{cfg.start_image}]: ").strip()
        if v:
            cfg.start_image = int(v)
        v = input(f"End image [{cfg.end_image}]: ").strip()
        if v:
            cfg.end_image = int(v)
    def _show_config(self):
        cfg = self.config
        print("\nТЕКУЩАЯ КОНФИГУРАЦИЯ\n")
        for k, v in vars(cfg).items():
            print(f"{k}: {v}")