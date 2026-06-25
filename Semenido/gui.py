import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from console import AppConfig


class AppGui:
    def __init__(self, run_pipeline_callback):
        self.config = AppConfig()
        self.run_pipeline_callback = run_pipeline_callback
        self.root = tk.Tk()
        self.root.title("Настройки проекта и обработка")
        self.root.geometry("1200x800")
        self._init_variables()
        self._create_widgets()

    def _init_variables(self):
        self.grid_rows = tk.IntVar(value=self.config.grid_rows)
        self.grid_cols = tk.IntVar(value=self.config.grid_cols)
        self.marker_step_mm = tk.DoubleVar(value=self.config.marker_step_mm)
        self.pca_roi_size = tk.IntVar(value=self.config.pca_roi_size)
        self.pca_corner_ratio = tk.DoubleVar(value=self.config.pca_corner_ratio)
        self.cluster_dist = tk.DoubleVar(value=self.config.cluster_dist)
        self.angle_thresh = tk.DoubleVar(value=self.config.angle_thresh)
        self.show_images = tk.BooleanVar(value=self.config.show_images)
        self.show_statistics = tk.BooleanVar(value=self.config.show_statistics)
        self.show_coordinate_systems = tk.BooleanVar(value=self.config.show_coordinate_systems)
        self.verbose = tk.BooleanVar(value=self.config.verbose)
        self.start_image = tk.IntVar(value=self.config.start_image)
        self.end_image = tk.IntVar(value=self.config.end_image)
        self.cross_mode = tk.StringVar(value=self.config.cross_mode)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = ttk.LabelFrame(main_frame, text=" Параметры конфигурации ", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        grid_frame = ttk.LabelFrame(left_panel, text="Размеры сетки", padding=5)
        grid_frame.pack(fill=tk.X, pady=5)
        self._add_row(grid_frame, "Rows:", self.grid_rows)
        self._add_row(grid_frame, "Cols:", self.grid_cols)
        self._add_row(grid_frame, "Step (mm):", self.marker_step_mm)
        det_frame = ttk.LabelFrame(left_panel, text="Настройки детектора", padding=5)
        det_frame.pack(fill=tk.X, pady=5)
        self._add_row(det_frame, "PCA ROI size:", self.pca_roi_size)
        self._add_row(det_frame, "PCA Corner ratio:", self.pca_corner_ratio)
        coord_frame = ttk.LabelFrame(left_panel, text="Координатные системы", padding=5)
        coord_frame.pack(fill=tk.X, pady=5)
        self._add_row(coord_frame, "Cluster dist:", self.cluster_dist)
        self._add_row(coord_frame, "Angle thresh:", self.angle_thresh)
        work_frame = ttk.LabelFrame(left_panel, text="Режим работы и кадры", padding=5)
        work_frame.pack(fill=tk.X, pady=5)
        self._add_row(work_frame, "Start image:", self.start_image)
        self._add_row(work_frame, "End image:", self.end_image)
        ttk.Label(work_frame, text="Cross mode:").pack(anchor=tk.W)
        ttk.Radiobutton(work_frame, text="grid", value="grid", variable=self.cross_mode).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(work_frame, text="single", value="single", variable=self.cross_mode).pack(anchor=tk.W, padx=20)
        view_frame = ttk.LabelFrame(left_panel, text="Отображение", padding=5)
        view_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(view_frame, text="Показывать изображения", variable=self.show_images).pack(anchor=tk.W)
        ttk.Checkbutton(view_frame, text="Показывать статистику", variable=self.show_statistics).pack(anchor=tk.W)
        ttk.Checkbutton(view_frame, text="Координатные системы", variable=self.show_coordinate_systems).pack(
            anchor=tk.W)
        ttk.Checkbutton(view_frame, text="Verbose (Логи)", variable=self.verbose).pack(anchor=tk.W)
        btn_start = ttk.Button(left_panel, text="ЗАПУСТИТЬ ОБРАБОТКУ", command=self._on_start_click)
        btn_start.pack(fill=tk.X, pady=15, ipady=5)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_label = ttk.Label(right_panel, text="42",
                                     anchor=tk.CENTER, relief="solid")
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=5)
        self.btn_next = ttk.Button(right_panel, text="Следующий кадр", state=tk.DISABLED, command=self._on_next_click)
        self.btn_next.pack(fill=tk.X, pady=2)
        self.next_frame_event = threading.Event()

    def _add_row(self, frame, label_text, text_var):
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label_text, width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=text_var, width=10).pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def _save_config(self):
        try:
            self.config.grid_rows = int(self.grid_rows.get())
            self.config.grid_cols = int(self.grid_cols.get())
            self.config.marker_step_mm = float(self.marker_step_mm.get())
            self.config.pca_roi_size = int(self.pca_roi_size.get())
            self.config.pca_corner_ratio = float(self.pca_corner_ratio.get())
            self.config.cluster_dist = float(self.cluster_dist.get())
            self.config.angle_thresh = float(self.angle_thresh.get())
            self.config.show_images = bool(self.show_images.get())
            self.config.show_statistics = bool(self.show_statistics.get())
            self.config.show_coordinate_systems = bool(self.show_coordinate_systems.get())
            self.config.verbose = bool(self.verbose.get())
            self.config.start_image = int(self.start_image.get())
            self.config.end_image = int(self.end_image.get())
            self.config.cross_mode = str(self.cross_mode.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте правильность ввода числовых параметров!")
            return False
        return True

    def _on_start_click(self):
        if not self._save_config():
            return
        self.next_frame_event.clear()
        threading.Thread(target=self.run_pipeline_callback, args=(self.config, self.update_image_ui),
                         daemon=True).start()

    def update_image_ui(self, img_bgr, title=""):
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((700, 500))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.root.after(0, self._set_image, img_tk)
        if self.config.show_images:
            self.root.after(0, lambda: self.btn_next.config(state=tk.NORMAL))
            self.next_frame_event.wait()
            self.next_frame_event.clear()
            self.root.after(0, lambda: self.btn_next.config(state=tk.DISABLED))
    def _set_image(self, img_tk):
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk
    def _on_next_click(self):
        self.next_frame_event.set()
    def mainloop(self):
        self.root.mainloop()