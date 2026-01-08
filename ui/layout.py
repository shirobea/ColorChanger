"""レイアウト組み立て専用のMixin（簡素版）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from .scale_utils import calc_scale_value_from_pointer

if TYPE_CHECKING:
    from .app import BeadsApp


class LayoutMixin:
    """ウィジェット生成と配置だけを担うメソッド群。"""

    def _build_layout(self: "BeadsApp") -> None:
        control_frame, preview_frame = self._create_frames()
        self._build_control_panel(control_frame)
        self._build_preview_panel(preview_frame)
        self._finalize_window_layout()

    def _create_frames(self: "BeadsApp") -> tuple[ttk.Frame, ttk.Frame]:
        # 左カラムにスクロール可能なキャンバスを用意し、設定項目が多くても隠れないようにする
        control_container = ttk.Frame(self.root)
        control_container.grid(row=0, column=0, sticky="nsew")
        control_container.rowconfigure(0, weight=1)
        control_container.columnconfigure(0, weight=1)

        control_canvas = tk.Canvas(control_container, highlightthickness=0, borderwidth=0)
        control_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        control_canvas.configure(yscrollcommand=vscroll.set)

        control_frame = ttk.Frame(control_canvas, padding=8)
        control_window = control_canvas.create_window((0, 0), window=control_frame, anchor="nw", tags=("control_window",))

        def _sync_scrollregion(_event: tk.Event) -> None:
            # 内容高さに応じてスクロール範囲を更新
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def _fit_inner_width(event: tk.Event) -> None:
            # キャンバス幅に合わせて内部フレームの幅を調整
            control_canvas.itemconfigure(control_window, width=event.width)

        control_frame.bind("<Configure>", _sync_scrollregion)
        control_canvas.bind("<Configure>", _fit_inner_width)

        self.control_container = control_container  # 幅固定/スクロール制御用に保持
        self.control_canvas = control_canvas
        self.control_frame = control_frame  # 左側の制御パネルを固定幅化するため保持

        preview_frame = ttk.Frame(self.root, padding=8)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        self.preview_frame = preview_frame

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=0)
        preview_frame.rowconfigure(1, weight=1)
        return control_frame, preview_frame

    def _build_control_panel(self: "BeadsApp", control_frame: ttk.Frame) -> None:
        control_frame.columnconfigure(0, weight=1)

        header = ttk.Frame(control_frame)
        header.grid(row=0, column=0, sticky="we", pady=(0, 6))
        header.columnconfigure(1, weight=1)
        ttk.Button(header, text="入力画像を選択", command=self.select_image).grid(
            row=0, column=0, padx=(0, 8), pady=2, sticky="w"
        )
        self.convert_button = ttk.Button(header, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=1, padx=(8, 0), pady=2, sticky="e")

        noise_frame = ttk.LabelFrame(control_frame, text="入力ノイズ除去")
        noise_frame.grid(row=1, column=0, padx=4, pady=(0, 6), sticky="we")
        noise_frame.columnconfigure(1, weight=1)
        ttk.Label(noise_frame, text="フィルタ").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        noise_filter_box = ttk.Combobox(
            noise_frame,
            textvariable=self.noise_filter_var,
            values=list(self._get_noise_filter_registry().keys()),
            state="readonly",
            width=18,
        )
        noise_filter_box.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        self.noise_filter_box = noise_filter_box
        ttk.Label(noise_frame, text="カーネル(奇数)").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        size_spin = ttk.Spinbox(
            noise_frame,
            from_=3,
            to=15,
            increment=2,
            textvariable=self.noise_filter_size_var,
            width=8,
            command=lambda: self._sanitize_kernel_size(self.noise_filter_size_var.get()),
        )
        size_spin.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        size_spin.bind("<FocusOut>", lambda *_: self._sanitize_kernel_size(self.noise_filter_size_var.get()))
        btn_row = ttk.Frame(noise_frame)
        btn_row.grid(row=2, column=0, columnspan=2, padx=4, pady=(2, 2), sticky="we")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        self.noise_apply_button = ttk.Button(btn_row, text="ノイズ除去", command=self.apply_noise_reduction)
        self.noise_apply_button.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="we")
        self.noise_reset_button = ttk.Button(btn_row, text="リセット", command=self.reset_noise_reduction)
        self.noise_reset_button.grid(row=0, column=1, padx=(4, 0), pady=2, sticky="we")

        mode_frame = ttk.LabelFrame(control_frame, text="変換モード")
        mode_frame.grid(row=2, column=0, padx=4, pady=(0, 6), sticky="we")
        mode_frame.columnconfigure(1, weight=1)
        ttk.Label(mode_frame, text="モード").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.mode_var = tk.StringVar(value="Oklab")
        mode_box = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["全て", "なし", "RGB", "Lab", "Hunter Lab", "Oklab", "CMC(l:c)"],
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        mode_box.bind("<<ComboboxSelected>>", lambda *_: self._on_mode_changed())

        lab_metric_frame = ttk.Frame(mode_frame)
        lab_metric_frame.grid(row=1, column=0, columnspan=2, padx=4, pady=(2, 2), sticky="we")
        lab_metric_frame.columnconfigure(1, weight=1)
        ttk.Label(lab_metric_frame, text="距離式").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        lab_metric_box = ttk.Combobox(
            lab_metric_frame,
            textvariable=self.lab_metric_var,
            values=["CIEDE2000", "CIE76", "CIE94"],
            state="readonly",
            width=18,
        )
        lab_metric_box.grid(row=0, column=1, padx=4, pady=2, sticky="we")
        self.lab_metric_frame = lab_metric_frame
        self.lab_metric_box = lab_metric_box

        rgb_frame = ttk.LabelFrame(mode_frame, text="RGB重み（RGBモード限定）")
        rgb_frame.grid(row=2, column=0, columnspan=2, padx=4, pady=(4, 2), sticky="we")
        for col in range(4):
            rgb_frame.columnconfigure(col, weight=1)
        self.rgb_frame = rgb_frame
        self._build_rgb_sliders(rgb_frame)

        cmc_frame = ttk.LabelFrame(mode_frame, text="CMC(l:c)")
        cmc_frame.grid(row=3, column=0, columnspan=2, padx=4, pady=(2, 4), sticky="we")
        for col in range(4):
            cmc_frame.columnconfigure(col, weight=1)
        self.cmc_frame = cmc_frame
        self._build_cmc_sliders(cmc_frame)

        size_frame = ttk.LabelFrame(control_frame, text="出力サイズ")
        size_frame.grid(row=3, column=0, padx=4, pady=(0, 6), sticky="we")
        size_frame.columnconfigure(1, weight=1)
        ttk.Label(size_frame, text="幅(px)").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        ttk.Spinbox(size_frame, from_=1, to=2048, textvariable=self.width_var, width=8).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(size_frame, text="高さ(px)").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        ttk.Spinbox(size_frame, from_=1, to=2048, textvariable=self.height_var, width=8).grid(
            row=1, column=1, padx=4, pady=4, sticky="w"
        )

        size_frame.columnconfigure(2, weight=1)
        ttk.Label(size_frame, text="リサイズ方式").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        resize_box = ttk.Combobox(
            size_frame,
            textvariable=self.resize_method_var,
            values=["ニアレストネイバー", "バイリニア", "バイキュービック"],
            state="readonly",
            width=18,
        )
        resize_box.grid(row=2, column=1, padx=4, pady=4, sticky="w")
        self.resize_box = resize_box

        aspect_row = ttk.Frame(size_frame)
        aspect_row.grid(row=3, column=0, columnspan=2, padx=4, pady=(2, 4), sticky="w")
        ttk.Checkbutton(
            aspect_row, text="比率固定", variable=self.lock_aspect_var, command=self._on_aspect_toggle
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Button(aspect_row, text="1/2", command=self._halve_size).grid(row=0, column=1, padx=(0, 4), sticky="w")
        ttk.Button(aspect_row, text="リセット", command=self._reset_size).grid(row=0, column=2, padx=(0, 4), sticky="w")

        ttk.Label(size_frame, textvariable=self.physical_size_var, foreground="#333").grid(
            row=4, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w"
        )
        ttk.Label(size_frame, textvariable=self.plate_requirement_var, foreground="#333").grid(
            row=5, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w"
        )

        progress_frame = ttk.Frame(control_frame)
        progress_frame.grid(row=4, column=0, padx=4, pady=(0, 6), sticky="we")
        progress_frame.columnconfigure(0, weight=1)
        self.progress_label = ttk.Label(progress_frame, text="進捗: 0% (経過 0.0s)")
        self.progress_label.grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")
        self.save_button = ttk.Button(progress_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=0, column=1, padx=(8, 0), pady=(0, 2), sticky="e")
        self.color_usage_button = ttk.Button(
            progress_frame,
            text="色使用一覧",
            command=self.show_color_usage,
            state="disabled",
        )
        self.color_usage_button.grid(row=0, column=2, padx=(8, 0), pady=(0, 2), sticky="e")
        self.progress_bar = ttk.Progressbar(progress_frame, length=200)
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="we")

        self.diff_label = ttk.Label(
            control_frame,
            textvariable=self.diff_var,
            anchor="w",
            justify="left",
            wraplength=320,
            foreground="#444",
            padding=(4, 2),
        )
        self.diff_label.grid(row=5, column=0, padx=4, pady=(0, 5), sticky="we")

        log_frame = ttk.LabelFrame(control_frame, text="処理ログ")
        log_frame.grid(row=6, column=0, padx=4, pady=(0, 6), sticky="we")
        log_frame.columnconfigure(0, weight=1)
        self.log_label = ttk.Label(
            log_frame,
            textvariable=self.rgb_log_var,
            anchor="w",
            justify="left",
            wraplength=320,
            foreground="#333",
            padding=(4, 2),
        )
        self.log_label.grid(row=0, column=0, padx=2, pady=2, sticky="we")

        self._update_mode_frames()

    def _build_rgb_sliders(self: "BeadsApp", frame: ttk.Frame) -> None:
        ttk.Label(frame, text="R 重み").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        r_scale = ttk.Scale(
            frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_r_weight_var,
            command=lambda *_: self._on_rgb_r_change(),
            length=140,
        )
        r_scale.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        r_scale.bind("<Button-1>", self._on_rgb_r_pointer)
        r_scale.bind("<B1-Motion>", self._on_rgb_r_pointer)
        self.rgb_r_scale = r_scale
        ttk.Label(frame, textvariable=self.rgb_r_display, width=6).grid(row=0, column=2, padx=2, pady=4, sticky="w")

        ttk.Label(frame, text="G 重み").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        g_scale = ttk.Scale(
            frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_g_weight_var,
            command=lambda *_: self._on_rgb_g_change(),
            length=140,
        )
        g_scale.grid(row=1, column=1, padx=4, pady=4, sticky="we")
        g_scale.bind("<Button-1>", self._on_rgb_g_pointer)
        g_scale.bind("<B1-Motion>", self._on_rgb_g_pointer)
        self.rgb_g_scale = g_scale
        ttk.Label(frame, textvariable=self.rgb_g_display, width=6).grid(row=1, column=2, padx=2, pady=4, sticky="w")

        ttk.Label(frame, text="B 重み").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        b_scale = ttk.Scale(
            frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_b_weight_var,
            command=lambda *_: self._on_rgb_b_change(),
            length=140,
        )
        b_scale.grid(row=2, column=1, padx=4, pady=4, sticky="we")
        b_scale.bind("<Button-1>", self._on_rgb_b_pointer)
        b_scale.bind("<B1-Motion>", self._on_rgb_b_pointer)
        self.rgb_b_scale = b_scale
        ttk.Label(frame, textvariable=self.rgb_b_display, width=6).grid(row=2, column=2, padx=2, pady=4, sticky="w")

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=3, column=0, columnspan=3, padx=4, pady=(2, 4), sticky="we")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        ttk.Button(btn_row, text="最適重み算出", command=self.compute_optimal_rgb_weights).grid(
            row=0, column=0, padx=(0, 4), sticky="we"
        )
        ttk.Button(btn_row, text="RGBリセット", command=self.reset_rgb_weights).grid(
            row=0, column=1, padx=(4, 0), sticky="we"
        )

    def _build_cmc_sliders(self: "BeadsApp", frame: ttk.Frame) -> None:
        ttk.Label(frame, text="l（明るさ重み）").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        l_scale = ttk.Scale(
            frame,
            from_=0.5,
            to=3.0,
            orient="horizontal",
            variable=self.cmc_l_var,
            command=lambda *_: self._on_cmc_l_change(),
            length=140,
        )
        l_scale.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        l_scale.bind("<Button-1>", self._on_cmc_l_pointer)
        l_scale.bind("<B1-Motion>", self._on_cmc_l_pointer)
        self.cmc_l_scale = l_scale
        ttk.Label(frame, textvariable=self.cmc_l_display, width=6).grid(row=0, column=2, padx=2, pady=4, sticky="w")

        ttk.Label(frame, text="c（彩度重み）").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        c_scale = ttk.Scale(
            frame,
            from_=0.5,
            to=3.0,
            orient="horizontal",
            variable=self.cmc_c_var,
            command=lambda *_: self._on_cmc_c_change(),
            length=140,
        )
        c_scale.grid(row=1, column=1, padx=4, pady=4, sticky="we")
        c_scale.bind("<Button-1>", self._on_cmc_c_pointer)
        c_scale.bind("<B1-Motion>", self._on_cmc_c_pointer)
        self.cmc_c_scale = c_scale
        ttk.Label(frame, textvariable=self.cmc_c_display, width=6).grid(row=1, column=2, padx=2, pady=4, sticky="w")

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=2, column=0, columnspan=3, padx=4, pady=(2, 4), sticky="we")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)
        ttk.Button(btn_row, text="最適重み算出", command=self.compute_optimal_cmc_weights).grid(
            row=0, column=0, padx=(0, 4), sticky="we"
        )
        ttk.Button(btn_row, text="CMCリセット", command=self.reset_cmc_weights).grid(
            row=0, column=1, padx=(4, 0), sticky="we"
        )

    def _on_mode_changed(self: "BeadsApp") -> None:
        self._update_mode_frames()
        # 変換前でもモード選択を保持する
        if hasattr(self, "_remember_mode_selection"):
            self._remember_mode_selection()

    def _is_cmc_mode(self: "BeadsApp") -> bool:
        return self.mode_var.get().upper().startswith("CMC")

    def _is_rgb_mode(self: "BeadsApp") -> bool:
        return self.mode_var.get().upper() == "RGB"

    def _is_lab_mode(self: "BeadsApp") -> bool:
        return self.mode_var.get().upper().startswith("LAB")

    def _on_rgb_r_change(self: "BeadsApp") -> None:
        val = float(self.rgb_r_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_r_weight_var.set(clamped)
        self.rgb_r_display.set(f"{clamped:.1f}")

    def _on_rgb_g_change(self: "BeadsApp") -> None:
        val = float(self.rgb_g_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_g_weight_var.set(clamped)
        self.rgb_g_display.set(f"{clamped:.1f}")

    def _on_rgb_b_change(self: "BeadsApp") -> None:
        val = float(self.rgb_b_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_b_weight_var.set(clamped)
        self.rgb_b_display.set(f"{clamped:.1f}")

    def _on_rgb_r_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        return self._set_scale_by_pointer(event, self.rgb_r_weight_var, self._on_rgb_r_change)

    def _on_rgb_g_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        return self._set_scale_by_pointer(event, self.rgb_g_weight_var, self._on_rgb_g_change)

    def _on_rgb_b_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        return self._set_scale_by_pointer(event, self.rgb_b_weight_var, self._on_rgb_b_change)

    def _update_rgb_weight_controls(self: "BeadsApp") -> None:
        is_rgb = self._is_rgb_mode()
        state_token = "!disabled" if is_rgb else "disabled"
        for scale_name in ("rgb_r_scale", "rgb_g_scale", "rgb_b_scale"):
            scale = getattr(self, scale_name, None)
            if scale:
                try:
                    scale.state([state_token])
                except Exception:
                    pass
        for attr in ("rgb_r_label", "rgb_g_label", "rgb_b_label"):
            lbl = getattr(self, attr, None)
            if lbl:
                lbl.configure(foreground="#000" if is_rgb else "#888")

    def _on_cmc_l_change(self: "BeadsApp") -> None:
        val = float(self.cmc_l_var.get())
        clamped = max(0.5, min(3.0, val))
        if clamped != val:
            self.cmc_l_var.set(clamped)
        self.cmc_l_display.set(f"{clamped:.1f}")

    def _on_cmc_c_change(self: "BeadsApp") -> None:
        val = float(self.cmc_c_var.get())
        clamped = max(0.5, min(3.0, val))
        if clamped != val:
            self.cmc_c_var.set(clamped)
        self.cmc_c_display.set(f"{clamped:.1f}")

    def _on_cmc_l_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        return self._set_scale_by_pointer(event, self.cmc_l_var, self._on_cmc_l_change)

    def _on_cmc_c_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        return self._set_scale_by_pointer(event, self.cmc_c_var, self._on_cmc_c_change)

    def _update_cmc_controls(self: "BeadsApp") -> None:
        is_cmc = self._is_cmc_mode()
        state_token = "!disabled" if is_cmc else "disabled"
        for scale_name in ("cmc_l_scale", "cmc_c_scale"):
            scale = getattr(self, scale_name, None)
            if scale:
                try:
                    scale.state([state_token])
                except Exception:
                    pass
        for lbl_name in ("cmc_l_label", "cmc_c_label"):
            lbl = getattr(self, lbl_name, None)
            if lbl:
                lbl.configure(foreground="#000" if is_cmc else "#888")

    def _update_mode_frames(self: "BeadsApp") -> None:
        mode_upper = self.mode_var.get().upper()
        is_rgb = mode_upper == "RGB"
        is_cmc = mode_upper.startswith("CMC")
        is_lab = mode_upper.startswith("LAB")
        if hasattr(self, "rgb_frame"):
            if is_rgb:
                self.rgb_frame.grid()
            else:
                self.rgb_frame.grid_remove()
        if hasattr(self, "cmc_frame"):
            if is_cmc:
                self.cmc_frame.grid()
            else:
                self.cmc_frame.grid_remove()
        if hasattr(self, "lab_metric_frame"):
            if is_lab:
                self.lab_metric_frame.grid()
            else:
                self.lab_metric_frame.grid_remove()
        self._update_rgb_weight_controls()
        self._update_cmc_controls()
        is_all = mode_upper == "全て"
        if hasattr(self, "_set_color_usage_button_state"):
            if is_all:
                self._set_color_usage_button_state(False)
            else:
                has_usage = bool(getattr(self, "color_usage", None))
                has_base = getattr(self, "_color_usage_base_image", None) is not None
                self._set_color_usage_button_state(has_usage and has_base)

    def _build_preview_panel(self: "BeadsApp", preview_frame: ttk.Frame) -> None:
        self.input_canvas = ttk.Label(preview_frame, text="入力画像", anchor="center")
        self.input_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.input_canvas.bind("<ButtonPress-1>", self._on_input_press)
        self.input_canvas.bind("<ButtonRelease-1>", self._on_input_release)
        self.input_canvas.bind("<Leave>", self._on_input_release)

        self.output_container = ttk.Frame(preview_frame)
        self.output_container.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.output_container.rowconfigure(0, weight=1)
        self.output_container.columnconfigure(0, weight=1)

        self.output_canvas = ttk.Label(self.output_container, text="変換後", anchor="center")
        self.output_canvas.grid(row=0, column=0, sticky="nsew")
        self.output_canvas.bind("<ButtonPress-1>", self._on_output_press)
        self.output_canvas.bind("<ButtonRelease-1>", self._on_output_release)
        self.output_canvas.bind("<Leave>", self._on_output_release)

        self.output_grid_frame = ttk.Frame(self.output_container)
        self.output_grid_frame.grid(row=0, column=0, sticky="nsew")
        self.output_grid_frame.grid_remove()
        for row in range(4):
            self.output_grid_frame.rowconfigure(row, weight=1)
        for col in range(4):
            self.output_grid_frame.columnconfigure(col, weight=1)
        self.output_grid_cells: list[dict[str, object]] = []
        for idx in range(8):
            row = idx // 2
            col = idx % 2
            cell = ttk.Frame(self.output_grid_frame)
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            cell.rowconfigure(0, weight=1)
            cell.columnconfigure(0, weight=1)
            image_label = ttk.Label(cell, text="")
            image_label.grid(row=0, column=0, sticky="nsew")
            caption_label = ttk.Label(cell, text="", anchor="center")
            caption_label.grid(row=1, column=0, pady=(2, 0), sticky="ew")
            self.output_grid_cells.append({"frame": cell, "image": image_label, "caption": caption_label})

        self.preview_frame.bind("<Configure>", self._on_preview_resize)

    def _finalize_window_layout(self: "BeadsApp") -> None:
        self.width_var.trace_add("write", lambda *_: self._on_width_changed())
        self.height_var.trace_add("write", lambda *_: self._on_height_changed())
        self.width_var.trace_add("write", lambda *_: self._update_physical_size_display())
        self.height_var.trace_add("write", lambda *_: self._update_physical_size_display())

        self.root.update_idletasks()
        self._lock_control_column_width()
        if not self._restored_geometry:
            init_w = self.root.winfo_width()
            init_h = self.root.winfo_height()
            self.root.geometry(f"{init_w}x{init_h}")
        self.root.grid_propagate(False)
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_keyboard_shortcuts()

    def _bind_keyboard_shortcuts(self: "BeadsApp") -> None:
        self.root.bind_all("<KeyPress-space>", self._on_space_key, add="+")
        self._disable_button_space_activation()

    def _disable_button_space_activation(self: "BeadsApp") -> None:
        def _consume_and_toggle(event: "tk.Event") -> str:
            self._on_space_key(event)
            return "break"

        for cls in (
            "Button",
            "TButton",
            "Checkbutton",
            "TCheckbutton",
            "Entry",
            "TEntry",
            "Spinbox",
            "TSpinbox",
        ):
            try:
                self.root.bind_class(cls, "<KeyPress-space>", _consume_and_toggle)
            except Exception:
                pass

    def _lock_control_column_width(self: "BeadsApp") -> None:
        """モード切替や長文表示で右側プレビューが揺れないよう左カラム幅を固定。"""
        ctrl = getattr(self, "control_frame", None)
        canvas = getattr(self, "control_canvas", None)
        if ctrl is None:
            return
        rgb_visible = self.rgb_frame.winfo_ismapped() if hasattr(self, "rgb_frame") else False
        cmc_visible = self.cmc_frame.winfo_ismapped() if hasattr(self, "cmc_frame") else False
        lab_visible = self.lab_metric_frame.winfo_ismapped() if hasattr(self, "lab_metric_frame") else False
        # 一度全て表示させて必要幅を計測
        if hasattr(self, "rgb_frame"):
            self.rgb_frame.grid()
        if hasattr(self, "cmc_frame"):
            self.cmc_frame.grid()
        if hasattr(self, "lab_metric_frame"):
            self.lab_metric_frame.grid()
        self.root.update_idletasks()
        required = max(ctrl.winfo_reqwidth(), 340)
        wrap_width = max(300, required - 24)
        # コントロールパネルの幅を固定しつつ高さは内容に任せる
        ctrl.configure(width=required)
        ctrl.grid_propagate(True)
        if canvas is not None:
            canvas.configure(width=required + 12)  # スクロールバー分の余白を確保
        self.root.columnconfigure(0, minsize=required, weight=0)
        if hasattr(self, "diff_label"):
            try:
                self.diff_label.configure(wraplength=wrap_width)
            except Exception:
                pass
        if hasattr(self, "log_label"):
            try:
                self.log_label.configure(wraplength=wrap_width)
            except Exception:
                pass
        # 元の表示状態に戻す
        if hasattr(self, "rgb_frame") and not rgb_visible:
            self.rgb_frame.grid_remove()
        if hasattr(self, "cmc_frame") and not cmc_visible:
            self.cmc_frame.grid_remove()
        if hasattr(self, "lab_metric_frame") and not lab_visible:
            self.lab_metric_frame.grid_remove()

    def _set_scale_by_pointer(
        self, event: "tk.Event", var: tk.Variable, on_change: callable
    ) -> str:
        new_val = calc_scale_value_from_pointer(event)
        var.set(new_val)
        on_change()
        return "break"
