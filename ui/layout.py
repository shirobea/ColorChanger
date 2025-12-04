"""レイアウト組み立て専用のMixin。BeadsAppに継承させてUI構築を分離。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import BeadsApp


class LayoutMixin:
    """ウィジェット生成と配置だけを担うメソッド群。"""

    def _build_layout(self: "BeadsApp") -> None:
        """全体レイアウトの組み立てエントリ。"""
        control_frame, preview_frame = self._create_frames()
        self._build_control_panel(control_frame)
        self._build_preview_panel(preview_frame)
        self._finalize_window_layout()

    def _create_frames(self: "BeadsApp") -> tuple[ttk.Frame, ttk.Frame]:
        """フレーム生成と基本グリッド設定をまとめる。"""
        control_frame = ttk.Frame(self.root, padding=8)
        control_frame.grid(row=0, column=0, sticky="ns")

        preview_frame = ttk.Frame(self.root, padding=8)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        self.preview_frame = preview_frame

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=0)  # 差分テキスト行
        preview_frame.rowconfigure(1, weight=1)  # 画像行
        return control_frame, preview_frame

    def _build_control_panel(self: "BeadsApp", control_frame: ttk.Frame) -> None:
        """左側の操作パネルを組み立てる。"""
        ttk.Button(control_frame, text="入力画像を選択", command=self.select_image).grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="変換モード").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.mode_var = tk.StringVar(value="Oklab")
        mode_box = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["RGB", "Lab (CIEDE2000)", "Oklab"],
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(control_frame, text="幅(px)").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.Spinbox(control_frame, from_=1, to=2048, textvariable=self.width_var, width=8).grid(
            row=1, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="高さ(px)").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        ttk.Spinbox(control_frame, from_=1, to=2048, textvariable=self.height_var, width=8).grid(
            row=1, column=3, padx=5, pady=5, sticky="w"
        )

        ttk.Checkbutton(
            control_frame, text="比率固定", variable=self.lock_aspect_var, command=self._on_aspect_toggle
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")

        ttk.Button(control_frame, text="1/2", command=self._halve_size).grid(
            row=2, column=1, padx=5, pady=5, sticky="w"
        )
        ttk.Button(control_frame, text="リセット", command=self._reset_size).grid(
            row=2, column=2, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="減色後の色数").grid(row=3, column=1, padx=5, pady=5, sticky="e")
        self.num_colors_var = tk.StringVar(value="64")
        num_spin = ttk.Spinbox(control_frame, from_=2, to=256, textvariable=self.num_colors_var, width=8)
        num_spin.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.num_colors_spin = num_spin

        ttk.Label(control_frame, text="減色方式").grid(row=4, column=1, padx=5, pady=5, sticky="e")
        quantize_box = ttk.Combobox(
            control_frame,
            textvariable=self.quantize_method_var,
            values=["なし", "K-means", "Wu減色"],
            state="readonly",
            width=18,
        )
        quantize_box.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        quantize_box.bind(
            "<<ComboboxSelected>>",
            lambda *_: (
                self._update_num_colors_state(),
                self._update_pipeline_controls(),
            ),
        )
        self.quantize_box = quantize_box

        ttk.Label(control_frame, text="分割方式").grid(row=5, column=1, padx=5, pady=5, sticky="e")
        division_box = ttk.Combobox(
            control_frame,
            textvariable=self.division_method_var,
            values=["なし", "ブロック分割", "適応型ブロック分割"],
            state="readonly",
            width=18,
        )
        division_box.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        division_box.bind(
            "<<ComboboxSelected>>",
            lambda *_: (
                self._update_adaptive_controls(),
                self._update_pipeline_controls(),
            ),
        )
        self.division_box = division_box

        self.adaptive_label = ttk.Label(control_frame, text="細かさ（顔まわり）")
        self.adaptive_label.grid(row=6, column=1, padx=5, pady=5, sticky="e")
        adaptive_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.adaptive_weight_var,
            command=lambda *_: self._on_adaptive_weight_change(),
            length=140,
        )
        adaptive_scale.grid(row=6, column=2, padx=5, pady=5, sticky="we")
        adaptive_scale.bind("<Button-1>", self._on_adaptive_pointer)
        adaptive_scale.bind("<B1-Motion>", self._on_adaptive_pointer)
        self.adaptive_scale = adaptive_scale
        ttk.Label(control_frame, textvariable=self.adaptive_weight_display, width=5).grid(
            row=6, column=3, padx=2, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="処理順序").grid(row=7, column=1, padx=5, pady=5, sticky="e")
        self.pipeline_var = tk.StringVar(value="リサイズ→減色")
        pipeline_box = ttk.Combobox(
            control_frame,
            textvariable=self.pipeline_var,
            values=["リサイズ→減色", "減色→リサイズ", "ハイブリッド"],
            state="readonly",
            width=18,
        )
        pipeline_box.grid(row=7, column=2, padx=5, pady=5, sticky="w")
        self.pipeline_box = pipeline_box

        ttk.Checkbutton(
            control_frame,
            text="輪郭線強調（サリエンシー利用）",
            variable=self.contour_enhance_var,
        ).grid(row=8, column=0, padx=5, pady=5, sticky="w", columnspan=3)

        self.convert_button = ttk.Button(control_frame, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        self.progress_label = ttk.Label(control_frame, text="進捗: 0% (経過 0.0s)")
        self.progress_label.grid(row=8, column=3, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(control_frame, length=160)
        self.progress_bar.grid(row=9, column=3, padx=5, pady=5, sticky="w")

        self.save_button = ttk.Button(control_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=10, column=3, padx=5, pady=5, sticky="w")
        self._update_adaptive_controls()
        self._update_pipeline_controls()
        self._update_num_colors_state()

    def _build_preview_panel(self: "BeadsApp", preview_frame: ttk.Frame) -> None:
        """右側のプレビュー領域を組み立てる。"""
        self.saliency_toggle_button = ttk.Button(
            preview_frame,
            text="画像切り替え（通常）",
            command=self.toggle_saliency_view,
        )
        self.saliency_toggle_button.grid(row=0, column=0, padx=5, pady=(0, 4), sticky="nw")

        self.diff_label = ttk.Label(
            preview_frame,
            textvariable=self.diff_var,
            anchor="e",
            justify="left",
            wraplength=520,
            foreground="#444",
            padding=(3, 0),
        )
        self.diff_label.grid(row=0, column=1, padx=(10, 5), pady=(0, 4), sticky="ne")

        self.input_canvas = ttk.Label(preview_frame, text="入力画像", anchor="center")
        self.input_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.output_canvas = ttk.Label(preview_frame, text="変換後", anchor="center")
        self.output_canvas.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.output_canvas.bind("<ButtonPress-1>", self._on_output_press)
        self.output_canvas.bind("<ButtonRelease-1>", self._on_output_release)
        self.output_canvas.bind("<Leave>", self._on_output_release)

        self.preview_frame.bind("<Configure>", self._on_preview_resize)
        self._set_saliency_button_state(enabled=False)

    def _finalize_window_layout(self: "BeadsApp") -> None:
        """ウィンドウ設定の後処理をまとめる。"""
        self.width_var.trace_add("write", lambda *_: self._on_width_changed())
        self.height_var.trace_add("write", lambda *_: self._on_height_changed())

        self.root.update_idletasks()
        if not self._restored_geometry:
            init_w = self.root.winfo_width()
            init_h = self.root.winfo_height()
            self.root.geometry(f"{init_w}x{init_h}")
        self.root.grid_propagate(False)
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
