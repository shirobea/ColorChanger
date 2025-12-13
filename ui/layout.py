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
            values=["なし", "RGB", "Lab (CIEDE2000)", "Oklab"],
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
        ttk.Label(
            control_frame,
            textvariable=self.physical_size_var,
            foreground="#333",
        ).grid(row=2, column=3, padx=5, pady=5, sticky="w")
        ttk.Label(
            control_frame,
            textvariable=self.plate_requirement_var,
            foreground="#333",
        ).grid(row=3, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(control_frame, text="リサイズ方式").grid(row=3, column=1, padx=5, pady=5, sticky="e")
        resize_box = ttk.Combobox(
            control_frame,
            textvariable=self.resize_method_var,
            values=["ニアレストネイバー", "バイリニア", "バイキュービック", "ブロック分割", "適応型ブロック分割"],
            state="readonly",
            width=18,
        )
        resize_box.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        resize_box.bind(
            "<<ComboboxSelected>>",
            lambda *_: (
                self._update_adaptive_controls(),
                self._update_pipeline_controls(),
            ),
        )
        self.resize_box = resize_box

        ttk.Label(control_frame, text="減色後の色数").grid(row=4, column=1, padx=5, pady=5, sticky="e")
        self.num_colors_var = tk.StringVar(value="64")
        num_spin = ttk.Spinbox(control_frame, from_=2, to=256, textvariable=self.num_colors_var, width=8)
        num_spin.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        self.num_colors_spin = num_spin

        ttk.Label(control_frame, text="減色方式").grid(row=5, column=1, padx=5, pady=5, sticky="e")
        quantize_box = ttk.Combobox(
            control_frame,
            textvariable=self.quantize_method_var,
            values=["なし", "K-means", "Wu減色"],
            state="readonly",
            width=18,
        )
        quantize_box.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        quantize_box.bind(
            "<<ComboboxSelected>>",
            lambda *_: (
                self._update_num_colors_state(),
                self._update_pipeline_controls(),
            ),
        )
        self.quantize_box = quantize_box

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
        pipeline_box.bind("<<ComboboxSelected>>", lambda *_: self._update_pipeline_controls())
        self.pipeline_box = pipeline_box

        self.hybrid_label = ttk.Label(control_frame, text="ハイブリッド縮小率(%)")
        self.hybrid_label.grid(row=8, column=1, padx=5, pady=5, sticky="e")
        hybrid_scale = ttk.Scale(
            control_frame,
            from_=10,
            to=100,
            orient="horizontal",
            variable=self.hybrid_scale_var,
            command=lambda *_: self._on_hybrid_scale_change(),
            length=140,
        )
        hybrid_scale.grid(row=8, column=2, padx=5, pady=5, sticky="we")
        hybrid_scale.bind("<Button-1>", self._on_hybrid_pointer)
        hybrid_scale.bind("<B1-Motion>", self._on_hybrid_pointer)
        self.hybrid_scale = hybrid_scale
        ttk.Label(control_frame, textvariable=self.hybrid_scale_display, width=5).grid(
            row=8, column=3, padx=2, pady=5, sticky="w"
        )

        ttk.Checkbutton(
            control_frame,
            text="輪郭線強調（サリエンシー利用）",
            variable=self.contour_enhance_var,
        ).grid(row=9, column=0, padx=5, pady=5, sticky="w", columnspan=3)

        self.convert_button = ttk.Button(control_frame, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        self.progress_label = ttk.Label(control_frame, text="進捗: 0% (経過 0.0s)")
        self.progress_label.grid(row=9, column=3, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(control_frame, length=160)
        self.progress_bar.grid(row=10, column=3, padx=5, pady=5, sticky="w")

        self.save_button = ttk.Button(control_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=11, column=3, padx=5, pady=5, sticky="w")

        # --- 重要度編集ツールバー ---
        edit_frame = ttk.LabelFrame(control_frame, text="重要度編集")
        edit_frame.grid(row=13, column=0, columnspan=4, padx=5, pady=(8, 5), sticky="we")
        # ボタン幅ばらつきを抑えるためカラムの最小幅を揃える
        for col in range(3):
            edit_frame.columnconfigure(col, weight=1, minsize=70)

        # 画像⇔重要度マップ切替ボタン（最上段）
        self.saliency_toggle_button = ttk.Button(
            edit_frame,
            text="画像切り替え（通常）",
            command=self.toggle_saliency_view,
        )
        self.saliency_toggle_button.grid(row=0, column=0, padx=4, pady=(4, 6), sticky="we", columnspan=3)

        self.pen_radio = ttk.Radiobutton(edit_frame, text="ペン（加算）", variable=self.brush_mode_var, value="add")
        self.pen_radio.grid(row=1, column=0, padx=4, pady=2, sticky="w")
        self.eraser_radio = ttk.Radiobutton(edit_frame, text="消しゴム（減算）", variable=self.brush_mode_var, value="erase")
        self.eraser_radio.grid(row=1, column=1, padx=4, pady=2, sticky="w")

        ttk.Label(edit_frame, text="半径(px)").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        radius_scale = ttk.Scale(
            edit_frame,
            from_=3,
            to=64,
            orient="horizontal",
            variable=self.brush_radius_var,
            command=lambda *_: self._on_brush_radius_change(),
            length=140,
        )
        radius_scale.grid(row=2, column=1, padx=4, pady=2, sticky="we")
        radius_scale.bind("<Button-1>", self._on_brush_radius_pointer)
        radius_scale.bind("<B1-Motion>", self._on_brush_radius_pointer)
        self.brush_radius_scale = radius_scale
        ttk.Label(edit_frame, textvariable=self.brush_radius_display, width=4).grid(
            row=2, column=2, padx=2, pady=2, sticky="w"
        )

        ttk.Label(edit_frame, text="強さ").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        strength_scale = ttk.Scale(
            edit_frame,
            from_=5,
            to=100,
            orient="horizontal",
            variable=self.brush_strength_var,
            command=lambda *_: self._on_brush_strength_change(),
            length=140,
        )
        strength_scale.grid(row=3, column=1, padx=4, pady=2, sticky="we")
        strength_scale.bind("<Button-1>", self._on_brush_strength_pointer)
        strength_scale.bind("<B1-Motion>", self._on_brush_strength_pointer)
        self.brush_strength_scale = strength_scale
        ttk.Label(edit_frame, textvariable=self.brush_strength_display, width=4).grid(
            row=3, column=2, padx=2, pady=2, sticky="w"
        )

        self.undo_button = ttk.Button(edit_frame, text="↶", command=self._undo_importance)
        self.undo_button.grid(row=4, column=0, padx=4, pady=(4, 2), sticky="we")
        self.redo_button = ttk.Button(edit_frame, text="↷", command=self._redo_importance)
        self.redo_button.grid(row=4, column=1, padx=4, pady=(4, 2), sticky="we")
        self.reset_imp_button = ttk.Button(edit_frame, text="リセット", command=self._reset_importance_edits)
        self.reset_imp_button.grid(row=4, column=2, padx=4, pady=(4, 2), sticky="we")

        self.fill_hot_button = ttk.Button(edit_frame, text="全重要(赤)", command=self._fill_all_hot)
        self.fill_hot_button.grid(row=5, column=0, padx=4, pady=(2, 4), sticky="we", columnspan=2)
        self.fill_cold_button = ttk.Button(edit_frame, text="全て非重要(青)", command=self._fill_all_cold)
        self.fill_cold_button.grid(row=5, column=2, padx=4, pady=(2, 4), sticky="we")

        # 設定差分の表示は重要度編集欄の下へまとめる
        self.diff_label = ttk.Label(
            control_frame,
            textvariable=self.diff_var,
            anchor="w",
            justify="left",
            wraplength=320,
            foreground="#444",
            padding=(4, 2),
        )
        self.diff_label.grid(row=14, column=0, columnspan=4, padx=5, pady=(0, 5), sticky="we")

        self._update_adaptive_controls()
        self._update_pipeline_controls()
        self._update_num_colors_state()

    def _build_preview_panel(self: "BeadsApp", preview_frame: ttk.Frame) -> None:
        """右側のプレビュー領域を組み立てる。"""
        self.input_canvas = ttk.Label(preview_frame, text="入力画像", anchor="center")
        self.input_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.input_canvas.bind("<ButtonPress-1>", self._on_input_press)
        self.input_canvas.bind("<B1-Motion>", self._on_input_drag)
        self.input_canvas.bind("<ButtonRelease-1>", self._on_input_release)

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
        self.width_var.trace_add("write", lambda *_: self._update_physical_size_display())
        self.height_var.trace_add("write", lambda *_: self._update_physical_size_display())

        self.root.update_idletasks()
        if not self._restored_geometry:
            init_w = self.root.winfo_width()
            init_h = self.root.winfo_height()
            self.root.geometry(f"{init_w}x{init_h}")
        self.root.grid_propagate(False)
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_keyboard_shortcuts()

    def _bind_keyboard_shortcuts(self: "BeadsApp") -> None:
        """キーボードショートカットのバインドをまとめる。"""
        # add="+" で既存バインドを壊さずスペースキーを監視する
        self.root.bind_all("<KeyPress-space>", self._on_space_key, add="+")
        self._disable_button_space_activation()

    def _disable_button_space_activation(self: "BeadsApp") -> None:
        """スペースキーによる既定のウィジェット動作を無効化する。"""
        # ボタンやチェックボックスのスペースによるactivateと、Entry/Spinboxへの空白入力を抑止しつつ
        # グローバルなスペースショートカット（変換開始/中止）は発火させる。
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
