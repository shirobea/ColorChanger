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

        mode_frame = ttk.LabelFrame(control_frame, text="変換モード")
        mode_frame.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="we")
        mode_frame.columnconfigure(1, weight=1)
        self.mode_frame = mode_frame
        ttk.Label(mode_frame, text="モード").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.mode_var = tk.StringVar(value="Oklab")
        mode_box = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["なし", "RGB", "Lab (CIEDE2000)", "Oklab", "CMC(l:c)"],
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        mode_box.bind("<<ComboboxSelected>>", lambda *_: self._on_mode_changed())
        # RGB専用スライダー
        rgb_frame = ttk.LabelFrame(mode_frame, text="RGB重み（RGBモード限定）")
        rgb_frame.grid(row=1, column=0, columnspan=2, padx=4, pady=(4, 2), sticky="we")
        for col in range(4):
            rgb_frame.columnconfigure(col, weight=1)
        rgb_r_label = ttk.Label(rgb_frame, text="R 重み")
        rgb_r_label.grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.rgb_r_label = rgb_r_label
        rgb_r_scale = ttk.Scale(
            rgb_frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_r_weight_var,
            command=lambda *_: self._on_rgb_r_change(),
            length=140,
        )
        rgb_r_scale.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        rgb_r_scale.bind("<Button-1>", self._on_rgb_r_pointer)
        rgb_r_scale.bind("<B1-Motion>", self._on_rgb_r_pointer)
        self.rgb_r_scale = rgb_r_scale
        ttk.Label(rgb_frame, textvariable=self.rgb_r_display, width=6).grid(
            row=0, column=2, padx=2, pady=4, sticky="w"
        )
        rgb_g_label = ttk.Label(rgb_frame, text="G 重み")
        rgb_g_label.grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.rgb_g_label = rgb_g_label
        rgb_g_scale = ttk.Scale(
            rgb_frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_g_weight_var,
            command=lambda *_: self._on_rgb_g_change(),
            length=140,
        )
        rgb_g_scale.grid(row=1, column=1, padx=4, pady=4, sticky="we")
        rgb_g_scale.bind("<Button-1>", self._on_rgb_g_pointer)
        rgb_g_scale.bind("<B1-Motion>", self._on_rgb_g_pointer)
        self.rgb_g_scale = rgb_g_scale
        ttk.Label(rgb_frame, textvariable=self.rgb_g_display, width=6).grid(
            row=1, column=2, padx=2, pady=4, sticky="w"
        )
        rgb_b_label = ttk.Label(rgb_frame, text="B 重み")
        rgb_b_label.grid(row=2, column=0, padx=4, pady=4, sticky="e")
        self.rgb_b_label = rgb_b_label
        rgb_b_scale = ttk.Scale(
            rgb_frame,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.rgb_b_weight_var,
            command=lambda *_: self._on_rgb_b_change(),
            length=140,
        )
        rgb_b_scale.grid(row=2, column=1, padx=4, pady=4, sticky="we")
        rgb_b_scale.bind("<Button-1>", self._on_rgb_b_pointer)
        rgb_b_scale.bind("<B1-Motion>", self._on_rgb_b_pointer)
        self.rgb_b_scale = rgb_b_scale
        ttk.Label(rgb_frame, textvariable=self.rgb_b_display, width=6).grid(
            row=2, column=2, padx=2, pady=4, sticky="w"
        )
        self.rgb_frame = rgb_frame
        # CMC専用スライダー
        cmc_frame = ttk.LabelFrame(mode_frame, text="CMC(l:c)")
        cmc_frame.grid(row=2, column=0, columnspan=2, padx=4, pady=(2, 4), sticky="we")
        for col in range(4):
            cmc_frame.columnconfigure(col, weight=1)
        cmc_l_label = ttk.Label(cmc_frame, text="l（明るさ重み）")
        cmc_l_label.grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.cmc_l_label = cmc_l_label
        cmc_l_scale = ttk.Scale(
            cmc_frame,
            from_=0.5,
            to=3.0,
            orient="horizontal",
            variable=self.cmc_l_var,
            command=lambda *_: self._on_cmc_l_change(),
            length=140,
        )
        cmc_l_scale.grid(row=0, column=1, padx=4, pady=4, sticky="we")
        cmc_l_scale.bind("<Button-1>", self._on_cmc_l_pointer)
        cmc_l_scale.bind("<B1-Motion>", self._on_cmc_l_pointer)
        self.cmc_l_scale = cmc_l_scale
        ttk.Label(cmc_frame, textvariable=self.cmc_l_display, width=6).grid(
            row=0, column=2, padx=2, pady=4, sticky="w"
        )
        cmc_c_label = ttk.Label(cmc_frame, text="c（彩度重み）")
        cmc_c_label.grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.cmc_c_label = cmc_c_label
        cmc_c_scale = ttk.Scale(
            cmc_frame,
            from_=0.5,
            to=3.0,
            orient="horizontal",
            variable=self.cmc_c_var,
            command=lambda *_: self._on_cmc_c_change(),
            length=140,
        )
        cmc_c_scale.grid(row=1, column=1, padx=4, pady=4, sticky="we")
        cmc_c_scale.bind("<Button-1>", self._on_cmc_c_pointer)
        cmc_c_scale.bind("<B1-Motion>", self._on_cmc_c_pointer)
        self.cmc_c_scale = cmc_c_scale
        ttk.Label(cmc_frame, textvariable=self.cmc_c_display, width=6).grid(
            row=1, column=2, padx=2, pady=4, sticky="w"
        )
        self.cmc_frame = cmc_frame

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
            text="輪郭線強調（新方式）",
            variable=self.edge_enhance_var,
            command=self._on_edge_toggle,
        ).grid(row=9, column=0, padx=5, pady=5, sticky="w", columnspan=3)
        self.edge_label = ttk.Label(control_frame, text="輪郭強調の強さ")
        self.edge_label.grid(row=10, column=0, padx=5, pady=5, sticky="e")
        edge_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.edge_strength_var,
            command=lambda *_: self._on_edge_strength_change(),
            length=140,
        )
        edge_scale.grid(row=10, column=1, padx=5, pady=5, sticky="we", columnspan=2)
        edge_scale.bind("<Button-1>", self._on_edge_pointer)
        edge_scale.bind("<B1-Motion>", self._on_edge_pointer)
        self.edge_strength_scale = edge_scale
        ttk.Label(control_frame, textvariable=self.edge_strength_display, width=5).grid(
            row=10, column=3, padx=2, pady=5, sticky="w"
        )
        self.edge_thickness_label = ttk.Label(control_frame, text="輪郭の太さ")
        self.edge_thickness_label.grid(row=11, column=0, padx=5, pady=5, sticky="e")
        edge_thick_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.edge_thickness_var,
            command=lambda *_: self._on_edge_thickness_change(),
            length=140,
        )
        edge_thick_scale.grid(row=11, column=1, padx=5, pady=5, sticky="we", columnspan=2)
        edge_thick_scale.bind("<Button-1>", self._on_edge_thickness_pointer)
        edge_thick_scale.bind("<B1-Motion>", self._on_edge_thickness_pointer)
        self.edge_thickness_scale = edge_thick_scale
        ttk.Label(control_frame, textvariable=self.edge_thickness_display, width=5).grid(
            row=11, column=3, padx=2, pady=5, sticky="w"
        )
        self.edge_gain_label = ttk.Label(control_frame, text="輪郭ゲイン")
        self.edge_gain_label.grid(row=12, column=0, padx=5, pady=5, sticky="e")
        edge_gain_scale = ttk.Scale(
            control_frame,
            from_=0.0,
            to=5.0,
            orient="horizontal",
            variable=self.edge_gain_var,
            command=lambda *_: self._on_edge_gain_change(),
            length=140,
        )
        edge_gain_scale.grid(row=12, column=1, padx=5, pady=5, sticky="we", columnspan=2)
        edge_gain_scale.bind("<Button-1>", self._on_edge_gain_pointer)
        edge_gain_scale.bind("<B1-Motion>", self._on_edge_gain_pointer)
        self.edge_gain_scale = edge_gain_scale
        ttk.Label(control_frame, textvariable=self.edge_gain_display, width=5).grid(
            row=12, column=3, padx=2, pady=5, sticky="w"
        )
        self.edge_gamma_label = ttk.Label(control_frame, text="輪郭ガンマ")
        self.edge_gamma_label.grid(row=13, column=0, padx=5, pady=5, sticky="e")
        edge_gamma_scale = ttk.Scale(
            control_frame,
            from_=0.2,
            to=2.5,
            orient="horizontal",
            variable=self.edge_gamma_var,
            command=lambda *_: self._on_edge_gamma_change(),
            length=140,
        )
        edge_gamma_scale.grid(row=13, column=1, padx=5, pady=5, sticky="we", columnspan=2)
        edge_gamma_scale.bind("<Button-1>", self._on_edge_gamma_pointer)
        edge_gamma_scale.bind("<B1-Motion>", self._on_edge_gamma_pointer)
        self.edge_gamma_scale = edge_gamma_scale
        ttk.Label(control_frame, textvariable=self.edge_gamma_display, width=5).grid(
            row=13, column=3, padx=2, pady=5, sticky="w"
        )
        self.edge_saliency_label = ttk.Label(control_frame, text="サリエンシー寄与(%)")
        self.edge_saliency_label.grid(row=14, column=0, padx=5, pady=5, sticky="e")
        edge_saliency_scale = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.edge_saliency_weight_var,
            command=lambda *_: self._on_edge_saliency_change(),
            length=140,
        )
        edge_saliency_scale.grid(row=14, column=1, padx=5, pady=5, sticky="we", columnspan=2)
        edge_saliency_scale.bind("<Button-1>", self._on_edge_saliency_pointer)
        edge_saliency_scale.bind("<B1-Motion>", self._on_edge_saliency_pointer)
        self.edge_saliency_scale = edge_saliency_scale
        ttk.Label(control_frame, textvariable=self.edge_saliency_display, width=5).grid(
            row=14, column=3, padx=2, pady=5, sticky="w"
        )

        self.convert_button = ttk.Button(control_frame, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        self.progress_label = ttk.Label(control_frame, text="進捗: 0% (経過 0.0s)")
        self.progress_label.grid(row=15, column=3, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(control_frame, length=160)
        self.progress_bar.grid(row=16, column=3, padx=5, pady=5, sticky="w")

        self.save_button = ttk.Button(control_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=17, column=3, padx=5, pady=5, sticky="w")

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
        self.diff_label.grid(row=15, column=0, columnspan=4, padx=5, pady=(0, 5), sticky="we")

        self._update_adaptive_controls()
        self._update_pipeline_controls()
        self._update_num_colors_state()
        self._update_mode_frames()

    def _on_mode_changed(self: "BeadsApp") -> None:
        """モード変更時の付随UI更新（モード専用スライダーの表示/非表示）。"""
        self._update_mode_frames()

    def _is_cmc_mode(self: "BeadsApp") -> bool:
        """現在のモードがCMC(l:c)かどうかを判定する。"""
        return self.mode_var.get().upper().startswith("CMC")

    def _is_rgb_mode(self: "BeadsApp") -> bool:
        """現在のモードがRGBかどうかを判定する。"""
        return self.mode_var.get().upper() == "RGB"

    def _on_rgb_r_change(self: "BeadsApp") -> None:
        """R重みスライダー変更時の表示更新。"""
        val = float(self.rgb_r_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_r_weight_var.set(clamped)
        self.rgb_r_display.set(f"{clamped:.1f}")

    def _on_rgb_g_change(self: "BeadsApp") -> None:
        """G重みスライダー変更時の表示更新。"""
        val = float(self.rgb_g_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_g_weight_var.set(clamped)
        self.rgb_g_display.set(f"{clamped:.1f}")

    def _on_rgb_b_change(self: "BeadsApp") -> None:
        """B重みスライダー変更時の表示更新。"""
        val = float(self.rgb_b_weight_var.get())
        clamped = max(0.5, min(2.0, val))
        if clamped != val:
            self.rgb_b_weight_var.set(clamped)
        self.rgb_b_display.set(f"{clamped:.1f}")

    def _on_rgb_r_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        """R重みスライダーをクリック位置から設定。"""
        return self._set_scale_by_pointer(event, self.rgb_r_weight_var, self._on_rgb_r_change)

    def _on_rgb_g_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        """G重みスライダーをクリック位置から設定。"""
        return self._set_scale_by_pointer(event, self.rgb_g_weight_var, self._on_rgb_g_change)

    def _on_rgb_b_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        """B重みスライダーをクリック位置から設定。"""
        return self._set_scale_by_pointer(event, self.rgb_b_weight_var, self._on_rgb_b_change)

    def _update_rgb_weight_controls(self: "BeadsApp") -> None:
        """RGBモード時だけRGB重みスライダーを有効にする。"""
        is_rgb = self._is_rgb_mode()
        state_token = "!disabled" if is_rgb else "disabled"
        for scale_name in ("rgb_r_scale", "rgb_g_scale", "rgb_b_scale"):
            scale = getattr(self, scale_name, None)
            if scale:
                try:
                    scale.state([state_token])
                except Exception:
                    pass
        # ラベル色で無効状態を示す
        for attr in ("rgb_r_label", "rgb_g_label", "rgb_b_label"):
            lbl = getattr(self, attr, None)
            if lbl:
                lbl.configure(foreground="#000" if is_rgb else "#888")

    def _on_cmc_l_change(self: "BeadsApp") -> None:
        """CMCのl係数スライダー変更時の表示更新。"""
        val = float(self.cmc_l_var.get())
        clamped = max(0.5, min(3.0, val))
        if clamped != val:
            self.cmc_l_var.set(clamped)
        self.cmc_l_display.set(f"{clamped:.1f}")

    def _on_cmc_c_change(self: "BeadsApp") -> None:
        """CMCのc係数スライダー変更時の表示更新。"""
        val = float(self.cmc_c_var.get())
        clamped = max(0.5, min(3.0, val))
        if clamped != val:
            self.cmc_c_var.set(clamped)
        self.cmc_c_display.set(f"{clamped:.1f}")

    def _on_cmc_l_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        """CMC lスライダーをクリック位置から設定。"""
        return self._set_scale_by_pointer(event, self.cmc_l_var, self._on_cmc_l_change)

    def _on_cmc_c_pointer(self: "BeadsApp", event: "tk.Event") -> str:
        """CMC cスライダーをクリック位置から設定。"""
        return self._set_scale_by_pointer(event, self.cmc_c_var, self._on_cmc_c_change)

    def _update_cmc_controls(self: "BeadsApp") -> None:
        """CMCモード時だけスライダーを有効にする。"""
        is_cmc = self._is_cmc_mode()
        state_token = "!disabled" if is_cmc else "disabled"
        for scale_name in ("cmc_l_scale", "cmc_c_scale"):
            scale = getattr(self, scale_name, None)
            if scale:
                try:
                    scale.state([state_token])
                except Exception:
                    pass
        # ラベル色で無効状態を示す
        for lbl_name in ("cmc_l_label", "cmc_c_label"):
            lbl = getattr(self, lbl_name, None)
            if lbl:
                lbl.configure(foreground="#000" if is_cmc else "#888")

    def _update_mode_frames(self: "BeadsApp") -> None:
        """モードごとに関連フレームを表示/非表示にする。"""
        mode_upper = self.mode_var.get().upper()
        is_rgb = mode_upper == "RGB"
        is_cmc = mode_upper.startswith("CMC")
        # RGBフレーム
        if hasattr(self, "rgb_frame"):
            if is_rgb:
                self.rgb_frame.grid()
            else:
                self.rgb_frame.grid_remove()
        # CMCフレーム
        if hasattr(self, "cmc_frame"):
            if is_cmc:
                self.cmc_frame.grid()
            else:
                self.cmc_frame.grid_remove()
        # スライダー有効/無効を更新
        if hasattr(self, "_update_rgb_weight_controls"):
            self._update_rgb_weight_controls()
        self._update_cmc_controls()

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
