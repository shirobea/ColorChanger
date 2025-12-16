"""Tkinter UI for beads palette conversion (simplified)."""

from __future__ import annotations

import threading
import time
import json
from pathlib import Path
from typing import Optional
from math import ceil

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from palette import BeadPalette
from .controller import ConversionRunner
from .models import ConversionRequest
from .layout import LayoutMixin
from .actions import ActionsMixin
from .state import StateMixin
from .preview import PreviewMixin


class BeadsApp(LayoutMixin, ActionsMixin, StateMixin, PreviewMixin):
    """Main application window (必要機能のみに絞った版)。"""

    PIXEL_SIZE_MM = 2.6
    PLATE_PIXELS = 28

    def __init__(self, root: tk.Tk, palette: BeadPalette) -> None:
        self.root = root
        self.palette = palette
        self._window_state_path = Path(__file__).resolve().parent / "window_state.json"
        self._settings_path = Path(__file__).resolve().parent / "settings.json"
        self._restored_geometry = self._load_window_state()
        self._last_geometry: Optional[tuple[int, int, int, int]] = None
        self.input_image_path: Optional[Path] = None
        self.output_image: Optional[np.ndarray] = None
        self.output_path: Optional[Path] = None
        self.input_pil: Optional[Image.Image] = None
        self.output_pil: Optional[Image.Image] = None
        self._input_photo: Optional[ImageTk.PhotoImage] = None
        self._output_photo: Optional[ImageTk.PhotoImage] = None
        self.prev_output_pil: Optional[Image.Image] = None
        self.prev_settings: Optional[dict] = None
        self.last_settings: Optional[dict] = None
        self._pending_settings: Optional[dict] = None
        self.diff_var = tk.StringVar(value="")
        self.physical_size_var = tk.StringVar(value="完成サイズ: 幅・高さを入力してください")
        self.plate_requirement_var = tk.StringVar(value="28×28プレート: 幅・高さを入力してください")
        self.original_size: Optional[tuple[int, int]] = None
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self.status_var = tk.StringVar(value="")
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.resize_method_var = tk.StringVar(value="ニアレストネイバー")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.cmc_l_var = tk.DoubleVar(value=2.0)
        self.cmc_c_var = tk.DoubleVar(value=1.0)
        self.cmc_l_display = tk.StringVar(value="2.0")
        self.cmc_c_display = tk.StringVar(value="1.0")
        self.rgb_r_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_g_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_b_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_r_display = tk.StringVar(value="1.0")
        self.rgb_g_display = tk.StringVar(value="1.0")
        self.rgb_b_display = tk.StringVar(value="1.0")
        self.lab_metric_var = tk.StringVar(value="CIEDE2000")
        self._updating_size_fields = False
        self._closing = False  # 終了処理中フラグ
        self._showing_prev: bool = False
        self._runner = ConversionRunner(self._schedule_on_ui, lambda: self._closing)

        self._load_settings()
        self._build_layout()
        self._apply_saved_settings()

    # --- 設定復元と差分表示 ---
    def _apply_saved_settings(self) -> None:
        if not self.last_settings:
            return

        def _sanitize_choice(value: Optional[str], allowed: set[str], fallback: str) -> str:
            if value in allowed:
                return value  # type: ignore[return-value]
            return fallback

        allowed_resize = {"ニアレストネイバー", "バイリニア", "バイキュービック"}
        allowed_lab_metric = {"CIEDE2000", "CIE76", "CIE94"}
        self.width_var.set("")
        self.height_var.set("")
        mode = self.last_settings.get("モード")
        if mode:
            self.mode_var.set(mode)
        resize_label = _sanitize_choice(self.last_settings.get("リサイズ方式"), allowed_resize, self.resize_method_var.get())
        self.resize_method_var.set(resize_label)
        lab_metric = _sanitize_choice(self.last_settings.get("Lab距離式"), allowed_lab_metric, self.lab_metric_var.get())
        self.lab_metric_var.set(lab_metric)
        try:
            cmc_l = self.last_settings.get("CMC l")
            if cmc_l is not None:
                l_val = float(cmc_l)
                self.cmc_l_var.set(l_val)
                self.cmc_l_display.set(f"{max(0.5, min(3.0, l_val)):.1f}")
        except Exception:
            pass
        try:
            cmc_c = self.last_settings.get("CMC c")
            if cmc_c is not None:
                c_val = float(cmc_c)
                self.cmc_c_var.set(c_val)
                self.cmc_c_display.set(f"{max(0.5, min(3.0, c_val)):.1f}")
        except Exception:
            pass
        try:
            rgb_w = self.last_settings.get("RGB重み")
            if isinstance(rgb_w, (list, tuple)) and len(rgb_w) == 3:
                r, g, b = (float(x) for x in rgb_w)
                self.rgb_r_weight_var.set(r)
                self.rgb_g_weight_var.set(g)
                self.rgb_b_weight_var.set(b)
                self.rgb_r_display.set(f"{max(0.5, min(2.0, r)):.1f}")
                self.rgb_g_display.set(f"{max(0.5, min(2.0, g)):.1f}")
                self.rgb_b_display.set(f"{max(0.5, min(2.0, b)):.1f}")
        except Exception:
            pass
        sanitized_settings = dict(self.last_settings)
        sanitized_settings.setdefault("CMC l", f"{float(self.cmc_l_var.get()):.1f}")
        sanitized_settings.setdefault("CMC c", f"{float(self.cmc_c_var.get()):.1f}")
        sanitized_settings.setdefault(
            "RGB重み",
            [
                float(self.rgb_r_weight_var.get()),
                float(self.rgb_g_weight_var.get()),
                float(self.rgb_b_weight_var.get()),
            ],
        )
        sanitized_settings.setdefault("リサイズ方式", self.resize_method_var.get())
        sanitized_settings.setdefault("モード", self.mode_var.get())
        sanitized_settings.setdefault("Lab距離式", self.lab_metric_var.get())
        self.last_settings = sanitized_settings
        self._update_mode_frames()

    def _build_diff_overlay(self) -> str:
        if not self.last_settings or not self.prev_settings:
            return "変更された設定: なし"
        diffs: list[str] = []
        for key, prev_val in self.prev_settings.items():
            last_val = self.last_settings.get(key)
            if last_val != prev_val:
                diffs.append(f"{key}: {prev_val} → {last_val}")
        if not diffs:
            return "変更された設定: なし"
        return "変更された設定: " + " / " .join(diffs)

    # --- サイズ計算系 ---
    def _parse_int(self, value: str) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    def _set_size_fields(self, width: int, height: int) -> None:
        self._updating_size_fields = True
        self.width_var.set(str(width))
        self.height_var.set(str(height))
        self._updating_size_fields = False
        self._update_physical_size_display()

    def _update_physical_size_display(self) -> None:
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width and height:
            mm_w = width * self.PIXEL_SIZE_MM
            mm_h = height * self.PIXEL_SIZE_MM
            text = f"完成サイズ: 約 {mm_w/10:.1f}cm × {mm_h/10:.1f}cm"
            plates_x = ceil(width / self.PLATE_PIXELS)
            plates_y = ceil(height / self.PLATE_PIXELS)
            plates_total = plates_x * plates_y
            plate_text = f"28×28プレート: {plates_x} × {plates_y} 枚 (合計 {plates_total} 枚)"
        else:
            text = "完成サイズ: 幅・高さを入力してください"
            plate_text = "28×28プレート: 幅・高さを入力してください"
        self.physical_size_var.set(text)
        self.plate_requirement_var.set(plate_text)

    def _set_height_from_width(self, width: int) -> None:
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_h = max(1, int(round(orig_h / orig_w * width)))
        self._set_size_fields(width, new_h)

    def _set_width_from_height(self, height: int) -> None:
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_w = max(1, int(round(orig_w / orig_h * height)))
        self._set_size_fields(new_w, height)

    def _on_width_changed(self) -> None:
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        width = self._parse_int(self.width_var.get())
        if width and width > 0:
            self._set_height_from_width(width)

    def _on_height_changed(self) -> None:
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        height = self._parse_int(self.height_var.get())
        if height and height > 0:
            self._set_width_from_height(height)

    def _on_aspect_toggle(self) -> None:
        if not self.lock_aspect_var.get() or not self.original_size:
            return
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width and width > 0:
            self._set_height_from_width(width)
        elif height and height > 0:
            self._set_width_from_height(height)

    def _set_initial_target_size(self, image: Image.Image) -> None:
        img_w, img_h = image.size
        self._set_size_fields(img_w, img_h)

    def _halve_size(self) -> None:
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width is None or height is None:
            self.status_var.set("幅と高さは整数で入力してください。")
            return
        new_w = max(1, width // 2)
        new_h = max(1, height // 2)
        self._set_size_fields(new_w, new_h)
        if self.lock_aspect_var.get() and self.original_size:
            self._set_height_from_width(new_w)

    def _reset_size(self) -> None:
        if not self.original_size:
            self.status_var.set("先に入力画像を選択してください。")
            return
        orig_w, orig_h = self.original_size
        self._set_size_fields(orig_w, orig_h)

    # --- キーボード/プレビュー ---
    def _on_preview_resize(self, _event: tk.Event) -> None:
        self._refresh_previews()

    def _on_space_key(self, event: tk.Event) -> str:
        return super()._on_space_key(event)

    def _cancel_worker_safely(self, timeout: float = 2.0) -> None:
        """変換スレッドをキャンセルし、短時間だけ待機する。"""
        self._runner.cancel_and_wait(timeout=timeout)

    # --- 終了時 ---
    def _on_close(self) -> None:
        self._closing = True
        self._cancel_worker_safely()
        self._save_window_state()
        self.root.destroy()
