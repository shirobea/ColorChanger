"""Tkinter UI for beads palette conversion."""

import threading
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import converter
from palette import BeadPalette
from .models import ConversionRequest
from .layout import LayoutMixin
from .actions import ActionsMixin
from .state import StateMixin
from .preview import PreviewMixin


class BeadsApp(LayoutMixin, ActionsMixin, StateMixin, PreviewMixin):
    """Main application window."""

    def __init__(self, root: tk.Tk, palette: BeadPalette) -> None:
        self.root = root
        self.palette = palette
        # ウィンドウサイズと位置を保持するファイルパス
        self._window_state_path = Path(__file__).resolve().parent / "window_state.json"
        # 前回の設定を保存するパス
        self._settings_path = Path(__file__).resolve().parent / "settings.json"
        # 過去のウィンドウ配置を復元できたかのフラグ
        self._restored_geometry = self._load_window_state()
        # 直近のジオメトリ情報を控えておく
        self._last_geometry: Optional[tuple[int, int, int, int]] = None
        self.input_image_path: Optional[Path] = None
        self.output_image: Optional[np.ndarray] = None
        self.output_path: Optional[Path] = None
        self._input_photo: Optional[ImageTk.PhotoImage] = None
        self._output_photo: Optional[ImageTk.PhotoImage] = None
        self.prev_output_pil: Optional[Image.Image] = None
        self._prev_output_photo: Optional[ImageTk.PhotoImage] = None
        self._showing_prev: bool = False
        self.last_settings: Optional[dict] = None
        self.prev_settings: Optional[dict] = None
        self._pending_settings: Optional[dict] = None
        self.diff_var = tk.StringVar(value="")
        self.input_pil: Optional[Image.Image] = None
        self.output_pil: Optional[Image.Image] = None
        self.saliency_map: Optional[np.ndarray] = None  # サリエンシーマップ（変換処理向け）
        self.importance_map: Optional[np.ndarray] = None  # 顔強調込みの重要度マップ（表示・適応ブロック用）
        self.saliency_overlay_pil: Optional[Image.Image] = None  # 重要度マップをカラー化したプレビュー
        self._show_saliency: bool = False
        self.original_size: Optional[tuple[int, int]] = None
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        # ステータス表示は使わないので空文字で保持のみ
        self.status_var = tk.StringVar(value="")
        # 初期値は空欄にしておき、画像読込時に解像度を自動反映
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.contour_enhance_var = tk.BooleanVar(value=True)
        self.adaptive_weight_var = tk.DoubleVar(value=50.0)
        self.adaptive_weight_display = tk.StringVar(value="50")
        self.quantize_method_var = tk.StringVar(value="Wu減色")
        self.division_method_var = tk.StringVar(value="なし")
        self._updating_size_fields = False

        # 前回設定の復元
        self._load_settings()
        self._build_layout()


    
    def _compute_and_store_importance(self, image: Image.Image) -> None:
        """入力画像読込直後にサリエンシー＋顔重みを統合した重要度マップを計算し、オーバーレイを作る。"""
        self.status_var.set("重要度マップを生成中...")
        pil_for_sal = image
        max_side = 720
        if max(image.size) > max_side:
            # 計算負荷を抑えるために長辺を720pxに収めて計算
            pil_for_sal = image.copy()
            pil_for_sal.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        np_img = np.array(pil_for_sal, dtype=np.uint8)
        try:
            saliency_small = converter.compute_saliency_map(np_img)
            importance_small = converter.compute_importance_map(
                np_img, saliency_map=saliency_small, eye_importance_scale=0.8
            )
        except Exception as exc:
            self.status_var.set(f"重要度マップ生成に失敗しました: {exc}")
            self.saliency_map = None
            self.importance_map = None
            self.saliency_overlay_pil = None
            self._set_saliency_button_state(enabled=False)
            return

        saliency_full = cv2.resize(saliency_small, image.size, interpolation=cv2.INTER_LINEAR)
        saliency_full = np.clip(saliency_full.astype(np.float32), 0.0, 1.0)
        self.saliency_map = saliency_full
        importance_full = cv2.resize(importance_small, image.size, interpolation=cv2.INTER_LINEAR)
        importance_full = np.clip(importance_full.astype(np.float32), 0.0, 1.0)
        self.importance_map = importance_full
        self.saliency_overlay_pil = self._build_saliency_overlay(image, importance_full)
        self._set_saliency_button_state(enabled=True)
        self.status_var.set("重要度マップを生成しました。")


    
    def _build_diff_overlay(self) -> str:
        """最新と1つ前の設定差分だけを整形して返す。"""
        if not self.last_settings or not self.prev_settings:
            return "変更された設定: なし"
        diffs: list[str] = []
        for key, prev_val in self.prev_settings.items():
            last_val = self.last_settings.get(key)
            if last_val != prev_val:
                diffs.append(f"{key}: {prev_val} → {last_val}")
        if not diffs:
            return "変更された設定: なし"
        return "変更された設定: " + " / ".join(diffs)

    def _refresh_previews(self) -> None:
        """Render previews using nearest-neighbor to avoid blur and fit area."""
        self.root.update_idletasks()
        frame_w = self.preview_frame.winfo_width() or self.preview_frame.winfo_reqwidth() or 400
        frame_h = self.preview_frame.winfo_height() or self.preview_frame.winfo_reqheight() or 300
        cell_w = max(1, (frame_w - 20) // 2)
        cell_h = max(1, frame_h - 20)

        if self.input_pil:
            input_source = self.saliency_overlay_pil if (self._show_saliency and self.saliency_overlay_pil) else self.input_pil
            self._input_photo = self._resize_to_box(input_source, cell_w, cell_h)
            if self._input_photo:
                caption = "サリエンシー重ね表示" if self._show_saliency and self.saliency_overlay_pil else ""
                self.input_canvas.configure(image=self._input_photo, text=caption)
        display_pil = self.prev_output_pil if self._showing_prev else self.output_pil
        if display_pil:
            photo = self._resize_to_box(display_pil, cell_w, cell_h)
            if photo:
                if self._showing_prev:
                    self._prev_output_photo = photo
                    self.output_canvas.configure(image=self._prev_output_photo, text="1つ前の出力")
                else:
                    self._output_photo = photo
                    self.output_canvas.configure(image=self._output_photo, text="")
        else:
            self.output_canvas.configure(image="", text="変換後")

    def _resize_to_box(self, image: Image.Image, box_w: int, box_h: int) -> Optional[ImageTk.PhotoImage]:
        """Resize image to fit inside given box using nearest neighbor (no blur)."""
        img_w, img_h = image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = image.resize(new_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(resized)

    # --- 出力プレビュー比較（長押し） ---
    def _on_output_press(self, _event: tk.Event) -> None:
        """出力画像ラベルを押したら即座に1つ前の出力へ切り替える。"""
        if not self.prev_output_pil or not self.output_pil:
            return
        self._show_previous_output()

    def _on_output_release(self, _event: tk.Event) -> None:
        """指を離したら最新の出力表示に戻す。"""
        if self._showing_prev:
            self._showing_prev = False
            self._refresh_previews()

    def _show_previous_output(self) -> None:
        """実際に前回出力へ切り替えて再描画する。"""
        if not self.prev_output_pil:
            return
        self._showing_prev = True
        self._refresh_previews()

    # --- サイズ関連のユーティリティ ---
    def _parse_int(self, value: str) -> Optional[int]:
        """文字列を整数に変換（失敗時はNone）。"""
        try:
            return int(value)
        except ValueError:
            return None

    def _set_size_fields(self, width: int, height: int) -> None:
        """フィールド更新時の無限ループを避けながら数値を反映。"""
        self._updating_size_fields = True
        self.width_var.set(str(max(1, width)))
        self.height_var.set(str(max(1, height)))
        self._updating_size_fields = False

    def _set_height_from_width(self, width: int) -> None:
        """幅を基準に縦横比を保って高さを算出。"""
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_h = max(1, int(round(orig_h / orig_w * width)))
        self._set_size_fields(width, new_h)

    def _set_width_from_height(self, height: int) -> None:
        """高さを基準に縦横比を保って幅を算出。"""
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_w = max(1, int(round(orig_w / orig_h * height)))
        self._set_size_fields(new_w, height)

    def _on_width_changed(self) -> None:
        """幅入力時に比率固定なら高さを追従させる。"""
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        width = self._parse_int(self.width_var.get())
        if width and width > 0:
            self._set_height_from_width(width)

    def _on_height_changed(self) -> None:
        """高さ入力時に比率固定なら幅を追従させる。"""
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        height = self._parse_int(self.height_var.get())
        if height and height > 0:
            self._set_width_from_height(height)

    def _on_aspect_toggle(self) -> None:
        """比率固定ON時に現在の値から再計算する。"""
        if not self.lock_aspect_var.get() or not self.original_size:
            return
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width and width > 0:
            self._set_height_from_width(width)
        elif height and height > 0:
            self._set_width_from_height(height)

    def _set_initial_target_size(self, image: Image.Image) -> None:
        """画像読込時は元の解像度をそのまま初期値にする。"""
        img_w, img_h = image.size
        self._set_size_fields(img_w, img_h)

    def _halve_size(self) -> None:
        """幅・高さをまとめて半分にする。"""
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width is None or height is None:
            self.status_var.set("幅と高さは整数で入力してください。")
            return
        new_w = max(1, width // 2)
        new_h = max(1, height // 2)
        self._set_size_fields(new_w, new_h)
        if self.lock_aspect_var.get() and self.original_size:
            # 丸めによる比率ズレを抑えるため再計算
            self._set_height_from_width(new_w)

    def _reset_size(self) -> None:
        """幅・高さを元画像の解像度に戻す。"""
        if not self.original_size:
            self.status_var.set("先に入力画像を選択してください。")
            return
        orig_w, orig_h = self.original_size
        self._set_size_fields(orig_w, orig_h)

    def _on_adaptive_weight_change(self) -> None:
        """適応型ブロック用スライダーの値を0〜100で表示用に丸める。"""
        val = float(self.adaptive_weight_var.get())
        clamped = max(0.0, min(100.0, val))
        if clamped != val:
            self.adaptive_weight_var.set(clamped)
        self.adaptive_weight_display.set(f"{clamped:.0f}")
        # 方式が変更された直後にも状態を反映する
        self._update_adaptive_controls()

    def _on_adaptive_pointer(self, event: tk.Event) -> str:
        """スライダーをドラッグした位置から0〜100の値を設定する。"""
        scale: ttk.Scale = event.widget  # type: ignore[assignment]
        width = max(1, scale.winfo_width())
        fraction = max(0.0, min(1.0, event.x / width))
        new_val = 100.0 * fraction
        self.adaptive_weight_var.set(round(new_val, 0))
        self._on_adaptive_weight_change()
        return "break"

    def _update_num_colors_state(self) -> None:
        """減色方式が「なし」の場合は色数指定を無効化する。"""
        is_none = self.quantize_method_var.get() == "なし"
        state_token = "disabled" if is_none else "!disabled"
        try:
            self.num_colors_spin.state([state_token])
        except Exception:
            pass
        # 無効時も内部値は保持しておく（再度有効化した際に復元できるようにする）
        if hasattr(self, "num_colors_spin"):
            try:
                self.num_colors_spin.configure(foreground="#888" if is_none else "#000")
            except Exception:
                pass

    def _update_adaptive_controls(self) -> None:
        """減色方式に応じて適応ブロックスライダーの有効/無効を切り替える。"""
        is_adaptive = self.division_method_var.get() == "適応型ブロック分割"
        state_token = "!disabled" if is_adaptive else "disabled"
        # ttk.Scaleのstateはリスト指定で切り替える
        try:
            self.adaptive_scale.state([state_token])
        except Exception:
            pass
        # ラベル色を変えて無効時を分かりやすくする
        if hasattr(self, "adaptive_label"):
            self.adaptive_label.configure(foreground="#000" if is_adaptive else "#888")

    def _update_pipeline_controls(self) -> None:
        """ブロック分割・適応型ブロック分割時は処理順序を固定して選択不可にする。"""
        is_block_mode = self.division_method_var.get() in {"ブロック分割", "適応型ブロック分割"}
        if is_block_mode:
            # UI表示も固定
            self.pipeline_var.set("リサイズ→減色")
            try:
                self.pipeline_box.state(["disabled"])
            except Exception:
                pass
        else:
            try:
                self.pipeline_box.state(["!disabled"])
            except Exception:
                pass

