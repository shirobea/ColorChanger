"""プレビュー画像の生成・切替・オーバーレイ表示をまとめたMixin。"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, TYPE_CHECKING
import numpy as np
import cv2
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from .app import BeadsApp


class PreviewMixin:
    """プレビュー描画まわりのロジックを集約。"""

    def toggle_saliency_view(self: "BeadsApp") -> None:
        """入力プレビューを重要度オーバーレイと通常画像で切り替える。"""
        if self.saliency_overlay_pil is None:
            if self.input_pil is None:
                self.status_var.set("先に入力画像を選択してください。")
            else:
                self.status_var.set("重要度マップの準備ができていません。")
            return
        self._show_saliency = not self._show_saliency
        self._update_saliency_button_label()
        self._refresh_previews()

    def _set_saliency_button_state(self: "BeadsApp", enabled: bool) -> None:
        """サリエンシートグルボタンの状態とラベルをまとめて切り替える。"""
        if hasattr(self, "saliency_toggle_button"):
            state = "normal" if enabled else "disabled"
            self.saliency_toggle_button.configure(state=state)
        self._update_saliency_button_label()

    def _update_saliency_button_label(self: "BeadsApp") -> None:
        """現在の表示状態に応じてボタンラベルを更新する。"""
        if hasattr(self, "saliency_toggle_button"):
            label = "画像切り替え（重要度表示）" if self._show_saliency and self.saliency_overlay_pil else "画像切り替え（通常）"
            self.saliency_toggle_button.configure(text=label)

    def _build_saliency_overlay(
        self: "BeadsApp", base_image: Image.Image, saliency_map: np.ndarray, alpha: float = 0.55
    ) -> Image.Image:
        """重要度マップにカラーマップを適用して元画像へ重ねる。"""
        sal = np.clip(saliency_map.astype(np.float32), 0.0, 1.0)
        base_rgb = np.array(base_image.convert("RGB"), dtype=np.uint8)
        h, w = base_rgb.shape[:2]
        if sal.shape != (h, w):
            sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        sal8 = np.clip(sal * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(sal8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(heatmap_rgb, float(alpha), base_rgb, float(1.0 - alpha), 0)
        return Image.fromarray(blended)

    def _on_preview_resize(self: "BeadsApp", _event: tk.Event) -> None:
        """プレビューエリアのサイズ変更で再描画する。"""
        self._refresh_previews()

    def _refresh_previews(self: "BeadsApp") -> None:
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

    def _resize_to_box(self: "BeadsApp", image: Image.Image, box_w: int, box_h: int) -> Optional[ImageTk.PhotoImage]:
        """Resize image to fit inside given box using nearest neighbor (no blur)."""
        img_w, img_h = image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = image.resize(new_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(resized)

