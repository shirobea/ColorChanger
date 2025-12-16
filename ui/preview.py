"""プレビュー描画まわりのシンプルなMixin。"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, TYPE_CHECKING
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from .app import BeadsApp


class PreviewMixin:
    """入力/出力画像のプレビュー更新だけを扱う。"""

    def _on_output_press(self: "BeadsApp", _event: tk.Event) -> None:
        """出力プレビューを長押しで一つ前の出力に重ね表示する。"""
        if self.prev_output_pil:
            self._showing_prev = True
            self._refresh_previews()

    def _on_output_release(self: "BeadsApp", _event: tk.Event) -> None:
        """長押しを離したら最新出力に戻す。"""
        if self._showing_prev:
            self._showing_prev = False
            self._refresh_previews()

    def _on_preview_resize(self: "BeadsApp", _event: tk.Event) -> None:
        self._refresh_previews()

    def _refresh_previews(self: "BeadsApp") -> None:
        self.root.update_idletasks()
        frame_w = self.preview_frame.winfo_width() or self.preview_frame.winfo_reqwidth() or 400
        frame_h = self.preview_frame.winfo_height() or self.preview_frame.winfo_reqheight() or 300
        cell_w = max(1, (frame_w - 20) // 2)
        cell_h = max(1, frame_h - 20)

        if self.input_pil:
            photo = self._resize_to_box(self.input_pil, cell_w, cell_h)
            if photo:
                self._input_photo = photo
                self.input_canvas.configure(image=self._input_photo, text="")
        else:
            self.input_canvas.configure(image="", text="入力画像")

        display_pil: Optional[Image.Image] = None
        if self._showing_prev and self.prev_output_pil and self.output_pil:
            # サイズが異なる場合は現在出力に合わせて前回出力をリサイズし、100%で表示する
            prev = self.prev_output_pil
            base = self.output_pil
            if prev.size != base.size:
                prev = prev.resize(base.size, Image.Resampling.NEAREST)
            display_pil = prev
        elif self._showing_prev and self.prev_output_pil:
            display_pil = self.prev_output_pil
        else:
            display_pil = self.output_pil

        if display_pil:
            photo = self._resize_to_box(display_pil, cell_w, cell_h)
            if photo:
                self._output_photo = photo
                caption = "前回出力との比較表示" if self._showing_prev and self.prev_output_pil else ""
                self.output_canvas.configure(image=self._output_photo, text=caption)
        else:
            self.output_canvas.configure(image="", text="変換後")

    def _resize_to_box(self: "BeadsApp", image: Image.Image, box_w: int, box_h: int) -> Optional[ImageTk.PhotoImage]:
        img_w, img_h = image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = image.resize(new_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(resized)
