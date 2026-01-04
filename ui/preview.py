"""プレビュー描画まわりのシンプルなMixin。"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, TYPE_CHECKING
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from .app import BeadsApp


class PreviewMixin:
    """入力/出力画像のプレビュー更新だけを扱う。"""

    def _can_toggle_input_overlay(self: "BeadsApp") -> bool:
        """入力プレビューで比較表示できるかを判定する。"""
        if getattr(self, "_noise_busy", False):
            return False
        if getattr(self, "_input_using_filtered", False) and self.input_original_pil:
            return True
        if not getattr(self, "_input_using_filtered", False) and getattr(self, "input_filtered_pil", None):
            return True
        return False

    def _on_input_press(self: "BeadsApp", _event: tk.Event) -> None:
        """入力プレビューを長押しでノイズ除去前後を表示切替する。"""
        if getattr(self, "_noise_busy", False):
            return "break"
        if self._can_toggle_input_overlay():
            self._showing_input_overlay = True
            self._refresh_previews()
        return None

    def _on_input_release(self: "BeadsApp", _event: tk.Event) -> None:
        """長押しを離したら通常表示に戻す。"""
        if getattr(self, "_noise_busy", False):
            return "break"
        if self._showing_input_overlay:
            self._showing_input_overlay = False
            self._refresh_previews()
        return None

    def _on_output_press(self: "BeadsApp", _event: tk.Event) -> None:
        """出力プレビューを長押しで一つ前の出力に重ね表示する。"""
        if getattr(self, "_all_mode_results", None):
            return
        if self.prev_output_pil:
            self._showing_prev = True
            self._refresh_previews()

    def _on_output_release(self: "BeadsApp", _event: tk.Event) -> None:
        """長押しを離したら最新出力に戻す。"""
        if getattr(self, "_all_mode_results", None):
            return
        if self._showing_prev:
            self._showing_prev = False
            self._refresh_previews()

    def _on_preview_resize(self: "BeadsApp", _event: tk.Event) -> None:
        self._refresh_previews()

    def _refresh_previews(self: "BeadsApp") -> None:
        show_all = bool(getattr(self, "_all_mode_results", None))
        self._set_input_visible(not show_all)
        self.root.update_idletasks()
        frame_w = self.preview_frame.winfo_width() or self.preview_frame.winfo_reqwidth() or 400
        frame_h = self.preview_frame.winfo_height() or self.preview_frame.winfo_reqheight() or 300
        cell_w = max(1, (frame_w - 20) // 2)
        cell_h = max(1, frame_h - 20)

        input_display: Optional[Image.Image] = None
        if not show_all:
            input_caption = ""
            if self.input_pil:
                input_display = self.input_pil
                if self._input_using_filtered and self._showing_input_overlay and self.input_original_pil:
                    input_display = self.input_original_pil
                    input_caption = "元画像を表示中"
                elif (not self._input_using_filtered) and self._showing_input_overlay and self.input_filtered_pil:
                    input_display = self.input_filtered_pil
                    input_caption = "ノイズ除去後を表示中"

            if input_display:
                photo = self._resize_to_box(input_display, cell_w, cell_h)
                if photo:
                    self._input_photo = photo
                    self.input_canvas.configure(image=self._input_photo, text=input_caption)
            else:
                self.input_canvas.configure(image="", text="入力画像")

        if getattr(self, "_all_mode_results", None):
            self._set_output_grid_visible(True)
            self._refresh_all_mode_grid(frame_w, frame_h)
            return

        self._set_output_grid_visible(False)
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

    def _set_output_grid_visible(self: "BeadsApp", visible: bool) -> None:
        grid_frame = getattr(self, "output_grid_frame", None)
        output_canvas = getattr(self, "output_canvas", None)
        if not grid_frame or not output_canvas:
            return
        if visible:
            output_canvas.grid_remove()
            grid_frame.grid()
        else:
            grid_frame.grid_remove()
            output_canvas.grid()

    def _refresh_all_mode_grid(self: "BeadsApp", col_w: int, col_h: int) -> None:
        results = getattr(self, "_all_mode_results", None) or []
        base_w = 0
        base_h = 0
        for entry in results:
            pil = entry.get("pil")
            if isinstance(pil, Image.Image):
                base_w, base_h = pil.size
                break
        rows, cols = self._get_all_mode_grid_shape(base_w, base_h)
        # 行列数に合わせてグリッドを並べ替える
        grid_frame = getattr(self, "output_grid_frame", None)
        if grid_frame:
            for row in range(4):
                grid_frame.rowconfigure(row, weight=0)
            for col in range(4):
                grid_frame.columnconfigure(col, weight=0)
            for row in range(rows):
                grid_frame.rowconfigure(row, weight=1)
            for col in range(cols):
                grid_frame.columnconfigure(col, weight=1)
            for idx, cell in enumerate(getattr(self, "output_grid_cells", [])):
                frame = cell.get("frame")
                if frame is None:
                    continue
                row = idx // cols
                col = idx % cols
                frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        # 行列数に合わせて表示領域を計算
        out_w = getattr(self, "output_container", None)
        container_w = col_w
        container_h = col_h
        if out_w:
            measured_w = out_w.winfo_width()
            measured_h = out_w.winfo_height()
            if measured_w and measured_w >= col_w * 0.9:
                container_w = measured_w
            if measured_h and measured_h >= col_h * 0.9:
                container_h = measured_h
        cell_w = max(1, (container_w - 12) // cols)
        cell_h = max(1, (container_h - 12) // rows)
        caption_h = 18
        image_h = max(1, cell_h - caption_h)
        self._output_grid_photos = []
        for idx, cell in enumerate(getattr(self, "output_grid_cells", [])):
            image_label = cell.get("image")
            caption_label = cell.get("caption")
            if image_label is None or caption_label is None:
                continue
            if idx < len(results):
                entry = results[idx]
                caption = str(entry.get("label", ""))
                caption_label.configure(text=caption, wraplength=cell_w)
                pil = entry.get("pil")
                if pil is None and "image" in entry:
                    pil = Image.fromarray(entry["image"])
                if pil:
                    photo = self._resize_to_box(pil, cell_w, image_h)
                    if photo:
                        self._output_grid_photos.append(photo)
                        image_label.configure(image=photo, text="")
                else:
                    image_label.configure(image="", text="")
            else:
                caption_label.configure(text="")
                image_label.configure(image="", text="")

    def _resize_to_box(self: "BeadsApp", image: Image.Image, box_w: int, box_h: int) -> Optional[ImageTk.PhotoImage]:
        img_w, img_h = image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = image.resize(new_size, Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(resized)

    def _set_input_visible(self: "BeadsApp", visible: bool) -> None:
        input_canvas = getattr(self, "input_canvas", None)
        output_container = getattr(self, "output_container", None)
        if not input_canvas or not output_container:
            return
        if visible:
            input_canvas.grid()
            output_container.grid_configure(column=1, columnspan=1)
        else:
            input_canvas.grid_remove()
            output_container.grid_configure(column=0, columnspan=2)
