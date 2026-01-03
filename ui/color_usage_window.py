"""色使用一覧の別ウィンドウ表示を担当する。"""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw


class ColorUsageWindow:
    """色使用数の一覧を別ウィンドウで表示する。"""

    def __init__(
        self,
        parent: tk.Tk,
        rows: list[dict],
        on_close: Optional[Callable[[], None]] = None,
        on_select: Optional[Callable[[Optional[tuple[int, int, int]]], None]] = None,
    ) -> None:
        self._parent = parent
        self._on_close = on_close
        self._on_select = on_select
        self._rows: list[dict] = []
        self._sort_key = "count"
        self._sort_desc = True
        self._swatch_images: list[ImageTk.PhotoImage] = []
        self._item_rgb: dict[str, tuple[int, int, int]] = {}
        self._selected_rgb: Optional[tuple[int, int, int]] = None
        self._preview_base_image: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._grid_var = tk.BooleanVar(value=False)
        self._preview_zoom = 1.0
        self._preview_offset = (0, 0)
        self._pan_start: Optional[tuple[int, int]] = None
        self._pan_origin = (0, 0)
        self._preview_scaled_size: Optional[tuple[int, int]] = None
        self._preview_box_size: Optional[tuple[int, int]] = None
        self._preview_display_size: Optional[tuple[int, int]] = None

        window = tk.Toplevel(parent)
        window.title("色使用一覧")
        window.geometry("480x360")
        window.minsize(420, 280)
        window.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._window = window

        container = ttk.Frame(window, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)

        # 左に一覧、右にプレビューを配置する
        list_frame = ttk.Frame(container)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(container, text="選択色プレビュー")
        preview_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        preview_frame.rowconfigure(0, weight=0)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        style = ttk.Style(window)
        style.configure("ColorUsage.Treeview", rowheight=26)

        columns = ("color_id", "name", "count")
        tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="tree headings",
            selectmode="browse",
            style="ColorUsage.Treeview",
        )
        tree.heading("#0", text="色ボックス")
        tree.heading("color_id", text="色番号", command=lambda: self._on_sort("color_id"))
        tree.heading("name", text="色名")
        tree.heading("count", text="色数", command=lambda: self._on_sort("count"))
        tree.column("#0", width=70, anchor="center")
        tree.column("color_id", width=90, anchor="center")
        tree.column("name", width=160, anchor="w")
        tree.column("count", width=80, anchor="e")

        vscroll = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        grid_toggle = ttk.Checkbutton(
            preview_frame,
            text="グリッド表示",
            variable=self._grid_var,
            command=self._on_grid_toggle,
        )
        grid_toggle.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 0))

        preview_label = ttk.Label(preview_frame, text="色を選択してください", anchor="center")
        preview_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        preview_frame.bind("<Configure>", self._on_preview_resize)
        preview_label.bind("<MouseWheel>", self._on_preview_wheel)
        preview_label.bind("<Button-4>", self._on_preview_wheel)
        preview_label.bind("<Button-5>", self._on_preview_wheel)
        preview_label.bind("<ButtonPress-2>", self._on_preview_pan_start)
        preview_label.bind("<B2-Motion>", self._on_preview_pan_move)
        preview_label.bind("<ButtonRelease-2>", self._on_preview_pan_end)

        self._tree = tree
        self._preview_label = preview_label
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.update_rows(rows, reset_sort=True)

    def is_alive(self) -> bool:
        """ウィンドウが生きているかを判定する。"""
        try:
            return bool(self._window.winfo_exists())
        except Exception:
            return False

    def focus(self) -> None:
        """ウィンドウを前面に出す。"""
        try:
            self._window.lift()
            self._window.focus_force()
        except Exception:
            pass

    def update_rows(self, rows: list[dict], reset_sort: bool = True) -> None:
        """一覧データを更新する。"""
        self._rows = list(rows)
        if reset_sort:
            self._sort_key = "count"
            self._sort_desc = True
        self._render_rows()
        self._tree.selection_remove(self._tree.selection())
        self._notify_selection(None)

    def _handle_close(self) -> None:
        """ウィンドウを閉じた後の後始末。"""
        try:
            if self._on_close:
                self._on_close()
        finally:
            try:
                self._window.destroy()
            except Exception:
                pass

    def _on_sort(self, key: str) -> None:
        """列ヘッダー押下で並び替えを切り替える。"""
        if key == self._sort_key:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_key = key
            self._sort_desc = True if key == "count" else False
        self._render_rows()

    def _on_tree_select(self, _event: tk.Event) -> None:
        """色選択に応じて通知する。"""
        selection = self._tree.selection()
        if not selection:
            self._notify_selection(None)
            return
        rgb = self._item_rgb.get(selection[0])
        self._notify_selection(rgb)

    def _notify_selection(self, rgb: Optional[tuple[int, int, int]]) -> None:
        """選択色をコールバックへ渡す。"""
        self._selected_rgb = rgb
        if self._on_select:
            self._on_select(rgb)
        self._render_preview()

    def _render_rows(self) -> None:
        """Treeviewの内容を再描画する。"""
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._swatch_images = []
        self._item_rgb = {}
        for row in self._get_sorted_rows():
            rgb = row.get("rgb", (0, 0, 0))
            swatch = self._make_swatch(rgb)
            self._swatch_images.append(swatch)
            item_id = self._tree.insert(
                "",
                "end",
                image=swatch,
                values=(row.get("color_id", ""), row.get("name", ""), row.get("count", 0)),
            )
            self._item_rgb[item_id] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _get_sorted_rows(self) -> list[dict]:
        """現在のソート条件に沿って並び替える。"""
        if self._sort_key == "color_id":
            return sorted(
                self._rows,
                key=lambda r: self._parse_color_id(str(r.get("color_id", ""))),
                reverse=self._sort_desc,
            )
        if self._sort_key == "count":
            return sorted(
                self._rows,
                key=lambda r: int(r.get("count", 0)),
                reverse=self._sort_desc,
            )
        return list(self._rows)

    def _parse_color_id(self, text: str) -> int:
        """色番号の数字部分だけを取り出す。"""
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return 0
        try:
            return int(digits)
        except ValueError:
            return 0

    def _make_swatch(self, rgb: tuple[int, int, int]) -> ImageTk.PhotoImage:
        """色見本用の小さな四角画像を作る。"""
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        img = Image.new("RGB", (22, 22), (r, g, b))
        return ImageTk.PhotoImage(img)

    def set_preview_image(self, image: Optional[Image.Image]) -> None:
        """プレビュー用画像を更新する。"""
        self._preview_base_image = image
        self._render_preview()

    def _on_preview_resize(self, _event: tk.Event) -> None:
        """プレビュー枠のサイズ変更に合わせて再描画する。"""
        self._render_preview()

    def _on_grid_toggle(self) -> None:
        """グリッド表示の切り替えを反映する。"""
        self._render_preview()

    def _on_preview_pan_start(self, event: tk.Event) -> None:
        """ホイールクリックでドラッグ移動を開始する。"""
        if self._preview_base_image is None:
            return
        scaled = self._preview_scaled_size
        display = self._preview_display_size
        if not scaled or not display:
            return
        if scaled[0] <= display[0] and scaled[1] <= display[1]:
            return
        self._pan_start = (int(event.x), int(event.y))
        self._pan_origin = self._preview_offset

    def _on_preview_pan_move(self, event: tk.Event) -> None:
        """ドラッグ量に応じて表示位置を動かす。"""
        if self._preview_base_image is None or self._pan_start is None:
            return
        scaled = self._preview_scaled_size
        display = self._preview_display_size
        if not scaled or not display:
            return
        dx = int(event.x) - self._pan_start[0]
        dy = int(event.y) - self._pan_start[1]
        new_x = self._pan_origin[0] - dx
        new_y = self._pan_origin[1] - dy
        self._preview_offset = self._clamp_preview_offset((new_x, new_y), scaled, display)
        self._render_preview()

    def _on_preview_pan_end(self, _event: tk.Event) -> None:
        """ドラッグ終了時の後始末。"""
        self._pan_start = None

    def _on_preview_wheel(self, event: tk.Event) -> None:
        """プレビューの拡大縮小（等倍未満は許可しない）。"""
        if self._preview_base_image is None:
            return
        delta = 0
        if getattr(event, "delta", 0):
            delta = int(event.delta)
        elif getattr(event, "num", None) == 4:
            delta = 120
        elif getattr(event, "num", None) == 5:
            delta = -120
        if delta == 0:
            return
        factor = 1.1
        if delta > 0:
            self._preview_zoom *= factor
        else:
            self._preview_zoom /= factor
        # 等倍を下限としてクランプする
        if self._preview_zoom < 1.0:
            self._preview_zoom = 1.0
        self._render_preview()

    def _clamp_preview_offset(
        self,
        offset: tuple[int, int],
        scaled_size: tuple[int, int],
        display_size: tuple[int, int],
    ) -> tuple[int, int]:
        """表示範囲のはみ出しを防ぐ。"""
        max_x = max(0, int(scaled_size[0] - display_size[0]))
        max_y = max(0, int(scaled_size[1] - display_size[1]))
        x = min(max(int(offset[0]), 0), max_x)
        y = min(max(int(offset[1]), 0), max_y)
        return (x, y)

    def _get_grid_line_color(self) -> tuple[int, int, int, int]:
        """選択色の明るさからグリッド線の色を決める。"""
        if self._selected_rgb is None:
            return (0, 0, 0, 90)
        r, g, b = (int(self._selected_rgb[0]), int(self._selected_rgb[1]), int(self._selected_rgb[2]))
        # 選択色が暗いほど線を明るく、明るいほど線を暗くする
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        tone = int(round(255 - luminance))
        tone = max(0, min(255, tone))
        return (tone, tone, tone, 120)

    def _apply_grid_overlay(self, image: Image.Image, source_size: tuple[int, int]) -> Image.Image:
        """プレビュー画像にグリッドを重ねる。"""
        src_w, src_h = source_size
        if src_w <= 1 or src_h <= 1:
            return image
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # 半透明ラインで色を邪魔しすぎないようにする
        line_color = self._get_grid_line_color()
        dst_w, dst_h = base.size
        step_x = dst_w / src_w
        step_y = dst_h / src_h
        last_x = None
        for i in range(1, src_w):
            x = int(round(i * step_x))
            if x <= 0 or x >= dst_w or x == last_x:
                continue
            draw.line([(x, 0), (x, dst_h)], fill=line_color)
            last_x = x
        last_y = None
        for i in range(1, src_h):
            y = int(round(i * step_y))
            if y <= 0 or y >= dst_h or y == last_y:
                continue
            draw.line([(0, y), (dst_w, y)], fill=line_color)
            last_y = y
        return Image.alpha_composite(base, overlay)

    def _render_preview(self) -> None:
        """プレビュー表示を更新する。"""
        if self._preview_base_image is None:
            self._preview_photo = None
            self._preview_scaled_size = None
            self._preview_box_size = None
            self._preview_display_size = None
            self._preview_offset = (0, 0)
            self._preview_label.configure(text="色を選択してください", image="")
            return
        label = self._preview_label
        label.update_idletasks()
        box_w = label.winfo_width() or label.winfo_reqwidth() or 200
        box_h = label.winfo_height() or label.winfo_reqheight() or 200
        img_w, img_h = self._preview_base_image.size
        base_scale = min(box_w / img_w, box_h / img_h)
        base_scale = max(base_scale, 0.01)
        # 等倍（現状のフィット倍率）を下限にして拡大のみ許可する
        scale = base_scale * max(self._preview_zoom, 1.0)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = self._preview_base_image.resize(new_size, Image.Resampling.NEAREST)
        if self._grid_var.get():
            resized = self._apply_grid_overlay(resized, (img_w, img_h))
        resized_w, resized_h = resized.size
        display_w = min(resized_w, box_w)
        display_h = min(resized_h, box_h)
        self._preview_scaled_size = (resized_w, resized_h)
        self._preview_box_size = (box_w, box_h)
        self._preview_display_size = (display_w, display_h)
        self._preview_offset = self._clamp_preview_offset(
            self._preview_offset,
            (resized_w, resized_h),
            (display_w, display_h),
        )
        # 拡大時は切り出しでパン表示する
        if resized_w > display_w or resized_h > display_h:
            off_x, off_y = self._preview_offset
            display_img = resized.crop((off_x, off_y, off_x + display_w, off_y + display_h))
        else:
            display_img = resized
        self._preview_photo = ImageTk.PhotoImage(display_img)
        self._preview_label.configure(image=self._preview_photo, text="")
