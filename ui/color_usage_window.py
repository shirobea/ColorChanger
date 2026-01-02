"""色使用一覧の別ウィンドウ表示を担当する。"""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


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
        self._preview_base_image: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

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
        preview_frame.rowconfigure(0, weight=1)
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

        preview_label = ttk.Label(preview_frame, text="色を選択してください", anchor="center")
        preview_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        preview_frame.bind("<Configure>", self._on_preview_resize)

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
        if self._on_select:
            self._on_select(rgb)

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

    def _render_preview(self) -> None:
        """プレビュー表示を更新する。"""
        if self._preview_base_image is None:
            self._preview_photo = None
            self._preview_label.configure(text="色を選択してください", image="")
            return
        label = self._preview_label
        label.update_idletasks()
        box_w = label.winfo_width() or label.winfo_reqwidth() or 200
        box_h = label.winfo_height() or label.winfo_reqheight() or 200
        img_w, img_h = self._preview_base_image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        resized = self._preview_base_image.resize(new_size, Image.Resampling.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(resized)
        self._preview_label.configure(image=self._preview_photo, text="")
