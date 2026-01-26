"""List panel controller for color usage window."""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ColorUsageListController:
    def __init__(
        self,
        tree: ttk.Treeview,
        on_select: Callable[[Optional[tuple[int, int, int]]], None],
    ) -> None:
        self._tree = tree
        self._on_select = on_select
        self._rows: list[dict] = []
        self._sort_key = "count"
        self._sort_desc = True
        self._swatch_images: list[ImageTk.PhotoImage] = []
        self._item_rgb: dict[str, tuple[int, int, int]] = {}
        self._rgb_item: dict[tuple[int, int, int], str] = {}
        self._syncing_tree_selection = False
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Button-1>", self._on_tree_click, add="+")

    def on_sort(self, key: str) -> None:
        if key == self._sort_key:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_key = key
            self._sort_desc = True if key == "count" else False
        self._render_rows()

    def update_rows(self, rows: list[dict], reset_sort: bool = True) -> None:
        self._rows = list(rows)
        if reset_sort:
            self._sort_key = "count"
            self._sort_desc = True
        self._render_rows()
        self._tree.selection_remove(self._tree.selection())
        self._on_select(None)

    def set_selected_rgb(self, rgb: Optional[tuple[int, int, int]]) -> None:
        self._sync_tree_selection(rgb)

    def _on_tree_select(self, _event: tk.Event) -> None:
        if self._syncing_tree_selection:
            return
        selection = self._tree.selection()
        if not selection:
            self._on_select(None)
            return
        rgb = self._item_rgb.get(selection[0])
        self._on_select(rgb)

    def _on_tree_click(self, event: tk.Event) -> Optional[str]:
        region = self._tree.identify_region(event.x, event.y)
        if region == "heading":
            return None
        item_id = self._tree.identify_row(event.y)
        if not item_id:
            return None
        if item_id in self._tree.selection():
            self._tree.selection_remove(item_id)
            self._on_select(None)
        else:
            self._tree.selection_set(item_id)
            rgb = self._item_rgb.get(item_id)
            self._on_select(rgb)
        return "break"

    def _render_rows(self) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._swatch_images = []
        self._item_rgb = {}
        self._rgb_item = {}
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
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self._item_rgb[item_id] = rgb_tuple
            self._rgb_item[rgb_tuple] = item_id

    def _sync_tree_selection(self, rgb: Optional[tuple[int, int, int]]) -> None:
        if self._syncing_tree_selection:
            return
        if rgb is None:
            selection = self._tree.selection()
            if selection:
                self._syncing_tree_selection = True
                try:
                    self._tree.selection_remove(selection)
                finally:
                    self._syncing_tree_selection = False
            return
        item_id = self._rgb_item.get(rgb)
        if not item_id:
            selection = self._tree.selection()
            if selection:
                self._syncing_tree_selection = True
                try:
                    self._tree.selection_remove(selection)
                finally:
                    self._syncing_tree_selection = False
            return
        selection = self._tree.selection()
        if selection and selection[0] == item_id:
            return
        self._syncing_tree_selection = True
        try:
            self._tree.selection_set(item_id)
            self._tree.see(item_id)
        finally:
            self._syncing_tree_selection = False

    def _get_sorted_rows(self) -> list[dict]:
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
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return 0
        try:
            return int(digits)
        except ValueError:
            return 0

    def _make_swatch(self, rgb: tuple[int, int, int]) -> ImageTk.PhotoImage:
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        img = Image.new("RGB", (22, 22), (r, g, b))
        return ImageTk.PhotoImage(img)
