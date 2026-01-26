"""色使用一覧の別ウィンドウ表示を担当する。"""

from __future__ import annotations

from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

from .color_usage_list import ColorUsageListController
from .color_usage_preview import ColorUsagePreviewController


class ColorUsageWindow:
    """色使用数の一覧を別ウィンドウで表示する。"""

    def __init__(
        self,
        parent: tk.Tk,
        rows: list[dict],
        on_close: Optional[Callable[[], None]] = None,
        on_select: Optional[Callable[[Optional[tuple[int, int, int]]], None]] = None,
        dim_var: Optional[tk.DoubleVar] = None,
        dim_display_var: Optional[tk.StringVar] = None,
    ) -> None:
        self._parent = parent
        self._on_close = on_close
        self._on_select = on_select
        self._selected_rgb: Optional[tuple[int, int, int]] = None

        window = tk.Toplevel(parent)
        window.title("色使用一覧")
        window.geometry("480x360")
        window.minsize(420, 280)
        window.protocol("WM_DELETE_WINDOW", self._handle_close)
        try:
            window.state("zoomed")
        except tk.TclError:
            try:
                window.attributes("-zoomed", True)
            except tk.TclError:
                pass
        self._window = window

        container = ttk.Frame(window, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)

        list_frame = ttk.Frame(container)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(container, text="選択色プレビュー")
        preview_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        preview_frame.rowconfigure(0, weight=0)
        if dim_var is None:
            preview_frame.rowconfigure(1, weight=1)
        else:
            preview_frame.rowconfigure(1, weight=0)
            preview_frame.rowconfigure(2, weight=1)
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
        tree.heading("color_id", text="色番号")
        tree.heading("name", text="色名")
        tree.heading("count", text="色数")
        tree.column("#0", width=70, anchor="center")
        tree.column("color_id", width=90, anchor="center")
        tree.column("name", width=160, anchor="w")
        tree.column("count", width=80, anchor="e")

        vscroll = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        self._grid_var = tk.BooleanVar(value=False)
        grid_toggle = ttk.Checkbutton(
            preview_frame,
            text="グリッド表示",
            variable=self._grid_var,
        )
        grid_toggle.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 0))

        preview_row = 1
        tone_scale = None
        if dim_var is not None:
            tone_frame = ttk.Frame(preview_frame)
            tone_frame.grid(row=1, column=0, sticky="we", padx=4, pady=(4, 0))
            tone_frame.columnconfigure(1, weight=1)
            ttk.Label(tone_frame, text="非選択色の明暗").grid(row=0, column=0, padx=(0, 4), sticky="w")
            tone_scale = ttk.Scale(
                tone_frame,
                from_=-1.0,
                to=1.0,
                orient="horizontal",
                variable=dim_var,
                length=160,
            )
            tone_scale.grid(row=0, column=1, sticky="we")
            if dim_display_var is not None:
                ttk.Label(tone_frame, textvariable=dim_display_var, width=10, anchor="e").grid(
                    row=0,
                    column=2,
                    padx=(4, 0),
                    sticky="e",
                )
            preview_row = 2

        preview_canvas = tk.Canvas(preview_frame, highlightthickness=0)
        preview_canvas.grid(row=preview_row, column=0, sticky="nsew", padx=4, pady=4)

        self._list_controller = ColorUsageListController(tree, self._notify_selection)
        tree.heading("color_id", command=lambda: self._list_controller.on_sort("color_id"))
        tree.heading("count", command=lambda: self._list_controller.on_sort("count"))

        self._preview_controller = ColorUsagePreviewController(
            preview_canvas,
            grid_var=self._grid_var,
            on_select=self._notify_selection,
            dim_var=dim_var,
            empty_message="色を選択してください",
        )
        self._preview_controller.bind_shortcuts(window)
        grid_toggle.configure(command=self._preview_controller.on_grid_toggle)
        if tone_scale is not None:
            tone_scale.bind("<Button-1>", self._preview_controller.on_tone_pointer)
            tone_scale.bind("<B1-Motion>", self._preview_controller.on_tone_pointer)

        self.update_rows(rows, reset_sort=True)

    def is_alive(self) -> bool:
        try:
            return bool(self._window.winfo_exists())
        except Exception:
            return False

    def focus(self) -> None:
        try:
            self._window.lift()
            self._window.focus_force()
        except Exception:
            pass

    def update_rows(self, rows: list[dict], reset_sort: bool = True) -> None:
        self._list_controller.update_rows(rows, reset_sort=reset_sort)

    def set_preview_image(self, image, source_image=None) -> None:
        self._preview_controller.set_preview_image(image, source_image=source_image)

    def _handle_close(self) -> None:
        try:
            if self._on_close:
                self._on_close()
        finally:
            try:
                self._window.destroy()
            except Exception:
                pass

    def _notify_selection(self, rgb: Optional[tuple[int, int, int]]) -> None:
        self._selected_rgb = rgb
        self._list_controller.set_selected_rgb(rgb)
        self._preview_controller.set_selected_rgb(rgb)
        if self._on_select:
            self._on_select(rgb)
