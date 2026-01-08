"""色使用一覧の別ウィンドウ表示を担当する。"""

from __future__ import annotations

from typing import Callable, Optional
from collections import OrderedDict

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
        dim_var: Optional[tk.DoubleVar] = None,
        dim_display_var: Optional[tk.StringVar] = None,
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
        self._dim_var = dim_var
        self._dim_display_var = dim_display_var
        self._preview_zoom = 1.0
        self._preview_canvas: Optional[tk.Canvas] = None
        self._preview_image_item: Optional[int] = None
        self._preview_text_item: Optional[int] = None
        self._preview_scale = 1.0
        self._preview_image_pos = (0.0, 0.0)
        self._preview_region_size = (0, 0)
        self._preview_render_job: Optional[str] = None
        self._preview_zoom_anchor: Optional[tuple[float, float, float, float]] = None
        self._preview_cache: OrderedDict[tuple, ImageTk.PhotoImage] = OrderedDict()
        self._preview_cache_limit = 8
        self._preview_grid_suspended = False
        self._preview_grid_deferred_job: Optional[str] = None
        self._preview_grid_delay_ms = 120
        self._preview_box_size = (0, 0)
        self._preview_zoom_step = 0.01

        window = tk.Toplevel(parent)
        window.title("色使用一覧")
        window.geometry("480x360")
        window.minsize(420, 280)
        window.protocol("WM_DELETE_WINDOW", self._handle_close)
        # OS側の最大化を優先して一覧を見やすくする
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

        # 左に一覧、右にプレビューを配置する
        list_frame = ttk.Frame(container)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(container, text="選択色プレビュー")
        preview_frame.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
        preview_frame.rowconfigure(0, weight=0)
        if self._dim_var is None:
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

        preview_row = 1
        if self._dim_var is not None:
            tone_frame = ttk.Frame(preview_frame)
            tone_frame.grid(row=1, column=0, sticky="we", padx=4, pady=(4, 0))
            tone_frame.columnconfigure(1, weight=1)
            ttk.Label(tone_frame, text="非選択色の明暗").grid(row=0, column=0, padx=(0, 4), sticky="w")
            tone_scale = ttk.Scale(
                tone_frame,
                from_=-1.0,
                to=1.0,
                orient="horizontal",
                variable=self._dim_var,
                length=160,
            )
            tone_scale.grid(row=0, column=1, sticky="we")
            tone_scale.bind("<Button-1>", self._on_tone_pointer)
            tone_scale.bind("<B1-Motion>", self._on_tone_pointer)
            if self._dim_display_var is not None:
                ttk.Label(tone_frame, textvariable=self._dim_display_var, width=10, anchor="e").grid(
                    row=0,
                    column=2,
                    padx=(4, 0),
                    sticky="e",
                )
            preview_row = 2

        preview_canvas = tk.Canvas(preview_frame, highlightthickness=0)
        preview_canvas.grid(row=preview_row, column=0, sticky="nsew", padx=4, pady=4)
        preview_canvas.bind("<Configure>", self._on_preview_resize)
        preview_canvas.bind("<MouseWheel>", self._on_preview_wheel)
        preview_canvas.bind("<Button-4>", self._on_preview_wheel)
        preview_canvas.bind("<Button-5>", self._on_preview_wheel)
        preview_canvas.bind("<ButtonPress-3>", self._on_preview_pan_start)
        preview_canvas.bind("<B3-Motion>", self._on_preview_pan_move)
        preview_canvas.bind("<ButtonRelease-3>", self._on_preview_pan_end)

        self._tree = tree
        self._preview_canvas = preview_canvas
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Button-1>", self._on_tree_click, add="+")
        self.update_rows(rows, reset_sort=True)
        self._bind_keyboard_shortcuts()

    def _bind_keyboard_shortcuts(self) -> None:
        """プレビュー操作用のキーを登録する。"""
        self._window.bind("<KeyPress>", self._on_preview_key, add="+")

    def _on_preview_key(self, event: tk.Event) -> Optional[str]:
        """WASD/QEでプレビューを操作する。"""
        keysym = str(getattr(event, "keysym", "")).lower()
        if keysym in {"w", "a", "s", "d"}:
            self._move_preview_by_key(keysym)
            return "break"
        if keysym in {"q", "e"}:
            self._zoom_preview_by_key(keysym)
            return "break"
        if keysym == "f":
            self._grid_var.set(not self._grid_var.get())
            self._on_grid_toggle()
            return "break"
        return None

    def _move_preview_by_key(self, key: str) -> None:
        """WASDでプレビューを移動する。"""
        if self._preview_base_image is None:
            return
        canvas = self._preview_canvas
        if canvas is None:
            return
        box_w, box_h = self._get_preview_box_size()
        region_w, region_h = self._preview_region_size
        if region_w <= box_w and region_h <= box_h:
            return
        step = max(12, int(min(box_w, box_h) * 0.07))
        dx = 0
        dy = 0
        if key == "w":
            dy = -step
        elif key == "s":
            dy = step
        elif key == "a":
            dx = -step
        elif key == "d":
            dx = step
        self._move_preview_by_pixels(dx, dy)

    def _move_preview_by_pixels(self, dx: int, dy: int) -> None:
        """スクロール領域内で表示位置をずらす。"""
        canvas = self._preview_canvas
        if canvas is None:
            return
        box_w, box_h = self._get_preview_box_size()
        region_w, region_h = self._preview_region_size
        if region_w <= 0 or region_h <= 0:
            return
        max_x0 = max(0.0, region_w - box_w)
        max_y0 = max(0.0, region_h - box_h)
        xview = canvas.xview()
        yview = canvas.yview()
        cur_x0 = xview[0] * region_w
        cur_y0 = yview[0] * region_h
        new_x0 = min(max(cur_x0 + dx, 0.0), max_x0)
        new_y0 = min(max(cur_y0 + dy, 0.0), max_y0)
        if max_x0 > 0:
            canvas.xview_moveto(new_x0 / float(region_w))
        if max_y0 > 0:
            canvas.yview_moveto(new_y0 / float(region_h))

    def _zoom_preview_by_key(self, key: str) -> None:
        """Q/Eで画面中心を基準に拡大縮小する。"""
        if self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        delta = -120 if key == "q" else 120
        self._apply_preview_zoom(delta, box_w / 2, box_h / 2)

    def _on_tone_pointer(self, event: tk.Event) -> str:
        """クリック位置に合わせてスライダー値を動かす。"""
        return self._set_scale_by_pointer(event)

    def _set_scale_by_pointer(self, event: tk.Event) -> str:
        """クリック位置に合わせて値を設定する。"""
        # クリック位置を比率に換算してスライダーの値を更新する
        scale: ttk.Scale = event.widget  # type: ignore[assignment]
        width = max(1, scale.winfo_width())
        fraction = max(0.0, min(1.0, event.x / width))
        min_val = float(scale.cget("from"))
        max_val = float(scale.cget("to"))
        new_val = min_val + (max_val - min_val) * fraction
        if self._dim_var is not None:
            self._dim_var.set(new_val)
        return "break"

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

    def _on_tree_click(self, event: tk.Event) -> Optional[str]:
        """左クリックで選択/解除を切り替える。"""
        region = self._tree.identify_region(event.x, event.y)
        if region == "heading":
            return None
        item_id = self._tree.identify_row(event.y)
        if not item_id:
            return None
        if item_id in self._tree.selection():
            self._tree.selection_remove(item_id)
            self._notify_selection(None)
        else:
            self._tree.selection_set(item_id)
            rgb = self._item_rgb.get(item_id)
            self._notify_selection(rgb)
        return "break"

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
        if image is not self._preview_base_image:
            self._preview_cache.clear()
            self._preview_grid_suspended = False
            if self._preview_grid_deferred_job is not None and self._preview_canvas is not None:
                try:
                    self._preview_canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
                self._preview_grid_deferred_job = None
        self._preview_base_image = image
        self._render_preview()

    def _on_preview_resize(self, _event: tk.Event) -> None:
        """プレビュー枠のサイズ変更に合わせて再描画する。"""
        canvas = self._preview_canvas
        if canvas is not None:
            self._preview_box_size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
        self._render_preview()

    def _on_grid_toggle(self) -> None:
        """グリッド表示の切り替えを反映する。"""
        self._preview_grid_suspended = False
        if self._preview_grid_deferred_job is not None and self._preview_canvas is not None:
            try:
                self._preview_canvas.after_cancel(self._preview_grid_deferred_job)
            except Exception:
                pass
            self._preview_grid_deferred_job = None
        self._render_preview()

    def _on_preview_pan_start(self, event: tk.Event) -> None:
        """右クリックドラッグで移動を開始する。"""
        if self._preview_base_image is None:
            return
        canvas = self._preview_canvas
        if canvas is None:
            return
        canvas.scan_mark(event.x, event.y)

    def _on_preview_pan_move(self, event: tk.Event) -> None:
        """ドラッグ量に応じて表示位置を動かす。"""
        if self._preview_base_image is None:
            return
        canvas = self._preview_canvas
        if canvas is None:
            return
        canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_preview_pan_end(self, _event: tk.Event) -> None:
        """ドラッグ終了時の後始末。"""
        return None

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
        self._apply_preview_zoom(delta, float(event.x), float(event.y))

    def _apply_preview_zoom(self, delta: int, anchor_x: float, anchor_y: float) -> None:
        """指定座標を基準に拡大縮小する。"""
        canvas = self._preview_canvas
        if canvas is None or self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        base_scale = min(box_w / img_w, box_h / img_h)
        base_scale = max(base_scale, 0.01)
        old_zoom = self._preview_zoom
        old_scale = base_scale * max(old_zoom, 1.0)
        old_scaled_w = max(1, int(img_w * old_scale))
        old_scaled_h = max(1, int(img_h * old_scale))
        old_pos_x = 0.0 if old_scaled_w >= box_w else (box_w - old_scaled_w) / 2
        old_pos_y = 0.0 if old_scaled_h >= box_h else (box_h - old_scaled_h) / 2
        # 指定座標を基準に拡大縮小するための基準座標を求める
        cursor_canvas_x = float(canvas.canvasx(anchor_x))
        cursor_canvas_y = float(canvas.canvasy(anchor_y))
        base_x = (cursor_canvas_x - old_pos_x) / old_scale
        base_y = (cursor_canvas_y - old_pos_y) / old_scale
        base_x = min(max(base_x, 0.0), float(img_w))
        base_y = min(max(base_y, 0.0), float(img_h))

        factor = 1.1
        if delta > 0:
            new_zoom = old_zoom * factor
        else:
            new_zoom = old_zoom / factor
        # 等倍を下限としてクランプする
        if new_zoom < 1.0:
            new_zoom = 1.0
        # 倍率を丸めて不要な再計算を減らす
        step = max(self._preview_zoom_step, 0.001)
        new_zoom = round(new_zoom / step) * step
        if new_zoom < 1.0:
            new_zoom = 1.0
        if new_zoom == old_zoom:
            return
        self._preview_zoom = new_zoom
        # 連続操作中は描画を間引いて最後にまとめて更新する
        self._preview_zoom_anchor = (base_x, base_y, float(anchor_x), float(anchor_y))
        if self._grid_var.get():
            # グリッドは後から描き直して負荷を抑える
            self._preview_grid_suspended = True
            if self._preview_grid_deferred_job is not None:
                try:
                    canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
            self._preview_grid_deferred_job = canvas.after(
                self._preview_grid_delay_ms, self._flush_grid_overlay
            )
        if self._preview_render_job is not None:
            try:
                canvas.after_cancel(self._preview_render_job)
            except Exception:
                pass
        self._preview_render_job = canvas.after(16, self._flush_preview_zoom)

    def _flush_preview_zoom(self) -> None:
        """ホイール操作の描画をまとめて実行する。"""
        canvas = self._preview_canvas
        if canvas is None:
            self._preview_render_job = None
            self._preview_zoom_anchor = None
            return
        self._preview_render_job = None
        anchor = self._preview_zoom_anchor
        self._render_preview()
        if self._preview_base_image is None:
            self._preview_zoom_anchor = None
            return
        if not anchor:
            return
        self._preview_zoom_anchor = None
        box_w, box_h = self._get_preview_box_size()
        base_x, base_y, event_x, event_y = anchor
        new_scale = self._preview_scale
        new_pos_x, new_pos_y = self._preview_image_pos
        region_w, region_h = self._preview_region_size
        if region_w > box_w:
            view_x0 = new_pos_x + base_x * new_scale - event_x
            max_x0 = float(region_w - box_w)
            view_x0 = min(max(view_x0, 0.0), max_x0)
            canvas.xview_moveto(view_x0 / float(region_w))
        else:
            canvas.xview_moveto(0)
        if region_h > box_h:
            view_y0 = new_pos_y + base_y * new_scale - event_y
            max_y0 = float(region_h - box_h)
            view_y0 = min(max(view_y0, 0.0), max_y0)
            canvas.yview_moveto(view_y0 / float(region_h))
        else:
            canvas.yview_moveto(0)

    def _flush_grid_overlay(self) -> None:
        """スクロール終了後にグリッドを描き直す。"""
        canvas = self._preview_canvas
        self._preview_grid_deferred_job = None
        if canvas is None:
            self._preview_grid_suspended = False
            return
        if not self._grid_var.get():
            self._preview_grid_suspended = False
            return
        self._preview_grid_suspended = False
        self._render_preview()

    def _get_preview_box_size(self) -> tuple[int, int]:
        """プレビュー枠のサイズをキャッシュから取得する。"""
        box_w, box_h = self._preview_box_size
        if box_w > 0 and box_h > 0:
            return box_w, box_h
        canvas = self._preview_canvas
        if canvas is None:
            return (200, 200)
        box_w = canvas.winfo_width() or canvas.winfo_reqwidth() or 200
        box_h = canvas.winfo_height() or canvas.winfo_reqheight() or 200
        self._preview_box_size = (box_w, box_h)
        return box_w, box_h

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

    def _get_cached_preview_photo(self, key: tuple) -> Optional[ImageTk.PhotoImage]:
        """プレビュー画像のキャッシュを参照する。"""
        photo = self._preview_cache.get(key)
        if photo is not None:
            self._preview_cache.move_to_end(key)
        return photo

    def _store_preview_cache(self, key: tuple, photo: ImageTk.PhotoImage) -> None:
        """プレビュー画像のキャッシュを更新する。"""
        self._preview_cache[key] = photo
        self._preview_cache.move_to_end(key)
        while len(self._preview_cache) > self._preview_cache_limit:
            # 古いキャッシュを捨ててメモリ増加を抑える
            self._preview_cache.popitem(last=False)

    def _render_preview(self) -> None:
        """プレビュー表示を更新する。"""
        canvas = self._preview_canvas
        if canvas is None:
            return
        box_w, box_h = self._get_preview_box_size()
        if self._preview_base_image is None:
            if self._preview_image_item is not None:
                canvas.delete(self._preview_image_item)
                self._preview_image_item = None
            self._preview_photo = None
            if self._preview_text_item is None:
                self._preview_text_item = canvas.create_text(
                    box_w / 2,
                    box_h / 2,
                    text="色を選択してください",
                )
            else:
                canvas.itemconfig(self._preview_text_item, text="色を選択してください")
                canvas.coords(self._preview_text_item, box_w / 2, box_h / 2)
            canvas.configure(scrollregion=(0, 0, box_w, box_h))
            canvas.xview_moveto(0)
            canvas.yview_moveto(0)
            self._preview_scale = 1.0
            self._preview_image_pos = (0.0, 0.0)
            self._preview_region_size = (box_w, box_h)
            return
        if self._preview_text_item is not None:
            canvas.delete(self._preview_text_item)
            self._preview_text_item = None
        img_w, img_h = self._preview_base_image.size
        base_scale = min(box_w / img_w, box_h / img_h)
        base_scale = max(base_scale, 0.01)
        # 等倍（現状のフィット倍率）を下限にして拡大のみ許可する
        scale = base_scale * max(self._preview_zoom, 1.0)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        grid_on = self._grid_var.get() and not self._preview_grid_suspended
        line_color = self._get_grid_line_color() if grid_on else None
        cache_key = (id(self._preview_base_image), img_w, img_h, new_size, grid_on, line_color)
        photo = self._get_cached_preview_photo(cache_key)
        if photo is None:
            resized = self._preview_base_image.resize(new_size, Image.Resampling.NEAREST)
            if grid_on:
                resized = self._apply_grid_overlay(resized, (img_w, img_h))
            photo = ImageTk.PhotoImage(resized)
            self._store_preview_cache(cache_key, photo)
        resized_w, resized_h = new_size
        pos_x = 0.0 if resized_w >= box_w else (box_w - resized_w) / 2
        pos_y = 0.0 if resized_h >= box_h else (box_h - resized_h) / 2
        region_w = max(box_w, resized_w)
        region_h = max(box_h, resized_h)
        self._preview_scale = scale
        self._preview_image_pos = (pos_x, pos_y)
        self._preview_region_size = (region_w, region_h)
        self._preview_photo = photo
        if self._preview_image_item is None:
            self._preview_image_item = canvas.create_image(
                pos_x,
                pos_y,
                anchor="nw",
                image=self._preview_photo,
            )
        else:
            canvas.itemconfig(self._preview_image_item, image=self._preview_photo)
            canvas.coords(self._preview_image_item, pos_x, pos_y)
        canvas.configure(scrollregion=(0, 0, region_w, region_h))
        if resized_w <= box_w:
            canvas.xview_moveto(0)
        if resized_h <= box_h:
            canvas.yview_moveto(0)
