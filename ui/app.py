"""Tkinter UI for beads palette conversion."""

import threading
import time
import json
from pathlib import Path
from typing import Optional, Callable
from math import ceil

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
from .importance_editor import ImportanceEditor


class BeadsApp(LayoutMixin, ActionsMixin, StateMixin, PreviewMixin):
    """Main application window."""
    PIXEL_SIZE_MM = 2.6
    PLATE_PIXELS = 28

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
        self.physical_size_var = tk.StringVar(value="完成サイズ: 幅・高さを入力してください")
        self.plate_requirement_var = tk.StringVar(value="28×28プレート: 幅・高さを入力してください")
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
        # リサイズ方式の初期値（指定なし時は最近傍）
        self.resize_method_var = tk.StringVar(value="ニアレストネイバー")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.edge_enhance_var = tk.BooleanVar(value=False)
        self.edge_strength_var = tk.DoubleVar(value=60.0)
        self.edge_strength_display = tk.StringVar(value="60")
        self.edge_thickness_var = tk.DoubleVar(value=30.0)
        self.edge_thickness_display = tk.StringVar(value="30")
        self.edge_gain_var = tk.DoubleVar(value=2.5)
        self.edge_gain_display = tk.StringVar(value="2.5")
        self.edge_gamma_var = tk.DoubleVar(value=0.75)
        self.edge_gamma_display = tk.StringVar(value="0.75")
        self.edge_saliency_weight_var = tk.DoubleVar(value=50.0)  # 0-100%
        self.edge_saliency_display = tk.StringVar(value="50")
        self.adaptive_weight_var = tk.DoubleVar(value=50.0)
        self.adaptive_weight_display = tk.StringVar(value="50")
        self.hybrid_scale_var = tk.DoubleVar(value=100.0)
        self.hybrid_scale_display = tk.StringVar(value="100")
        self.quantize_method_var = tk.StringVar(value="Wu減色")
        # CMC(l:c) 用の係数
        self.cmc_l_var = tk.DoubleVar(value=2.0)
        self.cmc_c_var = tk.DoubleVar(value=1.0)
        self.cmc_l_display = tk.StringVar(value="2.0")
        self.cmc_c_display = tk.StringVar(value="1.0")
        # RGBモード用の重み
        self.rgb_r_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_g_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_b_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_r_display = tk.StringVar(value="1.0")
        self.rgb_g_display = tk.StringVar(value="1.0")
        self.rgb_b_display = tk.StringVar(value="1.0")
        self._updating_size_fields = False
        # 重要度編集用のUI状態
        self.brush_mode_var = tk.StringVar(value="add")
        self.brush_radius_var = tk.IntVar(value=18)
        self.brush_strength_var = tk.DoubleVar(value=25.0)  # 0-100を百分率で保持
        self.brush_radius_display = tk.StringVar(value="18")
        self.brush_strength_display = tk.StringVar(value="25")
        self._is_drawing = False
        self._stroke_points: list[tuple[float, float]] = []
        self._input_display_size: Optional[tuple[int, int]] = None
        self._base_importance_map: Optional[np.ndarray] = None
        self.importance_editor = ImportanceEditor()

        # 前回設定の復元
        self._load_settings()
        self._build_layout()
        self._apply_saved_settings()
        self._update_edge_controls()
        self._update_importance_controls_state(False)


    
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
        # 編集用の初期状態をセットしておく
        self._base_importance_map = importance_full.copy()
        self.importance_editor.load_map(importance_full)
        # 変換に渡すサリエンシーも編集対象と同期させる
        self.importance_map = self.importance_editor.current_map
        self.saliency_map = self.importance_map.copy() if self.importance_map is not None else saliency_full
        self._set_saliency_button_state(enabled=True)
        self._update_importance_controls_state(self._show_saliency)
        self.status_var.set("重要度マップを生成しました。")

    # --- 重要度編集まわり ---
    def _update_importance_controls_state(self, enabled: bool) -> None:
        """重要度編集ツールの有効/無効をまとめて切り替える。"""
        state_token = "!disabled" if enabled else "disabled"
        for widget_name in [
            "pen_radio",
            "eraser_radio",
            "brush_radius_scale",
            "brush_strength_scale",
            "undo_button",
            "redo_button",
            "reset_imp_button",
            "fill_hot_button",
            "fill_cold_button",
        ]:
            widget = getattr(self, widget_name, None)
            if widget:
                try:
                    widget.state([state_token])
                except Exception:
                    try:
                        widget.configure(state="normal" if enabled else "disabled")
                    except Exception:
                        pass

    def _on_brush_radius_change(self) -> None:
        """スライダー操作時に表示値を更新しつつ整数に丸める。"""
        val = int(round(float(self.brush_radius_var.get())))
        val = max(3, min(64, val))
        self.brush_radius_var.set(val)
        self.brush_radius_display.set(str(val))

    def _on_brush_strength_change(self) -> None:
        """強さスライダーの値(0-100)を表示用に反映。"""
        val = float(self.brush_strength_var.get())
        clamped = max(5.0, min(100.0, val))
        self.brush_strength_var.set(clamped)
        self.brush_strength_display.set(f"{clamped:.0f}")

    def _event_to_image_xy(self, event: tk.Event) -> Optional[tuple[float, float]]:
        """プレビュー座標を元画像の座標へ変換する（中央寄せ補正込み）。"""
        if not self.input_pil or not self._input_display_size:
            return None
        disp_w, disp_h = self._input_display_size
        if disp_w <= 0 or disp_h <= 0:
            return None
        widget_w = event.widget.winfo_width()
        widget_h = event.widget.winfo_height()
        # ラベル中央に画像が配置されるため余白を差し引く
        offset_x = max(0, (widget_w - disp_w) // 2)
        offset_y = max(0, (widget_h - disp_h) // 2)
        local_x = event.x - offset_x
        local_y = event.y - offset_y
        if local_x < 0 or local_y < 0 or local_x >= disp_w or local_y >= disp_h:
            return None  # 画像外は無視
        img_w, img_h = self.input_pil.size
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h
        x = float(local_x) * scale_x
        y = float(local_y) * scale_y
        return (x, y)

    def _can_edit_importance(self) -> bool:
        """重要度編集が可能かを判定する。"""
        return bool(
            self._show_saliency
            and self.importance_map is not None
            and self.saliency_overlay_pil is not None
            and self.input_pil is not None
            and self._input_display_size
        )

    def _on_input_press(self, event: tk.Event) -> None:
        """入力プレビュー上で筆を置いたときの処理。"""
        if not self._can_edit_importance():
            self.status_var.set("重要度表示をONにしてから編集してください。")
            return
        pos = self._event_to_image_xy(event)
        if pos is None:
            return
        if not self.importance_editor.begin_stroke():
            self.status_var.set("重要度マップが準備できていません。")
            return
        self._is_drawing = True
        self._stroke_points = [pos]

    def _on_input_drag(self, event: tk.Event) -> None:
        """ドラッグ中は座標を貯めておき、離したタイミングでまとめて反映する。"""
        if not self._is_drawing or not self._can_edit_importance():
            return
        current = self._event_to_image_xy(event)
        if current is None:
            return
        # 前回点との線分を即時反映（見た目の追従用）
        if self._stroke_points:
            seg_points = [self._stroke_points[-1], current]
            changed = self.importance_editor.paint_live(
                points=seg_points,
                radius=int(self.brush_radius_var.get()),
                strength=float(self.brush_strength_var.get()) / 100.0,
                mode=self.brush_mode_var.get(),
            )
            if changed:
                self._on_importance_map_changed()
        self._stroke_points.append(current)

    def _on_input_release(self, _event: tk.Event) -> None:
        """ドラッグ終了で状態をリセットする。"""
        if self._is_drawing and self._stroke_points:
            if self.importance_editor.commit_stroke():
                self._on_importance_map_changed()
        self._is_drawing = False
        self._stroke_points = []

    def _on_importance_map_changed(self) -> None:
        """マップ更新時の共通後処理（オーバーレイ再描画など）。"""
        self.importance_map = self.importance_editor.current_map
        if self.importance_map is not None and self.input_pil is not None:
            self.saliency_map = self.importance_map.copy()
            self.saliency_overlay_pil = self._build_saliency_overlay(self.input_pil, self.importance_map)
            self._refresh_previews()
            self.status_var.set("重要度マップを更新しました。")

    def _undo_importance(self) -> None:
        """直前のストロークを取り消す。"""
        if self.importance_editor.undo():
            self._on_importance_map_changed()
        else:
            self.status_var.set("取り消せる操作がありません。")

    def _redo_importance(self) -> None:
        """Undoした操作をやり直す。"""
        if self.importance_editor.redo():
            self._on_importance_map_changed()
        else:
            self.status_var.set("やり直せる操作がありません。")

    def _reset_importance_edits(self) -> None:
        """計算直後の重要度マップへ戻す。"""
        if self._base_importance_map is None:
            self.status_var.set("リセットできる重要度マップがありません。")
            return
        self.importance_editor.reset_to(self._base_importance_map)
        self._on_importance_map_changed()
        self.status_var.set("重要度マップを初期状態へ戻しました。")

    def _fill_importance_all(self, value: float, success_msg: str) -> None:
        """重要度マップ全体を一括で塗り替える。"""
        if not self._can_edit_importance():
            self.status_var.set("重要度表示をONにしてから編集してください。")
            return
        if self.importance_editor.current_map is None:
            self.status_var.set("重要度マップが準備できていません。")
            return
        if self.importance_editor.fill_all(value):
            self._on_importance_map_changed()
            self.status_var.set(success_msg)
        else:
            self.status_var.set("既に同じ状態のため変更はありません。")

    def _fill_all_hot(self) -> None:
        """全ピクセルを重要（1.0）にする。"""
        self._fill_importance_all(1.0, "重要度マップを全て重要(赤)に設定しました。")

    def _fill_all_cold(self) -> None:
        """全ピクセルを非重要（0.0）にする。"""
        self._fill_importance_all(0.0, "重要度マップを全て非重要(青)に設定しました。")


    
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
                # 編集用に現在表示サイズを覚えておく
                self._input_display_size = (self._input_photo.width(), self._input_photo.height())
                caption = "サリエンシー重ね表示" if self._show_saliency and self.saliency_overlay_pil else ""
                self.input_canvas.configure(image=self._input_photo, text=caption)
            else:
                self._input_display_size = None
        else:
            self._input_display_size = None
            self.input_canvas.configure(image="", text="入力画像")
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
        self._update_physical_size_display()

    def _update_physical_size_display(self) -> None:
        """入力ピクセル数から完成サイズとプレート枚数を計算して表示する。"""
        if self._updating_size_fields:
            return
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width is None or height is None:
            self.physical_size_var.set("完成サイズ: 幅と高さを整数で入力してください")
            self.plate_requirement_var.set("28×28プレート: 幅と高さを整数で入力してください")
            return
        width_mm = width * self.PIXEL_SIZE_MM
        height_mm = height * self.PIXEL_SIZE_MM
        self.physical_size_var.set(f"完成サイズ: {width_mm:.1f}mm × {height_mm:.1f}mm")
        plates_x = ceil(width / self.PLATE_PIXELS)
        plates_y = ceil(height / self.PLATE_PIXELS)
        total = plates_x * plates_y
        self.plate_requirement_var.set(
            f"28×28プレート: 横{plates_x}枚 × 縦{plates_y}枚（計{total}枚）"
        )

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

    def _on_edge_toggle(self) -> None:
        """輪郭強調ON/OFFに合わせてUIを更新。"""
        self._update_edge_controls()

    def _on_edge_strength_change(self) -> None:
        """輪郭強調の強さスライダーの表示値を更新する。"""
        val = float(self.edge_strength_var.get())
        clamped = max(0.0, min(100.0, val))
        if clamped != val:
            self.edge_strength_var.set(clamped)
        self.edge_strength_display.set(f"{clamped:.0f}")

    def _on_edge_thickness_change(self) -> None:
        """輪郭の太さスライダーの表示値を更新する。"""
        val = float(self.edge_thickness_var.get())
        clamped = max(0.0, min(100.0, val))
        if clamped != val:
            self.edge_thickness_var.set(clamped)
        self.edge_thickness_display.set(f"{clamped:.0f}")

    def _on_edge_gain_change(self) -> None:
        """輪郭強調ゲインの表示値を更新する。"""
        val = float(self.edge_gain_var.get())
        clamped = max(0.0, min(5.0, val))
        if clamped != val:
            self.edge_gain_var.set(clamped)
        self.edge_gain_display.set(f"{clamped:.2f}")

    def _on_edge_gamma_change(self) -> None:
        """輪郭重みのガンマ補正値を更新する。"""
        val = float(self.edge_gamma_var.get())
        clamped = max(0.2, min(2.5, val))
        if clamped != val:
            self.edge_gamma_var.set(clamped)
        self.edge_gamma_display.set(f"{clamped:.2f}")

    def _on_edge_saliency_change(self) -> None:
        """サリエンシー寄与率(%)を更新する。"""
        val = float(self.edge_saliency_weight_var.get())
        clamped = max(0.0, min(100.0, val))
        if clamped != val:
            self.edge_saliency_weight_var.set(clamped)
        self.edge_saliency_display.set(f"{clamped:.0f}")

    def _on_adaptive_pointer(self, event: tk.Event) -> str:
        """スライダーをドラッグした位置から0〜100の値を設定する。"""
        scale: ttk.Scale = event.widget  # type: ignore[assignment]
        width = max(1, scale.winfo_width())
        fraction = max(0.0, min(1.0, event.x / width))
        new_val = 100.0 * fraction
        self.adaptive_weight_var.set(round(new_val, 0))
        self._on_adaptive_weight_change()
        return "break"

    def _set_scale_by_pointer(
        self, event: tk.Event, var: tk.Variable, on_change: Callable[[], None]
    ) -> str:
        """スライダー内クリック位置から値を直接設定する共通処理。"""
        scale: ttk.Scale = event.widget  # type: ignore[assignment]
        width = max(1, scale.winfo_width())
        fraction = max(0.0, min(1.0, event.x / width))
        min_val = float(scale.cget("from"))
        max_val = float(scale.cget("to"))
        new_val = min_val + (max_val - min_val) * fraction
        var.set(new_val)
        on_change()
        return "break"

    def _on_edge_pointer(self, event: tk.Event) -> str:
        """輪郭強調スライダーの直接クリック用ハンドラ。"""
        return self._set_scale_by_pointer(event, self.edge_strength_var, self._on_edge_strength_change)

    def _on_edge_thickness_pointer(self, event: tk.Event) -> str:
        """輪郭太さスライダーの直接クリック用ハンドラ。"""
        return self._set_scale_by_pointer(event, self.edge_thickness_var, self._on_edge_thickness_change)

    def _on_edge_gain_pointer(self, event: tk.Event) -> str:
        """輪郭ゲインスライダーの直接クリック用ハンドラ。"""
        return self._set_scale_by_pointer(event, self.edge_gain_var, self._on_edge_gain_change)

    def _on_edge_gamma_pointer(self, event: tk.Event) -> str:
        """輪郭ガンマスライダーの直接クリック用ハンドラ。"""
        return self._set_scale_by_pointer(event, self.edge_gamma_var, self._on_edge_gamma_change)

    def _on_edge_saliency_pointer(self, event: tk.Event) -> str:
        """サリエンシー寄与スライダーの直接クリック用ハンドラ。"""
        return self._set_scale_by_pointer(event, self.edge_saliency_weight_var, self._on_edge_saliency_change)

    def _on_hybrid_pointer(self, event: tk.Event) -> str:
        """ハイブリッド縮小率スライダーのクリック位置から値を設定。"""
        return self._set_scale_by_pointer(event, self.hybrid_scale_var, self._on_hybrid_scale_change)

    def _on_brush_radius_pointer(self, event: tk.Event) -> str:
        """ブラシ半径スライダーのクリック位置から値を設定。"""
        return self._set_scale_by_pointer(event, self.brush_radius_var, self._on_brush_radius_change)

    def _on_brush_strength_pointer(self, event: tk.Event) -> str:
        """ブラシ強さスライダーのクリック位置から値を設定。"""
        return self._set_scale_by_pointer(event, self.brush_strength_var, self._on_brush_strength_change)

    def _on_hybrid_scale_change(self) -> None:
        """ハイブリッド縮小率スライダーの値を10?100で保持する。"""
        val = float(self.hybrid_scale_var.get())
        clamped = max(10.0, min(100.0, val))
        if clamped != val:
            self.hybrid_scale_var.set(clamped)
        self.hybrid_scale_display.set(f"{clamped:.0f}")

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
        """リサイズ方式が適応型ブロック分割のときだけスライダーを有効化。"""
        is_adaptive = self.resize_method_var.get() == "適応型ブロック分割"
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
        """処理順序は常に選択可。ハイブリッド時だけ縮小率スライダーを有効。"""
        try:
            self.pipeline_box.state(["!disabled"])
        except Exception:
            pass
        is_hybrid = self.pipeline_var.get() == "ハイブリッド"
        state_token = "!disabled" if is_hybrid else "disabled"
        try:
            self.hybrid_scale.state([state_token])
        except Exception:
            pass
        if hasattr(self, "hybrid_label"):
            self.hybrid_label.configure(foreground="#000" if is_hybrid else "#888")

    def _update_edge_controls(self) -> None:
        """輪郭強調のON/OFFに応じて強さスライダーを切り替える。"""
        enabled = self.edge_enhance_var.get()
        state_token = "!disabled" if enabled else "disabled"
        try:
            self.edge_strength_scale.state([state_token])
        except Exception:
            pass
        try:
            self.edge_thickness_scale.state([state_token])
        except Exception:
            pass
        for scale_name in ["edge_gain_scale", "edge_gamma_scale", "edge_saliency_scale"]:
            widget = getattr(self, scale_name, None)
            if widget:
                try:
                    widget.state([state_token])
                except Exception:
                    pass
        if hasattr(self, "edge_label"):
            self.edge_label.configure(foreground="#000" if enabled else "#888")
        if hasattr(self, "edge_thickness_label"):
            self.edge_thickness_label.configure(foreground="#000" if enabled else "#888")
        if hasattr(self, "edge_gain_label"):
            self.edge_gain_label.configure(foreground="#000" if enabled else "#888")
        if hasattr(self, "edge_gamma_label"):
            self.edge_gamma_label.configure(foreground="#000" if enabled else "#888")
        if hasattr(self, "edge_saliency_label"):
            self.edge_saliency_label.configure(foreground="#000" if enabled else "#888")

    def _apply_saved_settings(self) -> None:
        """settings.jsonに保存された前回値をUIへ反映する。"""
        if not self.last_settings:
            return

        def _sanitize_choice(value: Optional[str], allowed: set[str], fallback: str) -> str:
            """コンボボックスに存在しない値を弾きつつ既定値へフォールバックする。"""
            if value in allowed:
                return value  # type: ignore[return-value]
            return fallback

        allowed_quant = {"なし", "K-means", "Wu減色"}
        allowed_pipeline = {"リサイズ→減色", "減色→リサイズ", "ハイブリッド"}
        allowed_resize = {"ニアレストネイバー", "バイリニア", "バイキュービック", "ブロック分割", "適応型ブロック分割"}
        sanitized_settings = dict(self.last_settings)
        sanitized_settings.pop("分割方式", None)  # 旧キーは破棄
        sanitized_settings.pop("輪郭強調", None)  # 廃止した項目を破棄
        # 起動直後は幅・高さ欄を必ず空にする（前回値は復元しない）
        self.width_var.set("")
        self.height_var.set("")
        try:
            self.num_colors_var.set(str(int(self.last_settings.get("減色後色数", self.num_colors_var.get()))))
        except Exception:
            pass
        mode = self.last_settings.get("モード")
        if mode:
            self.mode_var.set(mode)
        resize_label = self.last_settings.get("リサイズ方式")
        resize_label = _sanitize_choice(resize_label, allowed_resize, self.resize_method_var.get())
        self.resize_method_var.set(resize_label)
        quant_label = self.last_settings.get("減色方式")
        quant_label = _sanitize_choice(quant_label, allowed_quant, self.quantize_method_var.get())
        pipeline = _sanitize_choice(self.last_settings.get("処理順序"), allowed_pipeline, self.pipeline_var.get())
        sanitized_settings["減色方式"] = quant_label
        sanitized_settings["処理順序"] = pipeline
        sanitized_settings["リサイズ方式"] = resize_label
        self.quantize_method_var.set(quant_label)
        self.pipeline_var.set(pipeline)
        adaptive_val = self.last_settings.get("適応細かさ")
        try:
            if adaptive_val is not None:
                val = float(adaptive_val)
                self.adaptive_weight_var.set(val)
                self.adaptive_weight_display.set(f"{val:.0f}")
        except Exception:
            pass
        hybrid_val = self.last_settings.get("ハイブリッド縮小率")
        try:
            if hybrid_val is not None:
                h_val = float(hybrid_val)
                self.hybrid_scale_var.set(h_val)
                self.hybrid_scale_display.set(f"{h_val:.0f}")
        except Exception:
            pass
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
        sanitized_settings.setdefault("CMC l", f"{float(self.cmc_l_var.get()):.1f}")
        sanitized_settings.setdefault("CMC c", f"{float(self.cmc_c_var.get()):.1f}")
        edge_flag = self.last_settings.get("輪郭強調(新)")
        if isinstance(edge_flag, bool):
            self.edge_enhance_var.set(edge_flag)
        edge_strength = self.last_settings.get("輪郭強さ")
        try:
            if edge_strength is not None:
                e_val = float(edge_strength)
                self.edge_strength_var.set(e_val)
                self.edge_strength_display.set(f"{max(0.0, min(100.0, e_val)):.0f}")
        except Exception:
            pass
        sanitized_settings.setdefault("輪郭強調(新)", bool(self.edge_enhance_var.get()))
        sanitized_settings.setdefault("輪郭強さ", f"{float(self.edge_strength_var.get()):.0f}")
        edge_thickness = self.last_settings.get("輪郭太さ")
        try:
            if edge_thickness is not None:
                t_val = float(edge_thickness)
                self.edge_thickness_var.set(t_val)
                self.edge_thickness_display.set(f"{max(0.0, min(100.0, t_val)):.0f}")
        except Exception:
            pass
        sanitized_settings.setdefault("輪郭太さ", f"{float(self.edge_thickness_var.get()):.0f}")
        edge_gain = self.last_settings.get("輪郭ゲイン")
        try:
            if edge_gain is not None:
                g_val = float(edge_gain)
                self.edge_gain_var.set(g_val)
                self.edge_gain_display.set(f"{max(0.0, min(5.0, g_val)):.2f}")
        except Exception:
            pass
        sanitized_settings.setdefault("輪郭ゲイン", f"{float(self.edge_gain_var.get()):.2f}")
        edge_gamma = self.last_settings.get("輪郭ガンマ")
        try:
            if edge_gamma is not None:
                gm_val = float(edge_gamma)
                self.edge_gamma_var.set(gm_val)
                self.edge_gamma_display.set(f"{max(0.2, min(2.5, gm_val)):.2f}")
        except Exception:
            pass
        sanitized_settings.setdefault("輪郭ガンマ", f"{float(self.edge_gamma_var.get()):.2f}")
        edge_saliency = self.last_settings.get("輪郭サリエンシー寄与(%)")
        try:
            if edge_saliency is not None:
                s_val = float(edge_saliency)
                self.edge_saliency_weight_var.set(s_val)
                self.edge_saliency_display.set(f"{max(0.0, min(100.0, s_val)):.0f}")
        except Exception:
            pass
        sanitized_settings.setdefault("輪郭サリエンシー寄与(%)", f"{float(self.edge_saliency_weight_var.get()):.0f}")
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
        sanitized_settings.setdefault(
            "RGB重み",
            [
                float(self.rgb_r_weight_var.get()),
                float(self.rgb_g_weight_var.get()),
                float(self.rgb_b_weight_var.get()),
            ],
        )
        # サニタイズ後の内容を今後の差分比較・保存に反映させる
        self.last_settings = sanitized_settings
        # 状態反映
        self._update_num_colors_state()
        self._update_adaptive_controls()
        self._update_edge_controls()
        if hasattr(self, "_update_mode_frames"):
            try:
                self._update_mode_frames()
            except Exception:
                pass
        self._update_pipeline_controls()
        self._update_cmc_controls()
