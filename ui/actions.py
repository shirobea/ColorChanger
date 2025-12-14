"""ユーザー操作を司るアクション層のMixin。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image

import converter
from .models import ConversionRequest

if TYPE_CHECKING:
    from .app import BeadsApp


class ActionsMixin:
    """画像選択・保存・変換開始/停止などのユーザー操作ハンドラ。"""

    def select_image(self: "BeadsApp") -> None:
        """Open file dialog and preview the chosen image."""
        path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_image_path = Path(path)
        try:
            image = Image.open(self.input_image_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("読込エラー", f"画像を開けませんでした: {exc}")
            return
        self.input_pil = image
        self.output_pil = None
        self.output_image = None
        self._output_photo = None
        self.saliency_map = None
        self.importance_map = None
        self._base_importance_map = None
        if hasattr(self, "importance_editor"):
            self.importance_editor.clear()
        self.saliency_overlay_pil = None
        self._show_saliency = False
        self._set_saliency_button_state(enabled=False)
        if hasattr(self, "_update_importance_controls_state"):
            try:
                self._update_importance_controls_state(False)
            except Exception:
                pass
        self.original_size = image.size
        self.output_canvas.configure(image="", text="変換後")
        self._set_initial_target_size(image)
        self._compute_and_store_importance(image)
        self._refresh_previews()

    def _on_space_key(self: "BeadsApp", _event: "tk.Event") -> str:
        """スペースキーで変換開始/中止をトグルするショートカット。"""
        # 変換中なら中止、待機中なら開始
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_conversion()
        else:
            self.start_conversion()
        # ここでイベント伝播を止める（ボタンのactivateを防ぐ）
        return "break"

    def start_conversion(self: "BeadsApp") -> None:
        """Kick off conversion in a worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return
        request = self._gather_request()
        if request is None:
            return
        self._pending_settings = self._build_pending_settings(request)
        self._prepare_conversion_ui()
        self.worker_thread = threading.Thread(
            target=self._run_conversion,
            args=(request, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()

    def _gather_request(self: "BeadsApp") -> Optional[ConversionRequest]:
        """入力値を検証してConversionRequestにまとめる。"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            num_colors = int(self.num_colors_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "幅・高さ・色数には整数を入力してください。")
            return None
        if width <= 0 or height <= 0:
            messagebox.showerror("入力エラー", "幅・高さは1以上にしてください。")
            return None
        cmc_l = float(self.cmc_l_var.get())
        cmc_c = float(self.cmc_c_var.get())
        cmc_l = max(0.5, min(3.0, cmc_l))
        cmc_c = max(0.5, min(3.0, cmc_c))
        quant_label = self.quantize_method_var.get()
        # UI表示用のラベルを内部向けのコードへ正規化する
        quant_method = {
            "なし": "none",
            "K-means": "kmeans",
            "Wu減色": "wu",
        }.get(quant_label, quant_label).lower()
        pipeline = {
            "リサイズ→減色": "resize_first",
            "減色→リサイズ": "quantize_first",
            "ハイブリッド": "hybrid",
        }.get(self.pipeline_var.get(), "resize_first")
        resize_label = self.resize_method_var.get()
        resize_method = {
            "ニアレストネイバー": "nearest",
            "バイリニア": "bilinear",
            "バイキュービック": "bicubic",
            "ブロック分割": "block",
            "適応型ブロック分割": "adaptive_block",
        }.get(resize_label, "nearest")
        keep_aspect = self.lock_aspect_var.get()
        adaptive_w = float(self.adaptive_weight_var.get()) / 100.0
        hybrid_scale = float(self.hybrid_scale_var.get()) / 100.0
        edge_enhance = self.edge_enhance_var.get()
        edge_strength = max(0.0, min(1.0, float(self.edge_strength_var.get()) / 100.0))
        edge_thickness = max(0.0, min(1.0, float(self.edge_thickness_var.get()) / 100.0))
        edge_gain = max(0.0, min(10.0, float(self.edge_gain_var.get())))
        edge_gamma = max(0.2, min(2.5, float(self.edge_gamma_var.get())))
        edge_saliency = max(0.0, min(1.0, float(self.edge_saliency_weight_var.get()) / 100.0))
        r_w = max(0.5, min(2.0, float(self.rgb_r_weight_var.get())))
        g_w = max(0.5, min(2.0, float(self.rgb_g_weight_var.get())))
        b_w = max(0.5, min(2.0, float(self.rgb_b_weight_var.get())))
        return ConversionRequest(
            width=width,
            height=height,
            num_colors=num_colors,
            mode=self.mode_var.get().replace(" (CIEDE2000)", ""),
            cmc_l=cmc_l,
            cmc_c=cmc_c,
            quantize_method=quant_method,
            keep_aspect=keep_aspect,
            pipeline=pipeline,
            adaptive_weight=adaptive_w,
            hybrid_scale=hybrid_scale,
            resize_method=resize_method,
            rgb_weights=(r_w, g_w, b_w),
            edge_enhance=edge_enhance,
            edge_strength=edge_strength,
            edge_thickness=edge_thickness,
            edge_gain=edge_gain,
            edge_gamma=edge_gamma,
            edge_saliency_weight=edge_saliency,
        )

    def _build_pending_settings(self: "BeadsApp", request: ConversionRequest) -> dict:
        """表示用に設定差分を作成する。"""
        quant_label = {
            "none": "なし",
            "kmeans": "K-means",
            "wu": "Wu減色",
            "block": "ブロック分割",
            "adaptive_block": "適応型ブロック分割",
        }.get(request.quantize_method, request.quantize_method)
        pipeline_label = {
            "resize_first": "リサイズ→減色",
            "quantize_first": "減色→リサイズ",
            "hybrid": "ハイブリッド",
        }.get(request.pipeline, request.pipeline)
        resize_label = {
            "nearest": "ニアレストネイバー",
            "bilinear": "バイリニア",
            "bicubic": "バイキュービック",
            "block": "ブロック分割",
            "adaptive_block": "適応型ブロック分割",
        }.get(request.resize_method, request.resize_method)
        cmc_l = f"{request.cmc_l:.1f}"
        cmc_c = f"{request.cmc_c:.1f}"
        return {
            "幅": request.width,
            "高さ": request.height,
            "減色後色数": request.num_colors,
            "モード": request.mode,
            "CMC l": cmc_l,
            "CMC c": cmc_c,
            "減色方式": quant_label,
            "処理順序": pipeline_label,
            "適応細かさ": f"{request.adaptive_weight*100:.0f}",
            "ハイブリッド縮小率": f"{request.hybrid_scale*100:.0f}",
            "リサイズ方式": resize_label,
            "輪郭強調(新)": request.edge_enhance,
            "輪郭強さ(0-100)": f"{request.edge_strength*100:.0f}",
            "輪郭太さ(0-100)": f"{request.edge_thickness*100:.0f}",
            "RGB重み": [round(request.rgb_weights[0], 1), round(request.rgb_weights[1], 1), round(request.rgb_weights[2], 1)],
        }

    def _prepare_conversion_ui(self: "BeadsApp") -> None:
        """変換開始直前のUI初期化。"""
        self.save_button.configure(state="disabled")
        self.cancel_event = threading.Event()
        self._start_time = time.perf_counter()
        self.update_progress(0.0)
        self.status_var.set("変換中...")
        self.convert_button.configure(text="変換中止", state="normal", command=self.cancel_conversion)

    def cancel_conversion(self: "BeadsApp") -> None:
        """ユーザー操作で変換を中断する。"""
        if self.cancel_event:
            self.cancel_event.set()
        self.status_var.set("中止要求を送信しました...")
        self.convert_button.configure(state="disabled", text="停止中...")
        self._start_time = None
        self._reset_progress_display()

    def _run_conversion(
        self: "BeadsApp",
        request: ConversionRequest,
        cancel_event: Optional[threading.Event],
    ) -> None:
        """バックグラウンドで変換を実行するワーカー。"""

        def progress_cb(value: float) -> None:
            self.root.after(0, self.update_progress, value)

        try:
            result = converter.convert_image(
                input_path=str(self.input_image_path),
                output_size=(request.width, request.height),
                mode=request.mode,
                cmc_l=request.cmc_l,
                cmc_c=request.cmc_c,
                palette=self.palette,
                num_colors=request.num_colors,
                quantize_method=request.quantize_method,
                keep_aspect=request.keep_aspect,
                pipeline=request.pipeline,
                eye_importance_scale=0.8,
                edge_enhance=request.edge_enhance,
                edge_strength=request.edge_strength,
                edge_thickness=request.edge_thickness,
                edge_gain=request.edge_gain,
                edge_gamma=request.edge_gamma,
                edge_saliency_weight=request.edge_saliency_weight,
                adaptive_saliency_weight=request.adaptive_weight,
                hybrid_scale_percent=request.hybrid_scale * 100.0,
                resize_method=request.resize_method,
                rgb_weights=request.rgb_weights,
                progress_callback=progress_cb,
                cancel_event=cancel_event,
                saliency_map=self.saliency_map,
            )
        except converter.ConversionCancelled:
            self.root.after(0, self._on_cancelled)
            return
        except Exception as exc:
            self.root.after(
                0,
                lambda: messagebox.showerror("変換失敗", f"変換中にエラーが発生しました:\n{exc}"),
            )
            self.root.after(0, lambda: self._handle_failure("変換に失敗しました"))
            return

        # 自動保存は行わず、メモリ上で保持するのみ
        self.output_path = None
        if self.output_pil:
            self.prev_output_pil = self.output_pil
            self.prev_settings = self.last_settings
        else:
            self.prev_output_pil = None
            self.prev_settings = None
        self._showing_prev = False
        self.output_image = result
        self.output_pil = Image.fromarray(result)
        self.last_settings = self._pending_settings
        self._pending_settings = None
        self.diff_var.set(self._build_diff_overlay())
        self._save_settings()

        def on_finish() -> None:
            self._refresh_previews()
            self.update_progress(1.0)
            self._restore_convert_button()
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
            self.cancel_event = None
            self.worker_thread = None
            self.status_var.set("変換完了（保存ボタンで任意の場所に保存できます）")

        self.root.after(0, on_finish)

    def _on_cancelled(self: "BeadsApp") -> None:
        """キャンセル完了時のUI復帰処理。"""
        # 途中で止めても直前に成功していた出力は残す
        self._reset_after_stop("中止しました", clear_canvas=False, preserve_output=True)

    def _handle_failure(self: "BeadsApp", status: str) -> None:
        """失敗時の共通後始末。"""
        self._reset_after_stop(status, clear_canvas=False)

    def save_image(self: "BeadsApp") -> None:
        """出力画像を任意パスへ保存。"""
        if self.output_image is None:
            self.status_var.set("出力画像がまだありません。")
            return
        initial_dir = str(self.input_image_path.parent) if self.input_image_path else str(Path.cwd())
        default_name = (
            f"{self.input_image_path.stem}_beads.png" if self.input_image_path else "output_beads.png"
        )
        path = filedialog.asksaveasfilename(
            title="出力先を選択",
            defaultextension=".png",
            initialdir=initial_dir,
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("保存をキャンセルしました。")
            return
        try:
            Image.fromarray(self.output_image).save(path)
            self.output_path = Path(path)
            self.status_var.set("保存しました")
        except Exception as exc:
            messagebox.showerror("保存失敗", f"出力画像の保存に失敗しました:\n{exc}")

    def _reset_progress_display(self: "BeadsApp") -> None:
        """進捗バーと時間表示を初期化。"""
        self.progress_label.configure(text="進捗: 0% (経過 0.0s)")
        self.progress_bar["value"] = 0

    def update_progress(self: "BeadsApp", value: float) -> None:
        """進捗UIを更新する。"""
        clamped = max(0.0, min(1.0, value))
        percent = int(clamped * 100)
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
        self.progress_label.configure(text=f"進捗: {percent}% (経過 {elapsed:.1f}s)")
        self.progress_bar["value"] = percent

    def _reset_after_stop(
        self: "BeadsApp",
        status: str,
        clear_canvas: bool,
        preserve_output: bool = False,
    ) -> None:
        """停止時に共通で状態をリセットする。"""
        self.cancel_event = None
        self.worker_thread = None
        self._start_time = None
        if not preserve_output:
            self.output_image = None
            self.output_pil = None
            self.prev_output_pil = None
            self.output_path = None
            self.diff_var.set("")
        self._pending_settings = None
        self._reset_progress_display()
        self._restore_convert_button()
        if preserve_output:
            # 以前の出力があれば保存できるように戻す
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        else:
            self.save_button.configure(state="disabled")
        if clear_canvas and not preserve_output:
            self.output_canvas.configure(image="", text="変換後")
        self.status_var.set(status)

    def _restore_convert_button(self: "BeadsApp") -> None:
        """変換ボタンを初期状態に戻す。"""
        self.convert_button.configure(text="変換実行", state="normal", command=self.start_conversion)
