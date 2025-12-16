"""ユーザー操作を司るアクション層のMixin。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Callable, Any

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

import converter
from .models import ConversionRequest

if TYPE_CHECKING:
    from .app import BeadsApp


class ActionsMixin:
    """画像選択・保存・変換開始/停止などのユーザー操作ハンドラ。"""

    def _schedule_on_ui(self: "BeadsApp", delay_ms: int, func: Callable[..., None], *args: Any) -> None:
        """閉じる途中はUIスレッドへの投げ込みを抑止する。"""
        if getattr(self, "_closing", False):
            return
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return
        try:
            self.root.after(delay_ms, func, *args)
        except Exception:
            pass

    def select_image(self: "BeadsApp") -> None:
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
        # 入力を変えたら前回出力のプレビューは破棄してブレンド表示の混在を防ぐ
        self.prev_output_pil = None
        self._showing_prev = False
        self._output_photo = None
        self.original_size = image.size
        self.output_canvas.configure(image="", text="変換後")
        self._set_initial_target_size(image)
        self._refresh_previews()

    def _on_space_key(self: "BeadsApp", _event: "tk.Event") -> str:
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_conversion()
        else:
            self.start_conversion()
        return "break"

    def start_conversion(self: "BeadsApp") -> None:
        if self._runner.is_running:
            return
        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return
        request = self._gather_request()
        if request is None:
            return
        self._pending_settings = self._build_pending_settings(request)
        self._prepare_conversion_ui()
        started = self._runner.start(
            request=request,
            input_path=str(self.input_image_path),
            palette=self.palette,
            on_progress=self.update_progress,
            on_success=self._on_conversion_success,
            on_cancelled=self._on_cancelled,
            on_error=self._handle_worker_error,
        )
        if not started:
            self.status_var.set("既に変換中です。")

    def _gather_request(self: "BeadsApp") -> Optional[ConversionRequest]:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "幅・高さには整数を入力してください。")
            return None
        if width <= 0 or height <= 0:
            messagebox.showerror("入力エラー", "幅・高さは1以上にしてください。")
            return None
        cmc_l = float(self.cmc_l_var.get())
        cmc_c = float(self.cmc_c_var.get())
        cmc_l = max(0.5, min(3.0, cmc_l))
        cmc_c = max(0.5, min(3.0, cmc_c))
        resize_label = self.resize_method_var.get()
        resize_method = {
            "ニアレストネイバー": "nearest",
            "バイリニア": "bilinear",
            "バイキュービック": "bicubic",
        }.get(resize_label, "nearest")
        keep_aspect = self.lock_aspect_var.get()
        r_w = max(0.5, min(2.0, float(self.rgb_r_weight_var.get())))
        g_w = max(0.5, min(2.0, float(self.rgb_g_weight_var.get())))
        b_w = max(0.5, min(2.0, float(self.rgb_b_weight_var.get())))
        return ConversionRequest(
            width=width,
            height=height,
            mode=self.mode_var.get().replace(" (CIEDE2000)", ""),
            lab_metric=self.lab_metric_var.get(),
            cmc_l=cmc_l,
            cmc_c=cmc_c,
            keep_aspect=keep_aspect,
            resize_method=resize_method,
            rgb_weights=(r_w, g_w, b_w),
        )

    def _build_pending_settings(self: "BeadsApp", request: ConversionRequest) -> dict:
        cmc_l = f"{request.cmc_l:.1f}"
        cmc_c = f"{request.cmc_c:.1f}"
        resize_label = {
            "nearest": "ニアレストネイバー",
            "bilinear": "バイリニア",
            "bicubic": "バイキュービック",
        }.get(request.resize_method, request.resize_method)
        return {
            "幅": request.width,
            "高さ": request.height,
            "モード": request.mode,
            "Lab距離式": request.lab_metric,
            "CMC l": cmc_l,
            "CMC c": cmc_c,
            "リサイズ方式": resize_label,
            "RGB重み": [round(request.rgb_weights[0], 1), round(request.rgb_weights[1], 1), round(request.rgb_weights[2], 1)],
        }

    def _prepare_conversion_ui(self: "BeadsApp") -> None:
        self.save_button.configure(state="disabled")
        self._start_time = time.perf_counter()
        self.update_progress(0.0)
        self.status_var.set("変換中...")
        self.convert_button.configure(text="変換中止", state="normal", command=self.cancel_conversion)

    def cancel_conversion(self: "BeadsApp") -> None:
        self._runner.cancel()
        self.status_var.set("中止要求を送信しました...")
        self.convert_button.configure(state="disabled", text="停止中...")
        self._start_time = None
        self._reset_progress_display()

    def _on_conversion_success(self: "BeadsApp", result: np.ndarray) -> None:
        """ワーカースレッド成功時のUI側反映。"""
        if getattr(self, "_closing", False):
            return
        self.output_path = None
        self.prev_output_pil = self.output_pil
        self.prev_settings = self.last_settings
        self._showing_prev = False
        self.output_image = result
        self.output_pil = Image.fromarray(result)
        self.last_settings = self._pending_settings
        self._pending_settings = None
        self.diff_var.set(self._build_diff_overlay())
        self._save_settings()
        self._refresh_previews()
        self.update_progress(1.0)
        self._restore_convert_button()
        self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        self.status_var.set("変換完了（保存ボタンで任意の場所に保存できます）")
        self._showing_prev = False

    def _handle_worker_error(self: "BeadsApp", exc: Exception) -> None:
        """ワーカースレッドで例外が出た場合のUI処理。"""
        messagebox.showerror("変換失敗", f"変換中にエラーが発生しました:\n{exc}")
        self._handle_failure("変換に失敗しました")

    def _on_cancelled(self: "BeadsApp") -> None:
        self._reset_after_stop("中止しました", clear_canvas=False, preserve_output=True)

    def _handle_failure(self: "BeadsApp", status: str) -> None:
        self._reset_after_stop(status, clear_canvas=False)

    def save_image(self: "BeadsApp") -> None:
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
        self.progress_label.configure(text="進捗: 0% (経過 0.0s)")
        self.progress_bar["value"] = 0

    def update_progress(self: "BeadsApp", value: float) -> None:
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
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        else:
            self.save_button.configure(state="disabled")
        if clear_canvas and not preserve_output:
            self.output_canvas.configure(image="", text="変換後")
        self.status_var.set(status)

    def _restore_convert_button(self: "BeadsApp") -> None:
        self.convert_button.configure(text="変換実行", state="normal", command=self.start_conversion)
