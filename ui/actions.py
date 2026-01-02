"""ユーザー操作を司るアクション層のMixin。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Callable, Any

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageFilter

import converter
import cv2
from color_spaces import rgb_to_lab
from .models import ConversionRequest
from .color_usage_window import ColorUsageWindow
import numpy as np

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
        self.input_original_pil = image
        self.input_filtered_pil = None
        self.input_pil = image
        self._input_using_filtered = False
        self._showing_input_overlay = False
        filters = self._get_noise_filter_registry()
        if self.noise_filter_var.get() not in filters:
            self.noise_filter_var.set(next(iter(filters)))
        self.output_pil = None
        self.output_image = None
        self.color_usage = []
        self._set_color_usage_button_state(False)
        self._refresh_color_usage_window(reset_sort=False)
        # 入力を変えたら前回出力のプレビューは破棄してブレンド表示の混在を防ぐ
        self.prev_output_pil = None
        self._showing_prev = False
        self._output_photo = None
        self.original_size = image.size
        self.output_canvas.configure(image="", text="変換後")
        self._set_initial_target_size(image)
        self._refresh_previews()
        self.rgb_log_var.set("入力画像を読み込みました。RGB最適化を行う場合はボタンを押してください。")

    def _sanitize_kernel_size(self: "BeadsApp", raw_size: object) -> int:
        """メディアン用カーネルサイズを奇数・下限付きで整える。"""
        try:
            size = int(raw_size)
        except Exception:
            size = 3
        size = max(3, size)
        if size % 2 == 0:
            size += 1
        self.noise_filter_size_var.set(size)
        return size

    def _set_noise_busy(self: "BeadsApp", busy: bool) -> None:
        """ノイズ除去処理中はUI操作を抑止する。"""
        self._noise_busy = busy
        state_token = "disabled" if busy else "normal"
        for btn_name in ("noise_apply_button", "noise_reset_button"):
            btn = getattr(self, btn_name, None)
            if btn:
                try:
                    btn.configure(state=state_token)
                except Exception:
                    pass
        if busy:
            self.status_var.set("ノイズ除去を実行中です...")

    def _get_noise_filter_registry(self: "BeadsApp") -> dict[str, Callable[[Image.Image], Image.Image]]:
        """追加しやすいようフィルタ名と実装を辞書でまとめる。"""
        size = self._sanitize_kernel_size(self.noise_filter_size_var.get())
        return {
            "メディアン": lambda img: Image.fromarray(
                cv2.medianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), size)
            ),
            "ガウシアン": lambda img: Image.fromarray(
                cv2.GaussianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), (size, size), 0)
            ),
            "バイラテラル": lambda img: Image.fromarray(
                cv2.bilateralFilter(
                    np.asarray(img.convert("RGB"), dtype=np.uint8),
                    size,  # 近傍直径
                    sigmaColor=float(size * 6),  # 色差に対する標準偏差
                    sigmaSpace=float(size * 2),  # 座標距離に対する標準偏差
                )
            ),
            "非局所的平均": lambda img: Image.fromarray(
                cv2.fastNlMeansDenoisingColored(
                    np.asarray(img.convert("RGB"), dtype=np.uint8),
                    None,
                    h=8.0,
                    hColor=10.0,
                    templateWindowSize=size,
                    searchWindowSize=max(7, size * 3),
                )
            ),
        }

    def apply_noise_reduction(self: "BeadsApp") -> None:
        """入力画像にノイズ除去フィルタを適用する（別スレッドで実行）。"""
        if self.input_original_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        if getattr(self, "_noise_busy", False):
            return
        filters = self._get_noise_filter_registry()
        name = self.noise_filter_var.get()
        filter_func = filters.get(name)
        if filter_func is None:
            messagebox.showerror("フィルタ未対応", f"選択したフィルタ「{name}」は未対応です。")
            return

        self._set_noise_busy(True)

        def _worker() -> None:
            try:
                # PILオブジェクトはスレッド間で共有しないようコピー
                src = self.input_original_pil.copy() if self.input_original_pil else None
                if src is None:
                    raise ValueError("入力画像が見つかりません。")
                filtered_img = filter_func(src)
            except Exception as exc:  # スレッド内で例外を捕捉してUIスレッドへ渡す
                self._schedule_on_ui(0, lambda: self._on_noise_failed(exc))
                return
            self._schedule_on_ui(0, lambda: self._on_noise_finished(name, filtered_img))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_noise_finished(self: "BeadsApp", name: str, filtered: Image.Image) -> None:
        """ノイズ除去成功時のUI反映（UIスレッドで実行）。"""
        self.input_filtered_pil = filtered
        self.input_pil = filtered
        self._input_using_filtered = True
        self._showing_input_overlay = False
        self.status_var.set(f"{name}フィルタでノイズ除去しました。")
        self.rgb_log_var.set(f"{name}フィルタでノイズ除去しました。")
        self._refresh_previews()
        self._set_noise_busy(False)

    def _on_noise_failed(self: "BeadsApp", exc: Exception) -> None:
        """ノイズ除去失敗時のUI処理（UIスレッドで実行）。"""
        messagebox.showerror("ノイズ除去失敗", f"ノイズ除去中にエラーが発生しました:\n{exc}")
        self._set_noise_busy(False)

    def reset_noise_reduction(self: "BeadsApp") -> None:
        """ノイズ除去前の元画像に戻す。"""
        if self.input_original_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        if getattr(self, "_noise_busy", False):
            return
        self._set_noise_busy(True)
        self.input_filtered_pil = None
        self.input_pil = self.input_original_pil
        self._input_using_filtered = False
        self._showing_input_overlay = False
        self.status_var.set("入力画像を元に戻しました。")
        self.rgb_log_var.set("ノイズ除去をリセットしました。")
        self._refresh_previews()
        self._set_noise_busy(False)

    def _on_space_key(self: "BeadsApp", _event: "tk.Event") -> str:
        if self.worker_thread and self.worker_thread.is_alive():
            self.cancel_conversion()
        else:
            self.start_conversion()
        return "break"

    def _get_active_input_array(self: "BeadsApp") -> Optional[np.ndarray]:
        """現在表示中の入力画像をRGB配列にして返す。"""
        if self.input_pil is None:
            return None
        return np.asarray(self.input_pil.convert("RGB"), dtype=np.uint8)

    def start_conversion(self: "BeadsApp") -> None:
        if self._runner.is_running:
            return
        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return
        request = self._gather_request()
        if request is None:
            return
        input_image = self._get_active_input_array()
        if input_image is None:
            messagebox.showerror("入力エラー", "入力画像を読み込んでから変換してください。")
            return
        self._pending_settings = self._build_pending_settings(request)
        self._prepare_conversion_ui()
        started = self._runner.start(
            request=request,
            input_path=str(self.input_image_path),
            input_image=input_image,
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
        self._set_color_usage_button_state(False)
        self._start_time = time.perf_counter()
        self.update_progress(0.0)
        self.status_var.set("変換中...")
        self.convert_button.configure(text="変換中止", state="normal", command=self.cancel_conversion)

    def _set_color_usage_button_state(self: "BeadsApp", enabled: bool) -> None:
        """色使用一覧ボタンの有効/無効を切り替える。"""
        btn = getattr(self, "color_usage_button", None)
        if not btn:
            return
        state_token = "normal" if enabled else "disabled"
        try:
            btn.configure(state=state_token)
        except Exception:
            pass

    def _build_color_usage_rows(self: "BeadsApp", image: np.ndarray, settings: Optional[dict]) -> list[dict]:
        """変換後画像から色使用数の一覧を作る。"""
        if image is None:
            return []
        mode = ""
        if settings:
            mode = str(settings.get("モード", ""))
        if mode.lower() in {"none", "なし"}:
            return []
        palette_map: dict[tuple[int, int, int], dict[str, str]] = {}
        for color in self.palette:
            rgb = tuple(int(round(v)) for v in color.rgb)
            palette_map[rgb] = {"color_id": color.color_id, "name": color.name}
        flat = image.reshape(-1, 3)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        rows: list[dict] = []
        for rgb_arr, count in zip(colors, counts):
            rgb = (int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]))
            info = palette_map.get(rgb)
            if not info:
                continue
            rows.append(
                {
                    "color_id": info["color_id"],
                    "name": info["name"],
                    "count": int(count),
                    "rgb": rgb,
                }
            )
        rows.sort(key=lambda r: int(r.get("count", 0)), reverse=True)
        return rows

    def _refresh_color_usage_window(self: "BeadsApp", reset_sort: bool) -> None:
        """色使用一覧ウィンドウが開いている場合のみ更新する。"""
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            window.update_rows(self.color_usage, reset_sort=reset_sort)

    def _update_color_usage_preview(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> None:
        """色使用一覧のプレビューを更新する。"""
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            window.set_preview_image(self._make_color_usage_preview(rgb))

    def _make_color_usage_preview(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> Optional[Image.Image]:
        """選択色以外を薄くしたプレビュー画像を作る。"""
        if self.output_image is None or not self.color_usage:
            return None
        base = np.asarray(self.output_image, dtype=np.uint8)
        if rgb is None:
            return Image.fromarray(base)
        target = np.array(rgb, dtype=np.uint8)
        mask = np.all(base == target, axis=2)
        if not mask.any():
            return Image.fromarray(base)
        # 選択していない色は白に寄せて薄く見せる
        dim_ratio = 0.15
        dim = base.astype(np.float32) * dim_ratio + 255.0 * (1.0 - dim_ratio)
        dim = dim.astype(np.uint8)
        result = base.copy()
        result[~mask] = dim[~mask]
        return Image.fromarray(result)

    def _on_color_usage_window_closed(self: "BeadsApp") -> None:
        """色使用一覧ウィンドウを閉じた後の参照をクリアする。"""
        self._color_usage_window = None

    def _on_color_usage_select(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> None:
        """色使用一覧で選択された色に合わせてプレビューを更新する。"""
        self._update_color_usage_preview(rgb)

    def show_color_usage(self: "BeadsApp") -> None:
        """色使用一覧ウィンドウを開く。"""
        if not getattr(self, "color_usage", None):
            messagebox.showinfo("色使用一覧", "変換後の色一覧がありません。")
            return
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            window.focus()
            return
        self._color_usage_window = ColorUsageWindow(
            self.root,
            self.color_usage,
            on_close=self._on_color_usage_window_closed,
            on_select=self._on_color_usage_select,
        )
        self._update_color_usage_preview(None)

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
        self.color_usage = self._build_color_usage_rows(result, self._pending_settings)
        self._set_color_usage_button_state(bool(self.color_usage))
        self._refresh_color_usage_window(reset_sort=True)
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
            self.color_usage = []
        self._pending_settings = None
        self._reset_progress_display()
        self._restore_convert_button()
        if preserve_output:
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        else:
            self.save_button.configure(state="disabled")
        self._set_color_usage_button_state(bool(self.output_image is not None and self.color_usage))
        self._refresh_color_usage_window(reset_sort=False)
        if clear_canvas and not preserve_output:
            self.output_canvas.configure(image="", text="変換後")
        self.status_var.set(status)

    def _restore_convert_button(self: "BeadsApp") -> None:
        self.convert_button.configure(text="変換実行", state="normal", command=self.start_conversion)

    # --- CMC(l:c)最適化 ---
    def compute_optimal_cmc_weights(self: "BeadsApp") -> None:
        """入力画像の明度/彩度分布からCMC(l:c)の重みを推定する。"""
        if self.input_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        # Lab空間での明度と彩度の広がりを見てバランスを決める
        lab = rgb_to_lab(np.asarray(self.input_pil.convert("RGB"), dtype=np.uint8))
        l_vals = lab[..., 0]
        chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
        l_spread = float(np.percentile(l_vals, 95) - np.percentile(l_vals, 5))
        c_spread = float(np.percentile(chroma, 95) - np.percentile(chroma, 5))
        l_score = float(l_vals.std()) + 0.5 * l_spread
        c_score = float(chroma.std()) + 0.5 * c_spread
        balance = float(np.clip((l_score - c_score) / max(l_score, c_score, 1e-3), -0.8, 0.8))
        l_w = float(np.clip(2.0 + balance * 1.2, 0.5, 3.0))
        c_w = float(np.clip(1.0 - balance * 0.8, 0.5, 3.0))
        self.cmc_l_var.set(l_w)
        self.cmc_c_var.set(c_w)
        self._on_cmc_l_change()
        self._on_cmc_c_change()
        msg = f"CMC最適化結果: l={l_w:.2f}, c={c_w:.2f}（明度/彩度分布から算出）"
        self.status_var.set(msg)
        self.rgb_log_var.set(msg)

    def reset_cmc_weights(self: "BeadsApp") -> None:
        """CMC(l:c)の重みをデフォルトに戻す。"""
        self.cmc_l_var.set(2.0)
        self.cmc_c_var.set(1.0)
        self._on_cmc_l_change()
        self._on_cmc_c_change()
        self.status_var.set("CMC重みをデフォルト(2.0 / 1.0)に戻しました。")
        self.rgb_log_var.set("CMC重みをデフォルト(2.0 / 1.0)に戻しました。")

    # --- RGB最適化/プレビュー ---
    def compute_optimal_rgb_weights(self: "BeadsApp") -> None:
        """入力画像から単純な平均値ベースでRGB重みを算出する。"""
        if self.input_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        arr = np.asarray(self.input_pil, dtype=np.float32)
        means = arr.mean(axis=(0, 1))
        safe_means = np.clip(means, 1e-3, None)
        target = float(safe_means.mean())
        weights = target / safe_means
        weights = np.clip(weights, 0.5, 1.5)
        r_w, g_w, b_w = (float(weights[0]), float(weights[1]), float(weights[2]))
        self.rgb_r_weight_var.set(r_w)
        self.rgb_g_weight_var.set(g_w)
        self.rgb_b_weight_var.set(b_w)
        self._on_rgb_r_change()
        self._on_rgb_g_change()
        self._on_rgb_b_change()
        msg = f"最適化結果: R={r_w:.2f}, G={g_w:.2f}, B={b_w:.2f}（単純平均合わせ）"
        self.status_var.set(msg)
        self.rgb_log_var.set(msg)

    def reset_rgb_weights(self: "BeadsApp") -> None:
        """RGB重みを1.0に戻し、表示を更新する。"""
        self.rgb_r_weight_var.set(1.0)
        self.rgb_g_weight_var.set(1.0)
        self.rgb_b_weight_var.set(1.0)
        self._on_rgb_r_change()
        self._on_rgb_g_change()
        self._on_rgb_b_change()
        self.rgb_log_var.set("RGB重みを1.0にリセットしました。")
        self.status_var.set("RGB重みを1.0にリセットしました。")
