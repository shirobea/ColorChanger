"""Tkinter UI for beads palette conversion."""

import threading
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import converter
from palette import BeadPalette


@dataclass(frozen=True)
class ConversionRequest:
    """UIから取得した変換パラメータ一式。"""

    width: int
    height: int
    num_colors: int
    mode: str
    quantize_method: str
    keep_aspect: bool
    pipeline: str
    use_saliency: bool
    saliency_strength: float


class BeadsApp:
    """Main application window."""

    def __init__(self, root: tk.Tk, palette: BeadPalette) -> None:
        self.root = root
        self.palette = palette
        # ウィンドウサイズと位置を保持するファイルパス
        self._window_state_path = Path(__file__).resolve().parent / "window_state.json"
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
        self.input_pil: Optional[Image.Image] = None
        self.output_pil: Optional[Image.Image] = None
        self.original_size: Optional[tuple[int, int]] = None
        self.cancel_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        # ステータス表示は使わないので空文字で保持のみ
        self.status_var = tk.StringVar(value="")
        # 初期値は空欄にしておき、画像読込時に解像度を自動反映
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.use_saliency_var = tk.BooleanVar(value=True)
        self.saliency_strength_var = tk.DoubleVar(value=0.9)
        self.saliency_strength_display = tk.StringVar(value="0.90")
        self.quantize_method_var = tk.StringVar(value="Wu減色")
        self._updating_size_fields = False
        self._saliency_min = 0.01
        self._saliency_max = 3.0

        self._build_layout()

    def _build_layout(self) -> None:
        """Create layout and controls."""
        control_frame, preview_frame = self._create_frames()
        self._build_control_panel(control_frame)
        self._build_preview_panel(preview_frame)
        self._finalize_window_layout()

    def _create_frames(self) -> tuple[ttk.Frame, ttk.Frame]:
        """フレーム生成と基本グリッド設定をまとめる。"""
        control_frame = ttk.Frame(self.root, padding=8)
        control_frame.grid(row=0, column=0, sticky="ns")

        preview_frame = ttk.Frame(self.root, padding=8)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        self.preview_frame = preview_frame

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=0)  # 差分テキスト行
        preview_frame.rowconfigure(1, weight=1)  # 画像行
        return control_frame, preview_frame

    def _build_control_panel(self, control_frame: ttk.Frame) -> None:
        """左側の操作パネルを組み立てる。"""
        ttk.Button(control_frame, text="入力画像を選択", command=self.select_image).grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="変換モード").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.mode_var = tk.StringVar(value="Oklab")
        mode_box = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["RGB", "Lab (CIEDE2000)", "Oklab"],
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(control_frame, text="幅(px)").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.Spinbox(control_frame, from_=1, to=2048, textvariable=self.width_var, width=8).grid(
            row=1, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="高さ(px)").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        ttk.Spinbox(control_frame, from_=1, to=2048, textvariable=self.height_var, width=8).grid(
            row=1, column=3, padx=5, pady=5, sticky="w"
        )

        ttk.Checkbutton(
            control_frame, text="比率固定", variable=self.lock_aspect_var, command=self._on_aspect_toggle
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")

        ttk.Button(control_frame, text="1/2", command=self._halve_size).grid(
            row=2, column=1, padx=5, pady=5, sticky="w"
        )

        ttk.Button(control_frame, text="リセット", command=self._reset_size).grid(
            row=2, column=2, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="減色後の色数").grid(row=3, column=1, padx=5, pady=5, sticky="e")
        self.num_colors_var = tk.StringVar(value="64")
        ttk.Spinbox(control_frame, from_=2, to=256, textvariable=self.num_colors_var, width=8).grid(
            row=3, column=2, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="減色方式").grid(row=4, column=1, padx=5, pady=5, sticky="e")
        quantize_box = ttk.Combobox(
            control_frame,
            textvariable=self.quantize_method_var,
            values=["K-means", "Wu減色", "ブロック減色"],
            state="readonly",
            width=18,
        )
        quantize_box.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(control_frame, text="処理順序").grid(row=5, column=1, padx=5, pady=5, sticky="e")
        self.pipeline_var = tk.StringVar(value="リサイズ→減色")
        pipeline_box = ttk.Combobox(
            control_frame,
            textvariable=self.pipeline_var,
            values=["リサイズ→減色", "減色→リサイズ", "ハイブリッド"],
            state="readonly",
            width=18,
        )
        pipeline_box.grid(row=5, column=2, padx=5, pady=5, sticky="w")

        ttk.Checkbutton(
            control_frame,
            text="顔パーツ優先（サリエンシー）",
            variable=self.use_saliency_var,
        ).grid(row=6, column=0, padx=5, pady=5, sticky="w", columnspan=2)

        ttk.Label(control_frame, text="サリエンシー強度").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        strength_scale = ttk.Scale(
            control_frame,
            from_=self._saliency_min,
            to=self._saliency_max,
            orient="horizontal",
            variable=self.saliency_strength_var,
            command=lambda *_: self._on_saliency_strength_change(),
            length=140,
        )
        strength_scale.grid(row=7, column=1, padx=5, pady=5, sticky="we")
        strength_scale.bind("<Button-1>", self._on_saliency_pointer)
        strength_scale.bind("<B1-Motion>", self._on_saliency_pointer)
        self.saliency_scale = strength_scale
        ttk.Label(control_frame, textvariable=self.saliency_strength_display, width=5).grid(
            row=7, column=2, padx=2, pady=5, sticky="w"
        )

        self.convert_button = ttk.Button(control_frame, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        self.progress_label = ttk.Label(control_frame, text="進捗: 0% (経過 0.0s)")
        self.progress_label.grid(row=6, column=3, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(control_frame, length=160)
        self.progress_bar.grid(row=7, column=3, padx=5, pady=5, sticky="w")

        self.save_button = ttk.Button(control_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=8, column=3, padx=5, pady=5, sticky="w")

    def _build_preview_panel(self, preview_frame: ttk.Frame) -> None:
        """右側のプレビュー領域を組み立てる。"""
        self.diff_label = ttk.Label(
            preview_frame,
            textvariable=self.diff_var,
            anchor="e",
            justify="left",
            wraplength=520,
            foreground="#444",
            padding=(3, 0),
        )
        self.diff_label.grid(row=0, column=1, padx=(10, 5), pady=(0, 4), sticky="ne")

        self.input_canvas = ttk.Label(preview_frame, text="入力画像", anchor="center")
        self.input_canvas.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.output_canvas = ttk.Label(preview_frame, text="変換後", anchor="center")
        self.output_canvas.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.output_canvas.bind("<ButtonPress-1>", self._on_output_press)
        self.output_canvas.bind("<ButtonRelease-1>", self._on_output_release)
        self.output_canvas.bind("<Leave>", self._on_output_release)

        self.preview_frame.bind("<Configure>", self._on_preview_resize)

    def _finalize_window_layout(self) -> None:
        """ウィンドウ設定の後処理をまとめる。"""
        self.width_var.trace_add("write", lambda *_: self._on_width_changed())
        self.height_var.trace_add("write", lambda *_: self._on_height_changed())

        self.root.update_idletasks()  # レイアウト計算を反映
        if not self._restored_geometry:
            init_w = self.root.winfo_width()
            init_h = self.root.winfo_height()
            self.root.geometry(f"{init_w}x{init_h}")
        self.root.grid_propagate(False)  # 子ウィジェットの要求で親が拡大しないようにする
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def select_image(self) -> None:
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
        self.original_size = image.size
        self.output_canvas.configure(image="", text="変換後")
        self._set_initial_target_size(image)
        self._refresh_previews()

    def start_conversion(self) -> None:
        """Kick off conversion in a worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            # 二重起動防止
            return

        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return

        request = self._gather_request()
        if request is None:
            return

        # 今回の設定を保持しておき、完了後に差分表示へ利用する
        self._pending_settings = self._build_pending_settings(request)

        self._prepare_conversion_ui()
        self.worker_thread = threading.Thread(
            target=self._run_conversion,
            args=(request, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()

    def _gather_request(self) -> Optional[ConversionRequest]:
        """入力値を検証してConversionRequestにまとめる。"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            num_colors = int(self.num_colors_var.get())
            saliency_strength = float(self.saliency_strength_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "幅・高さ・色数には整数を、強度には数値を入力してください。")
            return None

        if width <= 0 or height <= 0 or num_colors <= 1:
            messagebox.showerror("入力エラー", "幅・高さは1以上、減色後の色数は2以上にしてください。")
            return None

        saliency_strength = max(self._saliency_min, min(self._saliency_max, saliency_strength))
        keep_aspect = self.lock_aspect_var.get()

        mode_label = self.mode_var.get()
        mode = "Lab" if mode_label == "Lab (CIEDE2000)" else mode_label

        quantize_label = self.quantize_method_var.get()
        if quantize_label == "ブロック減色":
            quantize_method = "block"
        elif quantize_label == "Wu減色":
            quantize_method = "wu"
        else:
            quantize_method = "kmeans"

        pipeline_label = self.pipeline_var.get()
        if pipeline_label == "減色→リサイズ":
            pipeline = "quantize_first"
        elif pipeline_label == "ハイブリッド":
            pipeline = "hybrid"
        else:
            pipeline = "resize_first"

        return ConversionRequest(
            width=width,
            height=height,
            num_colors=num_colors,
            mode=mode,
            quantize_method=quantize_method,
            keep_aspect=keep_aspect,
            pipeline=pipeline,
            use_saliency=self.use_saliency_var.get(),
            saliency_strength=saliency_strength,
        )

    def _build_pending_settings(self, request: ConversionRequest) -> dict:
        """設定差分表示用の辞書を生成する。"""
        return {
            "幅": request.width,
            "高さ": request.height,
            "色数": request.num_colors,
            "色空間": self.mode_var.get(),
            "減色方式": self.quantize_method_var.get(),
            "処理順序": self.pipeline_var.get(),
            "比率固定": request.keep_aspect,
            "サリエンシー": request.use_saliency,
            "サリエンシー強度": round(request.saliency_strength, 2),
        }

    def _prepare_conversion_ui(self) -> None:
        """変換開始時のUI状態をまとめて切り替える。"""
        self.convert_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.cancel_event = threading.Event()
        self._start_time = time.perf_counter()
        self.update_progress(0.0)
        self.status_var.set("変換中...")
        self.convert_button.configure(text="変換中止", state="normal", command=self.cancel_conversion)

    def cancel_conversion(self) -> None:
        """ユーザー操作で変換を中断する。"""
        if self.cancel_event:
            self.cancel_event.set()
        self.status_var.set("中止要求を送信しました...")
        self.convert_button.configure(state="disabled", text="停止中...")
        self._start_time = None
        self._reset_progress_display()

    def _run_conversion(
        self,
        request: ConversionRequest,
        cancel_event: Optional[threading.Event],
    ) -> None:
        """Background conversion worker."""
        def progress_cb(value: float) -> None:
            self.root.after(0, self.update_progress, value)

        try:
            result = converter.convert_image(
                input_path=str(self.input_image_path),
                output_size=(request.width, request.height),
                mode=request.mode,
                palette=self.palette,
                num_colors=request.num_colors,
                quantize_method=request.quantize_method,
                keep_aspect=request.keep_aspect,
                pipeline=request.pipeline,
                use_saliency=request.use_saliency,
                saliency_strength=request.saliency_strength,
                progress_callback=progress_cb,
                cancel_event=cancel_event,
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

        # Save and show preview
        out_dir = Path.cwd() / "picture"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{self.input_image_path.stem}_beads.png"
        try:
            Image.fromarray(result).save(output_path)
            self.output_path = output_path
        except Exception as exc:
            self.root.after(
                0,
                lambda: messagebox.showerror("保存失敗", f"出力画像の保存に失敗しました:\n{exc}"),
            )
        # 1つ前の出力と設定を保持して比較に使う
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
        # 直前との差分を常時表示用に更新
        self.diff_var.set(self._build_diff_overlay())

        def on_finish() -> None:
            self._refresh_previews()
            self.update_progress(1.0)
            self._restore_convert_button()
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
            self.cancel_event = None
            self.worker_thread = None
            if self.output_path:
                self.status_var.set("自動保存完了")
            else:
                self.status_var.set("変換完了（保存ボタンで任意の場所に保存できます）")

        self.root.after(0, on_finish)

    def _on_cancelled(self) -> None:
        """キャンセル完了時のUI復帰処理。"""
        self._reset_after_stop("中止しました", clear_canvas=True)

    def _handle_failure(self, status: str) -> None:
        """失敗時の共通後始末。"""
        self._reset_after_stop(status, clear_canvas=False)

    def update_progress(self, value: float) -> None:
        """Update progress UI."""
        clamped = max(0.0, min(1.0, value))
        percent = int(clamped * 100)
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
        self.progress_label.configure(text=f"進捗: {percent}% (経過 {elapsed:.1f}s)")
        self.progress_bar["value"] = percent

    def _reset_progress_display(self) -> None:
        """進捗バーと時間表示を初期化。"""
        self.progress_label.configure(text="進捗: 0% (経過 0.0s)")
        self.progress_bar["value"] = 0

    def _reset_after_stop(self, status: str, clear_canvas: bool) -> None:
        """停止時に共通で状態をリセットする。"""
        self.cancel_event = None
        self.worker_thread = None
        self._start_time = None
        self.output_image = None
        self.output_pil = None
        self.prev_output_pil = None
        self.output_path = None
        self._pending_settings = None
        self._reset_progress_display()
        self._restore_convert_button()
        self.save_button.configure(state="disabled")
        self.diff_var.set("")
        if clear_canvas:
            self.output_canvas.configure(image="", text="変換後")
        self.status_var.set(status)

    def _restore_convert_button(self) -> None:
        """変換ボタンを初期状態に戻す。"""
        self.convert_button.configure(text="変換実行", state="normal", command=self.start_conversion)

    def save_image(self) -> None:
        """Save output image to user-chosen path without modal completion dialog."""
        if self.output_image is None:
            self.status_var.set("出力画像がまだありません。")
            return
        initial_dir = str(Path.cwd() / "picture")
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

    def _on_preview_resize(self, event: tk.Event) -> None:
        """Handle preview area resizing to re-render images."""
        self._refresh_previews()

    # --- ウィンドウ位置・サイズの永続化 ---
    def _on_window_configure(self, event: tk.Event) -> None:
        """ウィンドウが動いたりリサイズされたら現在位置を覚えておく。"""
        if event.widget is not self.root:
            return
        self._last_geometry = (
            self.root.winfo_width(),
            self.root.winfo_height(),
            self.root.winfo_x(),
            self.root.winfo_y(),
        )

    def _load_window_state(self) -> bool:
        """前回終了時のウィンドウ配置を読み込む。"""
        try:
            if not self._window_state_path.exists():
                return False
            data = json.loads(self._window_state_path.read_text(encoding="utf-8"))
            width = int(data.get("width", 0))
            height = int(data.get("height", 0))
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            if width > 0 and height > 0:
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                # 初期値として保持
                self._last_geometry = (width, height, x, y)
                return True
        except Exception:
            # 読み込み失敗時は静かに既定レイアウトに戻す
            return False
        return False

    def _save_window_state(self) -> None:
        """直近に記録したウィンドウ配置をファイルへ保存する。"""
        # まだConfigureイベントが走っていない場合に備えて直接取得しておく
        if self._last_geometry is None:
            self._last_geometry = (
                self.root.winfo_width(),
                self.root.winfo_height(),
                self.root.winfo_x(),
                self.root.winfo_y(),
            )
        width, height, x, y = self._last_geometry
        payload = {
            "width": int(width),
            "height": int(height),
            "x": int(x),
            "y": int(y),
        }
        try:
            self._window_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            # 保存失敗時はアプリ終了を妨げない
            pass

    def _on_close(self) -> None:
        """終了時に位置・サイズを保存してから閉じる。"""
        self._save_window_state()
        self.root.destroy()

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
            self._input_photo = self._resize_to_box(self.input_pil, cell_w, cell_h)
            if self._input_photo:
                self.input_canvas.configure(image=self._input_photo, text="")
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

    def _on_saliency_strength_change(self) -> None:
        """サリエンシー強度スライダーの値を表示用に丸める。"""
        val = float(self.saliency_strength_var.get())
        clamped = max(self._saliency_min, min(self._saliency_max, val))
        snapped = round(clamped, 2)  # 0.01刻み
        if snapped != val:
            self.saliency_strength_var.set(snapped)
        self.saliency_strength_display.set(f"{snapped:.2f}")

    def _on_saliency_pointer(self, event: tk.Event) -> str:
        """クリック・ドラッグ位置に応じて0.01刻みで値を設定する。"""
        scale: ttk.Scale = event.widget  # type: ignore[assignment]
        width = max(1, scale.winfo_width())
        # widget左端を0として相対位置を求める
        fraction = max(0.0, min(1.0, event.x / width))
        new_val = self._saliency_min + fraction * (self._saliency_max - self._saliency_min)
        self.saliency_strength_var.set(round(new_val, 2))
        self._on_saliency_strength_change()
        # 既定動作を抑止し、ジャンプとドラッグを自前で制御する
        return "break"

