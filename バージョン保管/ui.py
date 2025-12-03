"""Tkinter UI for beads palette conversion."""

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import converter
from palette import BeadPalette


class BeadsApp:
    """Main application window."""

    def __init__(self, root: tk.Tk, palette: BeadPalette) -> None:
        self.root = root
        self.palette = palette
        self.input_image_path: Optional[Path] = None
        self.output_image: Optional[np.ndarray] = None
        self.output_path: Optional[Path] = None
        self._input_photo: Optional[ImageTk.PhotoImage] = None
        self._output_photo: Optional[ImageTk.PhotoImage] = None
        self.status_var = tk.StringVar(value="準備完了")

        self._build_layout()

    def _build_layout(self) -> None:
        """Create layout and controls."""
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew")

        preview_frame = ttk.Frame(self.root, padding=10)
        preview_frame.grid(row=1, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)

        # Controls
        ttk.Button(control_frame, text="入力画像を選択", command=self.select_image).grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="変換モード").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.mode_var = tk.StringVar(value="RGB")
        mode_box = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["RGB", "Lab (CIEDE2000)", "Oklab"],
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(control_frame, text="短辺ピクセル数").grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.short_side_var = tk.StringVar(value="64")
        ttk.Spinbox(control_frame, from_=8, to=512, textvariable=self.short_side_var, width=8).grid(
            row=1, column=2, padx=5, pady=5, sticky="w"
        )

        ttk.Label(control_frame, text="減色後の色数").grid(row=2, column=1, padx=5, pady=5, sticky="e")
        self.num_colors_var = tk.StringVar(value="32")
        ttk.Spinbox(control_frame, from_=2, to=256, textvariable=self.num_colors_var, width=8).grid(
            row=2, column=2, padx=5, pady=5, sticky="w"
        )

        self.convert_button = ttk.Button(control_frame, text="変換実行", command=self.start_conversion)
        self.convert_button.grid(row=0, column=3, padx=10, pady=5, sticky="w")

        self.progress_label = ttk.Label(control_frame, text="進捗: 0%")
        self.progress_label.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(control_frame, length=160)
        self.progress_bar.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        self.save_button = ttk.Button(control_frame, text="出力画像を保存", command=self.save_image, state="disabled")
        self.save_button.grid(row=3, column=3, padx=5, pady=5, sticky="w")

        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="#444")
        self.status_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        # Previews
        self.input_canvas = ttk.Label(preview_frame, text="入力画像", anchor="center")
        self.input_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.output_canvas = ttk.Label(preview_frame, text="変換後", anchor="center")
        self.output_canvas.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

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
        self._input_photo = self._image_to_photo(image, max_size=380)
        self.input_canvas.configure(image=self._input_photo, text="")
        self.output_canvas.configure(text="変換後")
        self._output_photo = None

    def start_conversion(self) -> None:
        """Kick off conversion in a worker thread."""
        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return

        try:
            short_side = int(self.short_side_var.get())
            num_colors = int(self.num_colors_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "短辺ピクセル数と減色後の色数には整数を入力してください。")
            return

        if short_side <= 0 or num_colors <= 1:
            messagebox.showerror("入力エラー", "短辺ピクセル数は1以上、減色後の色数は2以上にしてください。")
            return

        mode_label = self.mode_var.get()
        if mode_label == "Lab (CIEDE2000)":
            mode = "Lab"
        else:
            mode = mode_label

        self.convert_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.update_progress(0.0)
        self.status_var.set("変換中...")

        thread = threading.Thread(
            target=self._run_conversion, args=(short_side, num_colors, mode), daemon=True
        )
        thread.start()

    def _run_conversion(self, short_side: int, num_colors: int, mode: str) -> None:
        """Background conversion worker."""
        def progress_cb(value: float) -> None:
            self.root.after(0, self.update_progress, value)

        try:
            result = converter.convert_image(
                input_path=str(self.input_image_path),
                output_size=short_side,
                mode=mode,
                palette=self.palette,
                num_colors=num_colors,
                progress_callback=progress_cb,
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda: messagebox.showerror("変換失敗", f"変換中にエラーが発生しました:\n{exc}"),
            )
            self.root.after(0, lambda: self.convert_button.configure(state="normal"))
            return

        # Save and show preview
        output_path = self.input_image_path.with_name(f"{self.input_image_path.stem}_beads.png")
        try:
            Image.fromarray(result).save(output_path)
            self.output_path = output_path
        except Exception as exc:
            self.root.after(
                0,
                lambda: messagebox.showerror("保存失敗", f"出力画像の保存に失敗しました:\n{exc}"),
            )
        self.output_image = result
        preview = Image.fromarray(result)
        self._output_photo = self._image_to_photo(preview, max_size=380)

        def on_finish() -> None:
            if self._output_photo:
                self.output_canvas.configure(image=self._output_photo, text="")
            self.update_progress(1.0)
            self.convert_button.configure(state="normal")
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
            if self.output_path:
                self.status_var.set(f"自動保存: {self.output_path}")
            else:
                self.status_var.set("変換完了（保存ボタンで任意の場所に保存できます）")

        self.root.after(0, on_finish)

    def update_progress(self, value: float) -> None:
        """Update progress UI."""
        clamped = max(0.0, min(1.0, value))
        percent = int(clamped * 100)
        self.progress_label.configure(text=f"進捗: {percent}%")
        self.progress_bar["value"] = percent

    def save_image(self) -> None:
        """Save output image to user-chosen path without modal completion dialog."""
        if self.output_image is None:
            self.status_var.set("出力画像がまだありません。")
            return
        initial_dir = str(self.input_image_path.parent) if self.input_image_path else "."
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
            self.status_var.set(f"保存しました: {path}")
        except Exception as exc:
            messagebox.showerror("保存失敗", f"出力画像の保存に失敗しました:\n{exc}")

    def _image_to_photo(self, image: Image.Image, max_size: int = 400) -> ImageTk.PhotoImage:
        """Resize image for preview while keeping aspect ratio."""
        w, h = image.size
        scale = min(max_size / max(w, h), 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)
