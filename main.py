"""Application entry point."""

from ui import BeadsApp
import tkinter as tk
from palette import load_palette
from tkinter import messagebox
from pathlib import Path
import sys


def resolve_palette_path(filename: str = "ColorPallet.csv") -> Path:
    """EXE同梱/外部配置どちらでもパレットを探す。"""
    # 実行環境ごとの配置揺れを吸収する
    candidates = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
    ]
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        candidates.append(base_dir / filename)
        candidates.append(Path(getattr(sys, "_MEIPASS", base_dir)) / filename)

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} が見つかりません。")


def main() -> None:
    """Launch the Tkinter application."""
    root = tk.Tk()
    root.title("Beads Dot Converter")

    try:
        palette_path = resolve_palette_path()
        palette = load_palette(palette_path)
    except Exception as exc:  # pragma: no cover - GUI entry
        messagebox.showerror("パレット読込エラー", f"ColorPallet.csv の読み込みに失敗しました。\n{exc}")
        root.destroy()
        return

    BeadsApp(root, palette)
    root.mainloop()


if __name__ == "__main__":
    main()
