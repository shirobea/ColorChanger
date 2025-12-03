"""Application entry point."""

from ui import BeadsApp
import tkinter as tk
from palette import load_palette
from tkinter import messagebox


def main() -> None:
    """Launch the Tkinter application."""
    root = tk.Tk()
    root.title("Beads Dot Converter")

    try:
        palette = load_palette("ColorPallet.csv")
    except Exception as exc:  # pragma: no cover - GUI entry
        messagebox.showerror("パレット読込エラー", f"ColorPallet.csv の読み込みに失敗しました。\n{exc}")
        root.destroy()
        return

    BeadsApp(root, palette)
    root.mainloop()


if __name__ == "__main__":
    main()
