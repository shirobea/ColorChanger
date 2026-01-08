"""Shared helpers for slider pointer interactions."""

from __future__ import annotations

import tkinter as tk


def calc_scale_value_from_pointer(event: tk.Event) -> float:
    """Calculate scale value from pointer position."""
    scale = event.widget
    width = max(1, scale.winfo_width())
    fraction = max(0.0, min(1.0, event.x / width))
    min_val = float(scale.cget("from"))
    max_val = float(scale.cget("to"))
    return min_val + (max_val - min_val) * fraction
