"""変換パイプライン（リサイズ + パレット写像のみ）。"""

from __future__ import annotations

from typing import Callable, Tuple
import threading

import cv2
import numpy as np

from palette import BeadPalette
from .io_utils import _compute_resize, _load_image_rgb
from .quantize import _map_centers_to_palette, _report

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]


class ConversionCancelled(Exception):
    """ユーザーによる中断を示す例外。"""


def _map_image_to_palette(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lab_metric: str = "CIEDE2000",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """減色せず、各画素を最短距離のパレット色へ写像する。"""
    start, end = progress_range
    _report(progress_callback, start, cancel_event)

    flat = image_rgb.reshape(-1, 3).astype(np.float32)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    mapping = _map_centers_to_palette(
        colors,
        palette,
        mode,
        progress_callback=progress_callback,
        progress_range=(start, end),
        cancel_event=cancel_event,
        cmc_l=cmc_l,
        cmc_c=cmc_c,
        rgb_weights=rgb_weights,
        lab_metric=lab_metric,
    )
    mapped = palette.rgb_array[mapping].astype(np.uint8)[inv].reshape(image_rgb.shape)
    _report(progress_callback, end, cancel_event)
    return mapped


def convert_image(
    input_path: str,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    keep_aspect: bool = True,
    resize_method: str = "nearest",
    lab_metric: str = "CIEDE2000",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """入力画像を指定サイズへリサイズし、パレットへ写像して返す。"""
    _report(progress_callback, 0.0, cancel_event)
    image_rgb = _load_image_rgb(input_path)
    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = _compute_resize((orig_h, orig_w), output_size, keep_aspect)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    interp = interp_map.get(resize_method.lower(), cv2.INTER_NEAREST)

    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=interp)
    _report(progress_callback, 0.3, cancel_event)

    mode_lower = mode.lower()
    if mode_lower in {"none", "なし"}:
        _report(progress_callback, 1.0, cancel_event)
        return resized

    mapped = _map_image_to_palette(
        resized,
        palette,
        mode,
        rgb_weights=rgb_weights,
        lab_metric=lab_metric,
        cmc_l=cmc_l,
        cmc_c=cmc_c,
        progress_callback=progress_callback,
        progress_range=(0.3, 1.0),
        cancel_event=cancel_event,
    )
    _report(progress_callback, 1.0, cancel_event)
    return mapped
