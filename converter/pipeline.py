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

ALL_MODE_SPECS = [
    {"label": "なし", "mode": "none"},
    {"label": "RGB", "mode": "RGB"},
    {"label": "Lab2000", "mode": "Lab", "lab_metric": "CIEDE2000"},
    {"label": "Lab94", "mode": "Lab", "lab_metric": "CIE94"},
    {"label": "Lab76", "mode": "Lab", "lab_metric": "CIE76"},
    {"label": "Hunter", "mode": "Hunter"},
    {"label": "Oklab", "mode": "Oklab"},
    {"label": "CMC", "mode": "CMC(l:c)"},
]


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
    input_path: str | None,
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
    input_image: np.ndarray | None = None,
) -> np.ndarray:
    """入力画像を指定サイズへリサイズし、パレットへ写像して返す。"""
    _report(progress_callback, 0.0, cancel_event)
    if input_image is not None:
        # 事前ノイズ除去などで渡されたRGB配列を優先する
        image_rgb = np.asarray(input_image, dtype=np.uint8)
    else:
        if input_path is None:
            raise ValueError("input_image または input_path を指定してください。")
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


def convert_all_modes(
    input_path: str | None,
    output_size: int | Tuple[int, int],
    palette: BeadPalette,
    keep_aspect: bool = True,
    resize_method: str = "nearest",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    input_image: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    """全ての変換モードで処理した結果を順番に返す。"""
    _report(progress_callback, 0.0, cancel_event)
    if input_image is not None:
        # 事前ノイズ除去などで渡されたRGB配列を優先する
        image_rgb = np.asarray(input_image, dtype=np.uint8)
    else:
        if input_path is None:
            raise ValueError("input_image または input_path を指定してください。")
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
    _report(progress_callback, 0.2, cancel_event)

    results: list[dict[str, np.ndarray]] = []
    total = len(ALL_MODE_SPECS)
    span = 0.8 / max(1, total)
    for idx, spec in enumerate(ALL_MODE_SPECS):
        start = 0.2 + span * idx
        end = start + span
        mode = str(spec.get("mode", ""))
        label = str(spec.get("label", ""))
        if mode.lower() in {"none", "なし"}:
            # 変換なしはリサイズ結果をそのまま使う
            _report(progress_callback, start, cancel_event)
            results.append({"label": label, "image": resized.copy()})
            _report(progress_callback, end, cancel_event)
            continue
        mapped = _map_image_to_palette(
            resized,
            palette,
            mode,
            rgb_weights=rgb_weights,
            lab_metric=str(spec.get("lab_metric", "CIEDE2000")),
            cmc_l=cmc_l,
            cmc_c=cmc_c,
            progress_callback=progress_callback,
            progress_range=(start, end),
            cancel_event=cancel_event,
        )
        results.append({"label": label, "image": mapped})

    _report(progress_callback, 1.0, cancel_event)
    return results
