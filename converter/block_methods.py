"""ブロック分割系のパイプラインとマッピング処理をまとめたモジュール。"""

from __future__ import annotations

from typing import Tuple, Callable, TYPE_CHECKING
import threading
import numpy as np
import cv2

from palette import BeadPalette
from .saliency import compute_importance_map
from .quantize import _map_centers_to_palette

if TYPE_CHECKING:
    from . import _PipelineConfig

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う（循環回避のためここで定義）。"""
    if cancel_event and cancel_event.is_set():
        from . import ConversionCancelled  # 遅延インポート
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


def _map_image_to_palette(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.4, 0.9),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """減色を行わずに元色をそのままパレットへ最短距離で写像する。"""
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
    )
    mapped = palette.rgb_array[mapping].astype(np.uint8)[inv].reshape(image_rgb.shape)
    _report(progress_callback, end, cancel_event)
    return mapped


def _run_block_pipeline(
    image_rgb: np.ndarray,
    config: _PipelineConfig,
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
) -> np.ndarray:
    """ブロック単位の最多色をとる減色パス。色数指定は無視する。"""
    return _dominant_block_image(
        image_rgb=image_rgb,
        target_size=config.target_size,
        palette=config.palette,
        mode=config.mode,
        progress_callback=config.progress_callback,
        cancel_event=config.cancel_event,
        saliency_map=saliency_map,
        contour_enhance=contour_enhance,
    )


def _dominant_block_image(
    image_rgb: np.ndarray,
    target_size: Size,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
) -> np.ndarray:
    """指定サイズのグリッドに分け、各ブロックの最多色で塗りつぶした後にパレットへ写像する。"""

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("幅・高さは1以上にしてください。")

    orig_h, orig_w = image_rgb.shape[:2]
    work_w = max(orig_w, target_w)
    work_h = max(orig_h, target_h)
    if (work_w, work_h) != (orig_w, orig_h):
        working = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    else:
        working = image_rgb
    saliency_work: np.ndarray | None = None
    if contour_enhance and saliency_map is not None:
        saliency_work = cv2.resize(np.clip(saliency_map.astype(np.float32), 0.0, 1.0), (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    x_edges = np.linspace(0, work_w, target_w + 1, dtype=int)
    y_edges = np.linspace(0, work_h, target_h + 1, dtype=int)

    dominant = np.empty((target_h, target_w, 3), dtype=np.uint8)
    total_blocks = target_w * target_h
    processed = 0

    for yi in range(target_h):
        y0, y1 = y_edges[yi], y_edges[yi + 1]
        row_slice = working[y0:y1, :, :]
        for xi in range(target_w):
            x0, x1 = x_edges[xi], x_edges[xi + 1]
            block = row_slice[:, x0:x1, :]
            if block.size == 0:
                fallback_px = working[min(y0, work_h - 1), min(x0, work_w - 1)]
                dominant_color = fallback_px
            else:
                flat = block.reshape(-1, 3)
                if contour_enhance and saliency_work is not None:
                    sal_block = saliency_work[y0:y1, x0:x1]
                    sal_flat = sal_block.reshape(-1).astype(np.float32)
                    if sal_flat.size and sal_flat.max() > 0:
                        sal_flat = sal_flat / sal_flat.max()
                    colors, inv = np.unique(flat, axis=0, return_inverse=True)
                    weights = np.zeros(len(colors), dtype=np.float32)
                    np.add.at(weights, inv, sal_flat if sal_flat.size == len(inv) else np.ones(len(inv), dtype=np.float32))
                    dominant_color = colors[weights.argmax()]
                else:
                    values, counts = np.unique(flat, axis=0, return_counts=True)
                    dominant_color = values[counts.argmax()]
            dominant[yi, xi] = dominant_color
            processed += 1

        if total_blocks:
            _report(progress_callback, 0.8 * processed / total_blocks, cancel_event)

    centers = dominant.reshape(-1, 3).astype(np.float32)
    mapping = _map_centers_to_palette(
        centers,
        palette,
        mode,
        progress_callback=progress_callback,
        progress_range=(0.8, 0.98),
        cancel_event=cancel_event,
    )
    mapped = palette.rgb_array[mapping].astype(np.uint8).reshape(target_h, target_w, 3)

    _report(progress_callback, 1.0, cancel_event)

    return mapped


def _adaptive_block_image(
    image_rgb: np.ndarray,
    target_size: Size,
    palette: BeadPalette,
    mode: str,
    saliency_weight: float,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    importance_map: np.ndarray | None = None,
    fine_scale: int = 2,
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
) -> np.ndarray:
    """重要度マップを用いて「細かい」or「通常」の2段階でブロック代表色を求める。"""

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("幅・高さは1以上にしてください。")

    if saliency_weight <= 0:
        return _dominant_block_image(
            image_rgb=image_rgb,
            target_size=target_size,
            palette=palette,
            mode=mode,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    fine_scale = max(1, int(fine_scale))

    orig_h, orig_w = image_rgb.shape[:2]
    work_w = max(orig_w, target_w)
    work_h = max(orig_h, target_h)
    if (work_w, work_h) != (orig_w, orig_h):
        working = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    else:
        working = image_rgb

    if importance_map is None:
        importance_map = compute_importance_map(image_rgb)
    importance_resized = cv2.resize(importance_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    importance_blurred = cv2.GaussianBlur(importance_resized, (0, 0), sigmaX=1.2)
    importance_blurred = np.clip(importance_blurred.astype(np.float32), 0.0, 1.0)
    saliency_work: np.ndarray | None = None
    if contour_enhance and saliency_map is not None:
        saliency_work = cv2.resize(np.clip(saliency_map.astype(np.float32), 0.0, 1.0), (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    weight = float(np.clip(saliency_weight, 0.0, 1.0))
    base_threshold = 0.50
    threshold = float(np.clip(base_threshold - weight * 0.25, 0.18, 0.72))
    fine_mask = (importance_blurred >= threshold).astype(np.uint8)

    coarse_mask = 1 - fine_mask
    fine_scale = max(1, int(fine_scale))

    fine_w = max(1, work_w // fine_scale)
    fine_h = max(1, work_h // fine_scale)
    fine_x_edges = np.linspace(0, work_w, fine_w + 1, dtype=int)
    fine_y_edges = np.linspace(0, work_h, fine_h + 1, dtype=int)
    x_edges = np.linspace(0, work_w, target_w + 1, dtype=int)
    y_edges = np.linspace(0, work_h, target_h + 1, dtype=int)

    working = cv2.resize(working, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    if saliency_work is not None:
        saliency_work = cv2.resize(saliency_work, (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    total_blocks = target_w * target_h
    processed = 0

    for yi in range(target_h):
        y0, y1 = y_edges[yi], y_edges[yi + 1]
        norm_h = max(1, y1 - y0)
        for xi in range(target_w):
            x0, x1 = x_edges[xi], x_edges[xi + 1]
            norm_w = max(1, x1 - x0)

            if fine_mask[yi, xi]:
                fine_w_blk = max(1, norm_w // fine_scale)
                fine_h_blk = max(1, norm_h // fine_scale)
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                fx0 = max(0, cx - fine_w_blk // 2)
                fx1 = min(work_w, fx0 + fine_w_blk)
                fy0 = max(0, cy - fine_h_blk // 2)
                fy1 = min(work_h, fy0 + fine_h_blk)
                sx0, sx1, sy0, sy1 = fx0, fx1, fy0, fy1
            else:
                sx0, sx1, sy0, sy1 = x0, x1, y0, y1

            block = working[sy0:sy1, sx0:sx1, :]

            if block.size == 0:
                fallback_px = working[min(sy0, work_h - 1), min(sx0, work_w - 1)]
                dominant_color = fallback_px
            else:
                flat = block.reshape(-1, 3)
                if contour_enhance and saliency_work is not None:
                    sal_block = saliency_work[sy0:sy1, sx0:sx1]
                    sal_flat = sal_block.reshape(-1).astype(np.float32)
                    if sal_flat.size and sal_flat.max() > 0:
                        sal_flat = sal_flat / sal_flat.max()
                    colors, inv = np.unique(flat, axis=0, return_inverse=True)
                    weights = np.zeros(len(colors), dtype=np.float32)
                    np.add.at(weights, inv, sal_flat if sal_flat.size == len(inv) else np.ones(len(inv), dtype=np.float32))
                    dominant_color = colors[weights.argmax()]
                else:
                    values, counts = np.unique(flat, axis=0, return_counts=True)
                    dominant_color = values[counts.argmax()]

            output[yi, xi] = dominant_color
            processed += 1
            if total_blocks:
                _report(progress_callback, 0.1 + 0.65 * processed / total_blocks, cancel_event)

    centers = output.reshape(-1, 3).astype(np.float32)
    mapping = _map_centers_to_palette(
        centers,
        palette,
        mode,
        progress_callback=progress_callback,
        progress_range=(0.8, 0.98),
        cancel_event=cancel_event,
    )
    mapped = palette.rgb_array[mapping].astype(np.uint8).reshape(target_h, target_w, 3)
    _report(progress_callback, 1.0, cancel_event)
    return mapped
