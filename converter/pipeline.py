"""変換パイプラインのオーケストレーター。リサイズ/減色/マッピングを束ね、convert_image を公開する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import threading

import cv2
import numpy as np

from palette import BeadPalette
from .io_utils import _compute_hybrid_size, _compute_resize, _load_image_rgb
from .saliency import (
    ImportanceWeights,
    _compute_saliency_map,
    _normalize_saliency,
    compute_importance_map,
)
from .quantize import _PaletteQuantizer
from .block_methods import (
    _adaptive_block_image,
    _dominant_block_image,
    _map_image_to_palette,
    _run_block_pipeline,
)

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]


@dataclass(frozen=True)
class _PipelineConfig:
    """変換に共通する設定値をまとめて扱う内部用データクラス。"""

    palette: BeadPalette
    mode: str
    num_colors: int
    quantize_method: str
    target_size: Size
    original_size: Size
    progress_callback: ProgressCb | None
    cancel_event: CancelEvent | None


class ConversionCancelled(Exception):
    """ユーザーによる中断を示す例外。"""


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う。"""
    if cancel_event and cancel_event.is_set():
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


def _apply_saliency_enhancement(image_rgb: np.ndarray, saliency: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """重要度の高い領域だけ輪郭を強調する（パレット外の色は使わずに保持）。"""
    h, w = image_rgb.shape[:2]
    saliency_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    saliency_resized = np.clip(saliency_resized, 0.0, 1.0)[..., None]

    img_f = image_rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.2)
    high_freq = img_f - blurred  # 高周波成分

    boost = 0.35 + strength * saliency_resized
    enhanced = img_f + high_freq * boost
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _run_quantize_first_pipeline(
    image_rgb: np.ndarray, quantizer: _PaletteQuantizer, config: _PipelineConfig
) -> np.ndarray:
    """減色→リサイズの順で処理する既存挙動のパス。"""
    quantized = quantizer.quantize_and_map(image_rgb, 0.35, 0.75)
    _report(config.progress_callback, 0.85, config.cancel_event)
    target_w, target_h = config.target_size
    output = cv2.resize(quantized, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_resize_first_pipeline(
    image_rgb: np.ndarray, quantizer: _PaletteQuantizer, config: _PipelineConfig
) -> np.ndarray:
    """リサイズ→減色の順で処理するパス。"""
    target_w, target_h = config.target_size
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    _report(config.progress_callback, 0.15, config.cancel_event)
    output = quantizer.quantize_and_map(resized, 0.45, 0.85)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_hybrid_pipeline(
    image_rgb: np.ndarray, quantizer: _PaletteQuantizer, config: _PipelineConfig
) -> np.ndarray:
    """長辺を軽く縮小してから減色し、最後に目的解像度へ近傍補間するハイブリッドパス。"""
    orig_w, orig_h = config.original_size
    target_w, target_h = config.target_size
    mid_w, mid_h = _compute_hybrid_size((orig_h, orig_w), (target_w, target_h))
    if (mid_w, mid_h) != (orig_w, orig_h):
        image_mid = cv2.resize(image_rgb, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
    else:
        image_mid = image_rgb
    _report(config.progress_callback, 0.12, config.cancel_event)
    quantized = quantizer.quantize_and_map(image_mid, 0.42, 0.82)
    _report(config.progress_callback, 0.9, config.cancel_event)
    output = cv2.resize(quantized, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_map_only_resize_first(
    image_rgb: np.ndarray, config: _PipelineConfig
) -> np.ndarray:
    """リサイズだけ行い、減色せずにパレットへ直接写像するパス。"""
    target_w, target_h = config.target_size
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mapped = _map_image_to_palette(
        resized,
        config.palette,
        config.mode,
        progress_callback=config.progress_callback,
        progress_range=(0.35, 0.95),
        cancel_event=config.cancel_event,
    )
    _report(config.progress_callback, 1.0, config.cancel_event)
    return mapped


def _run_map_only_quantize_first(
    image_rgb: np.ndarray, config: _PipelineConfig
) -> np.ndarray:
    """減色工程を挟まないまま元解像度をパレット写像し、その後リサイズするパス。"""
    mapped = _map_image_to_palette(
        image_rgb,
        config.palette,
        config.mode,
        progress_callback=config.progress_callback,
        progress_range=(0.15, 0.8),
        cancel_event=config.cancel_event,
    )
    target_w, target_h = config.target_size
    output = cv2.resize(mapped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_map_only_hybrid(
    image_rgb: np.ndarray, config: _PipelineConfig
) -> np.ndarray:
    """ハイブリッドサイズでパレット写像し、最終解像度へ近傍補間するパス（減色なし）。"""
    orig_w, orig_h = config.original_size
    target_w, target_h = config.target_size
    mid_w, mid_h = _compute_hybrid_size((orig_h, orig_w), (target_w, target_h))
    if (mid_w, mid_h) != (orig_w, orig_h):
        image_mid = cv2.resize(image_rgb, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
    else:
        image_mid = image_rgb
    mapped = _map_image_to_palette(
        image_mid,
        config.palette,
        config.mode,
        progress_callback=config.progress_callback,
        progress_range=(0.35, 0.9),
        cancel_event=config.cancel_event,
    )
    output = cv2.resize(mapped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def convert_image(
    input_path: str,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    num_colors: int,
    quantize_method: str = "kmeans",
    keep_aspect: bool = True,
    pipeline: str = "resize_first",
    contour_enhance: bool = False,
    eye_importance_scale: float = 0.8,
    adaptive_saliency_weight: float = 0.5,
    fine_scale: int = 2,
    saliency_map: np.ndarray | None = None,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """入力画像をビーズパレットへ変換する外部公開API。"""
    _report(progress_callback, 0.0, cancel_event)
    image_rgb = _load_image_rgb(input_path)
    original_rgb = image_rgb.copy()

    precomputed_saliency = _normalize_saliency(saliency_map) if saliency_map is not None else None

    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = _compute_resize((orig_h, orig_w), output_size, keep_aspect)

    config = _PipelineConfig(
        palette=palette,
        mode=mode,
        num_colors=num_colors,
        quantize_method=quantize_method,
        target_size=(target_w, target_h),
        original_size=(orig_w, orig_h),
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    saliency_for_contour = None
    if contour_enhance:
        saliency_for_contour = precomputed_saliency if precomputed_saliency is not None else _compute_saliency_map(original_rgb)

    pipeline_lower = pipeline.lower()
    if config.quantize_method.lower() == "block":
        return _run_block_pipeline(
            image_rgb,
            config,
            saliency_map=saliency_for_contour,
            contour_enhance=contour_enhance,
        )
    if config.quantize_method.lower() == "adaptive_block":
        importance = None
        if adaptive_saliency_weight > 0:
            importance = compute_importance_map(
                original_rgb, saliency_map=precomputed_saliency, eye_importance_scale=eye_importance_scale
            )
        return _adaptive_block_image(
            image_rgb=image_rgb,
            target_size=config.target_size,
            palette=config.palette,
            mode=config.mode,
            saliency_weight=adaptive_saliency_weight,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            importance_map=importance,
            fine_scale=fine_scale,
            saliency_map=saliency_for_contour,
            contour_enhance=contour_enhance,
        )

    if config.quantize_method.lower() == "none":
        map_only_runner = {
            "quantize_first": _run_map_only_quantize_first,
            "hybrid": _run_map_only_hybrid,
            "resize_first": _run_map_only_resize_first,
        }.get(pipeline_lower, _run_map_only_resize_first)
        return map_only_runner(image_rgb, config)

    quantizer = _PaletteQuantizer(config)
    pipeline_map = {
        "quantize_first": _run_quantize_first_pipeline,
        "hybrid": _run_hybrid_pipeline,
        "resize_first": _run_resize_first_pipeline,
    }
    runner = pipeline_map.get(pipeline_lower, _run_resize_first_pipeline)
    return runner(image_rgb, quantizer, config)
