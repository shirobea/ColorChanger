"""変換パイプラインのオーケストレーター。リサイズ/減色/マッピングを束ね、convert_image を公開する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import threading

import cv2
import numpy as np

from palette import BeadPalette
from .io_utils import _compute_hybrid_size, _compute_resize, _load_image_rgb
from .edges import apply_edge_enhancement
from .saliency import (
    _normalize_saliency,
    compute_importance_map,
)
from .quantize import _quantize, _quantize_wu, _map_centers_to_palette
from .block_methods import _adaptive_block_image, _dominant_block_image, _map_image_to_palette

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]


@dataclass(frozen=True)
class _PipelineConfig:
    """変換に共通する設定値をまとめて扱う内部用データクラス。"""

    palette: BeadPalette
    mode: str
    rgb_weights: tuple[float, float, float]
    cmc_l: float
    cmc_c: float
    num_colors: int
    quantize_method: str
    target_size: Size
    original_size: Size
    hybrid_scale_percent: float
    progress_callback: ProgressCb | None
    cancel_event: CancelEvent | None
    resize_interp: int


class ConversionCancelled(Exception):
    """ユーザーによる中断を示す例外。"""


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う。"""
    if cancel_event and cancel_event.is_set():
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


def _run_quantize_first_pipeline(
    image_rgb: np.ndarray, quantizer: _PaletteQuantizer, config: _PipelineConfig
) -> np.ndarray:
    """減色→リサイズの順で処理する既存挙動のパス。"""
    quantized = quantizer.quantize_and_map(image_rgb, 0.35, 0.75)
    _report(config.progress_callback, 0.85, config.cancel_event)
    target_w, target_h = config.target_size
    output = cv2.resize(quantized, (target_w, target_h), interpolation=config.resize_interp)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_resize_first_pipeline(
    image_rgb: np.ndarray, quantizer: _PaletteQuantizer, config: _PipelineConfig
) -> np.ndarray:
    """リサイズ→減色の順で処理するパス。"""
    target_w, target_h = config.target_size
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=config.resize_interp)
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
    mid_w, mid_h = _compute_hybrid_size(
        (orig_h, orig_w),
        (target_w, target_h),
        scale_percent=config.hybrid_scale_percent,
    )
    if (mid_w, mid_h) != (orig_w, orig_h):
        image_mid = cv2.resize(image_rgb, (mid_w, mid_h), interpolation=config.resize_interp)
    else:
        image_mid = image_rgb
    _report(config.progress_callback, 0.12, config.cancel_event)
    quantized = quantizer.quantize_and_map(image_mid, 0.42, 0.82)
    _report(config.progress_callback, 0.9, config.cancel_event)
    output = cv2.resize(quantized, (target_w, target_h), interpolation=config.resize_interp)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_map_only_resize_first(
    image_rgb: np.ndarray, config: _PipelineConfig
) -> np.ndarray:
    """リサイズだけ行い、減色せずにパレットへ直接写像するパス。"""
    target_w, target_h = config.target_size
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=config.resize_interp)
    mapped = _map_image_to_palette(
        resized,
        config.palette,
        config.mode,
        rgb_weights=config.rgb_weights,
        cmc_l=config.cmc_l,
        cmc_c=config.cmc_c,
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
        rgb_weights=config.rgb_weights,
        cmc_l=config.cmc_l,
        cmc_c=config.cmc_c,
        progress_callback=config.progress_callback,
        progress_range=(0.15, 0.8),
        cancel_event=config.cancel_event,
    )
    target_w, target_h = config.target_size
    output = cv2.resize(mapped, (target_w, target_h), interpolation=config.resize_interp)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _run_map_only_hybrid(
    image_rgb: np.ndarray, config: _PipelineConfig
) -> np.ndarray:
    """ハイブリッドサイズでパレット写像し、最終解像度へ近傍補間するパス（減色なし）。"""
    orig_w, orig_h = config.original_size
    target_w, target_h = config.target_size
    mid_w, mid_h = _compute_hybrid_size(
        (orig_h, orig_w),
        (target_w, target_h),
        scale_percent=config.hybrid_scale_percent,
    )
    if (mid_w, mid_h) != (orig_w, orig_h):
        image_mid = cv2.resize(image_rgb, (mid_w, mid_h), interpolation=config.resize_interp)
    else:
        image_mid = image_rgb
    mapped = _map_image_to_palette(
        image_mid,
        config.palette,
        config.mode,
        rgb_weights=config.rgb_weights,
        cmc_l=config.cmc_l,
        cmc_c=config.cmc_c,
        progress_callback=config.progress_callback,
        progress_range=(0.35, 0.9),
        cancel_event=config.cancel_event,
    )
    output = cv2.resize(mapped, (target_w, target_h), interpolation=config.resize_interp)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return output


def _quantize_without_palette(
    image_rgb: np.ndarray,
    k: int,
    method: str,
    progress_callback: ProgressCb | None,
    progress_range: tuple[float, float],
    cancel_event: CancelEvent | None,
) -> np.ndarray:
    """パレット写像を行わず、量子化中心色だけで画像を再構成する。"""
    start, end = progress_range
    _report(progress_callback, start, cancel_event)
    if method == "wu":
        centers, labels = _quantize_wu(image_rgb, k)
    else:
        centers, labels = _quantize(image_rgb, k)
    out = centers[labels].reshape(image_rgb.shape).astype(np.uint8)
    _report(progress_callback, end, cancel_event)
    return out


def convert_image(
    input_path: str,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    num_colors: int,
    quantize_method: str = "kmeans",
    keep_aspect: bool = True,
    pipeline: str = "resize_first",
    edge_enhance: bool = False,
    edge_strength: float = 0.6,
    edge_thickness: float = 0.0,
    edge_gain: float = 2.5,
    edge_gamma: float = 0.75,
    edge_saliency_weight: float = 0.5,
    eye_importance_scale: float = 0.8,
    adaptive_saliency_weight: float = 0.5,
    fine_scale: int = 2,
    saliency_map: np.ndarray | None = None,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    hybrid_scale_percent: float = 100.0,
    resize_method: str = "nearest",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """入力画像をビーズパレットへ変換する外部公開API。"""
    _report(progress_callback, 0.0, cancel_event)
    image_rgb = _load_image_rgb(input_path)
    original_rgb = image_rgb.copy()

    precomputed_saliency = _normalize_saliency(saliency_map) if saliency_map is not None else None

    if edge_enhance:
        strength = max(0.0, min(1.0, float(edge_strength)))
        thickness = max(0.0, min(1.0, float(edge_thickness)))
        gain = max(0.0, min(10.0, float(edge_gain)))
        gamma = float(edge_gamma)
        sal_w = max(0.0, min(1.0, float(edge_saliency_weight)))
        image_rgb = apply_edge_enhancement(
            image_rgb,
            saliency_map=precomputed_saliency,
            strength=strength,
            thickness=thickness,
            gain=gain,
            gamma=gamma,
            saliency_weight=sal_w,
        )

    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = _compute_resize((orig_h, orig_w), output_size, keep_aspect)

    # ユーザー指定の補間方式をOpenCV定数へマップ（不正値は最近傍へフォールバック）
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    resize_interp = interp_map.get(resize_method.lower(), cv2.INTER_NEAREST)
    mode_lower = mode.lower()
    use_palette = mode_lower not in {"なし", "none"}

    config = _PipelineConfig(
        palette=palette,
        mode=mode,
        rgb_weights=tuple(rgb_weights),
        cmc_l=float(cmc_l),
        cmc_c=float(cmc_c),
        num_colors=num_colors,
        quantize_method=quantize_method,
        target_size=(target_w, target_h),
        original_size=(orig_w, orig_h),
        hybrid_scale_percent=hybrid_scale_percent,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        resize_interp=resize_interp,
    )

    pipeline_lower = pipeline.lower()
    resize_method_lower = resize_method.lower()
    q_method = config.quantize_method.lower()

    def _scaled_cb(start: float, end: float) -> ProgressCb | None:
        """進捗を任意区間へスケーリングしたコールバックを生成。"""
        if config.progress_callback is None:
            return None
        span = max(0.0, end - start)

        def _cb(v: float) -> None:
            clamped = max(0.0, min(1.0, v))
            config.progress_callback(start + span * clamped)

        return _cb

    def _resize_with_method(img: np.ndarray, size: Size, start: float, end: float) -> np.ndarray:
        """リサイズ方式に応じた処理を行い、進捗を指定区間で更新する。"""
        scaled = _scaled_cb(start, end)
        if resize_method_lower in {"nearest", "bilinear", "bicubic"}:
            _report(config.progress_callback, start, config.cancel_event)
            interp = config.resize_interp
            resized = cv2.resize(img, size, interpolation=interp)
            _report(config.progress_callback, end, config.cancel_event)
            return resized

        if resize_method_lower == "block":
            return _dominant_block_image(
                image_rgb=img,
                target_size=size,
                palette=config.palette,
                mode=config.mode,
                rgb_weights=config.rgb_weights,
                cmc_l=config.cmc_l,
                cmc_c=config.cmc_c,
                progress_callback=scaled,
                cancel_event=config.cancel_event,
                use_palette=use_palette,
            )

        if resize_method_lower == "adaptive_block":
            importance = None
            if adaptive_saliency_weight > 0:
                importance = compute_importance_map(
                    original_rgb, saliency_map=precomputed_saliency, eye_importance_scale=eye_importance_scale
                )
            return _adaptive_block_image(
                image_rgb=img,
                target_size=size,
                palette=config.palette,
                mode=config.mode,
                rgb_weights=config.rgb_weights,
                cmc_l=config.cmc_l,
                cmc_c=config.cmc_c,
                saliency_weight=adaptive_saliency_weight,
                progress_callback=scaled,
                cancel_event=config.cancel_event,
                importance_map=importance,
                fine_scale=fine_scale,
                use_palette=use_palette,
            )

        # フォールバック: 最近傍
        _report(config.progress_callback, start, config.cancel_event)
        resized = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        _report(config.progress_callback, end, config.cancel_event)
        return resized

    def _quantize_palette(img: np.ndarray, start: float, end: float) -> np.ndarray:
        """パレットを用いた減色を進捗範囲付きで行う。"""
        scaled = _scaled_cb(start, end)
        _report(config.progress_callback, start, config.cancel_event)
        if q_method == "wu":
            centers, labels = _quantize_wu(img, config.num_colors)
        else:
            centers, labels = _quantize(img, config.num_colors)
        mapping = _map_centers_to_palette(
            centers,
            config.palette,
            config.mode,
            progress_callback=scaled,
            progress_range=(0.1, 0.9),
            cancel_event=config.cancel_event,
            cmc_l=config.cmc_l,
            cmc_c=config.cmc_c,
        )
        mapped_colors = config.palette.rgb_array[mapping].astype(np.uint8)
        out = mapped_colors[labels].reshape(img.shape).astype(np.uint8)
        _report(config.progress_callback, end, config.cancel_event)
        return out

    def _quantize_no_palette(img: np.ndarray, start: float, end: float) -> np.ndarray:
        """パレットを使わず量子化中心色で再構成。"""
        scaled = _scaled_cb(start, end)
        return _quantize_without_palette(
            img,
            config.num_colors,
            q_method,
            progress_callback=scaled,
            progress_range=(0.0, 1.0),
            cancel_event=config.cancel_event,
        )

    # --- パレットを使わないモード（モード=なし） ---
    if not use_palette:
        if q_method == "none" and resize_method_lower in {"block", "adaptive_block"}:
            return _resize_with_method(image_rgb, config.target_size, 0.0, 1.0)

        def _quantize_stage(img: np.ndarray, start: float, end: float) -> np.ndarray:
            if q_method == "none":
                _report(config.progress_callback, end, config.cancel_event)
                return img
            return _quantize_no_palette(img, start, end)

        if pipeline_lower == "quantize_first":
            quantized = _quantize_stage(image_rgb, 0.0, 0.6)
            output = _resize_with_method(quantized, config.target_size, 0.6, 1.0)
            _report(config.progress_callback, 1.0, config.cancel_event)
            return output

        if pipeline_lower == "hybrid":
            mid_w, mid_h = _compute_hybrid_size((orig_h, orig_w), (target_w, target_h), scale_percent=config.hybrid_scale_percent)
            if (mid_w, mid_h) != (orig_w, orig_h):
                mid_img = _resize_with_method(image_rgb, (mid_w, mid_h), 0.0, 0.3)
            else:
                mid_img = image_rgb
            quantized = _quantize_stage(mid_img, 0.3, 0.7)
            if (mid_w, mid_h) != (target_w, target_h):
                quantized = _resize_with_method(quantized, (target_w, target_h), 0.7, 1.0)
            _report(config.progress_callback, 1.0, config.cancel_event)
            return quantized

        # デフォルト: リサイズ→量子化
        resized = _resize_with_method(image_rgb, config.target_size, 0.0, 0.45)
        quantized = _quantize_stage(resized, 0.45, 1.0)
        _report(config.progress_callback, 1.0, config.cancel_event)
        return quantized

    # --- 既存のパレット使用パス ---
    if q_method == "none" and resize_method_lower in {"block", "adaptive_block"}:
        return _resize_with_method(image_rgb, config.target_size, 0.0, 1.0)

    def _quantize_stage_palette(img: np.ndarray, start: float, end: float) -> np.ndarray:
        if q_method == "none":
            # 減色を省略する場合でもパレット写像は行う（モード変更の効果を反映させる）
            scaled = _scaled_cb(start, end)
            return _map_image_to_palette(
                img,
                config.palette,
                config.mode,
                rgb_weights=config.rgb_weights,
                cmc_l=config.cmc_l,
                cmc_c=config.cmc_c,
                progress_callback=scaled,
                progress_range=(0.0, 1.0),
                cancel_event=config.cancel_event,
            )
        return _quantize_palette(img, start, end)

    if pipeline_lower == "quantize_first":
        quantized = _quantize_stage_palette(image_rgb, 0.0, 0.6)
        output = _resize_with_method(quantized, config.target_size, 0.6, 1.0)
        _report(config.progress_callback, 1.0, config.cancel_event)
        return output

    if pipeline_lower == "hybrid":
        mid_w, mid_h = _compute_hybrid_size((orig_h, orig_w), (target_w, target_h), scale_percent=config.hybrid_scale_percent)
        if (mid_w, mid_h) != (orig_w, orig_h):
            mid_img = _resize_with_method(image_rgb, (mid_w, mid_h), 0.0, 0.3)
        else:
            mid_img = image_rgb
        quantized = _quantize_stage_palette(mid_img, 0.3, 0.7)
        if (mid_w, mid_h) != (target_w, target_h):
            quantized = _resize_with_method(quantized, (target_w, target_h), 0.7, 1.0)
        _report(config.progress_callback, 1.0, config.cancel_event)
        return quantized

    # デフォルト: リサイズ→減色
    resized = _resize_with_method(image_rgb, config.target_size, 0.0, 0.45)
    quantized = _quantize_stage_palette(resized, 0.45, 1.0)
    _report(config.progress_callback, 1.0, config.cancel_event)
    return quantized
