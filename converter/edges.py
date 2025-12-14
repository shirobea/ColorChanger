"""輪郭線強調用のエッジ検出ユーティリティ。"""

from __future__ import annotations

import cv2
import numpy as np

from .saliency import _normalize_saliency


def _auto_canny_thresholds(gray: np.ndarray) -> tuple[int, int]:
    """中央値ベースでCannyの低/高閾値を自動決定する。"""
    med = float(np.median(gray))
    low = int(max(0.0, 0.66 * med))
    high = int(min(255.0, 1.33 * med + 30.0))
    if high <= low:
        high = min(255, low + 30)
    return low, high


def _build_edge_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Sobel+Cannyでエッジマスク(0-1)を生成する。"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    low, high = _auto_canny_thresholds(gray)
    edges = cv2.Canny(gray, low, high)
    edges = edges.astype(np.float32) / 255.0
    # 細線化しつつノイズを丸める
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=0.6)
    return np.clip(edges, 0.0, 1.0)


def apply_edge_enhancement(
    image_rgb: np.ndarray,
    saliency_map: np.ndarray | None = None,
    strength: float = 0.6,
    thickness: float = 0.0,
    saliency_weight: float = 0.5,
    gamma: float = 0.75,
    gain: float = 2.5,
    dilate_iters: int = 1,
) -> np.ndarray:
    """エッジマスクとサリエンシーを掛け合わせて輪郭強調した画像を返す。"""
    h, w = image_rgb.shape[:2]
    img_f = image_rgb.astype(np.float32)
    base_blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.1)
    high_freq = img_f - base_blur

    edges = _build_edge_mask(image_rgb)
    # 太さ調整: thickness(0-1)をカーネルサイズへ変換して膨張
    t = float(np.clip(thickness, 0.0, 1.0))
    if t > 0:
        # 1〜13px程度まで段階を増やす（UI 0-100%に対して約14段階）
        k = 1 + int(round(t * 12.0))
        k = max(1, min(13, k | 1))  # 奇数にする
        kernel = np.ones((k, k), dtype=np.uint8)
        iters = max(1, min(2, dilate_iters))
        edges = cv2.dilate(edges, kernel, iterations=iters)
    weight = edges
    if saliency_map is not None:
        sal = _normalize_saliency(saliency_map)
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        # サリエンシーの下駄を履かせつつミックス（死ににくくする）
        sal_w = float(np.clip(saliency_weight, 0.0, 1.0))
        weight = np.clip(weight * (0.3 + sal_w * sal), 0.0, 1.0)
    # 薄い重みを持ち上げる
    weight = np.power(np.clip(weight, 0.0, 1.0), float(np.clip(gamma, 0.2, 2.5)))
    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=0.8)

    s = float(np.clip(strength, 0.0, 1.0))
    g = float(np.clip(gain, 0.0, 10.0))
    boost = 1.0 + s * g * weight[..., None]
    enhanced = img_f + high_freq * boost
    return np.clip(enhanced, 0, 255).astype(np.uint8)
