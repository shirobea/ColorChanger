"""Color space utilities for RGB, Lab, and Oklab."""

from __future__ import annotations

import numpy as np
import cv2


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Lab (L in 0-100)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    flat = arr.reshape(-1, 3)
    # OpenCV expects BGR; convert and scale to 0-1 for correct range.
    bgr = flat[:, ::-1] / 255.0
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return lab.reshape(original_shape)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-1) to linear RGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to XYZ (D65, scaled to 0-100)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    rgb_norm = arr.reshape(-1, 3) / 255.0
    linear = srgb_to_linear(rgb_norm)

    xyz = np.empty_like(linear)
    # sRGB(D65) -> XYZ の標準行列
    xyz[:, 0] = 0.4124564 * linear[:, 0] + 0.3575761 * linear[:, 1] + 0.1804375 * linear[:, 2]
    xyz[:, 1] = 0.2126729 * linear[:, 0] + 0.7151522 * linear[:, 1] + 0.0721750 * linear[:, 2]
    xyz[:, 2] = 0.0193339 * linear[:, 0] + 0.1191920 * linear[:, 1] + 0.9503041 * linear[:, 2]
    xyz *= 100.0
    return xyz.reshape(original_shape)


def rgb_to_hunter_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Hunter Lab (D65)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    xyz = rgb_to_xyz(arr).reshape(-1, 3)

    x = xyz[:, 0] / 95.047
    y = xyz[:, 1] / 100.0
    z = xyz[:, 2] / 108.883
    # D65白色点で正規化してHunter Labへ変換
    sqrt_y = np.sqrt(np.maximum(y, 0.0))
    denom = np.maximum(sqrt_y, 1e-6)
    l_val = 100.0 * sqrt_y
    a_val = 175.0 * (x - y) / denom
    b_val = 70.0 * (y - z) / denom

    hunter = np.stack([l_val, a_val, b_val], axis=1)
    return hunter.reshape(original_shape)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Oklab."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    rgb_norm = arr.reshape(-1, 3) / 255.0
    linear = srgb_to_linear(rgb_norm)

    lms = np.empty_like(linear)
    lms[:, 0] = 0.4122214708 * linear[:, 0] + 0.5363325363 * linear[:, 1] + 0.0514459929 * linear[:, 2]
    lms[:, 1] = 0.2119034982 * linear[:, 0] + 0.6806995451 * linear[:, 1] + 0.1073969566 * linear[:, 2]
    lms[:, 2] = 0.0883024619 * linear[:, 0] + 0.2817188376 * linear[:, 1] + 0.6299787005 * linear[:, 2]

    lms_cbrt = np.cbrt(lms)

    oklab = np.empty_like(linear)
    oklab[:, 0] = 0.2104542553 * lms_cbrt[:, 0] + 0.7936177850 * lms_cbrt[:, 1] - 0.0040720468 * lms_cbrt[:, 2]
    oklab[:, 1] = 1.9779984951 * lms_cbrt[:, 0] - 2.4285922050 * lms_cbrt[:, 1] + 0.4505937099 * lms_cbrt[:, 2]
    oklab[:, 2] = 0.0259040371 * lms_cbrt[:, 0] + 0.7827717662 * lms_cbrt[:, 1] - 0.8086757660 * lms_cbrt[:, 2]

    return oklab.reshape(original_shape)


def cmc_delta_e(lab_sample: np.ndarray, lab_array: np.ndarray, l_weight: float = 2.0, c_weight: float = 1.0) -> np.ndarray:
    """Vectorized CMC(l:c) between one sample and many targets."""
    L1, a1, b1 = lab_sample.astype(np.float64)
    L2 = lab_array[:, 0].astype(np.float64)
    a2 = lab_array[:, 1].astype(np.float64)
    b2 = lab_array[:, 2].astype(np.float64)

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    deltaL = L1 - L2
    deltaC = C1 - C2
    deltaa = a1 - a2
    deltab = b1 - b2
    deltaH_sq = np.maximum(0.0, deltaa ** 2 + deltab ** 2 - deltaC ** 2)

    H1 = np.degrees(np.arctan2(b1, a1))
    if H1 < 0:
        H1 += 360.0
    if 164.0 <= H1 <= 345.0:
        T = 0.56 + abs(0.2 * np.cos(np.radians(H1 + 168.0)))
    else:
        T = 0.36 + abs(0.4 * np.cos(np.radians(H1 + 35.0)))

    F = 0.0
    denom_f = C1 ** 4 + 1900.0
    if denom_f > 0:
        F = np.sqrt((C1 ** 4) / denom_f)

    S_L = 0.511 if L1 < 16.0 else (0.040975 * L1) / (1 + 0.01765 * L1)
    S_C = 0.0638 * C1 / (1 + 0.0131 * C1) + 0.638
    S_H = S_C * (F * T + 1 - F)

    l_w = max(l_weight, 1e-6)
    c_w = max(c_weight, 1e-6)
    S_L = max(S_L, 1e-6)
    S_C = max(S_C, 1e-6)
    S_H = max(S_H, 1e-6)

    return np.sqrt(
        (deltaL / (l_w * S_L)) ** 2
        + (deltaC / (c_w * S_C)) ** 2
        + deltaH_sq / (S_H ** 2)
    )


def ciede76(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """CIE76のΔE。"""
    delta = lab_array.astype(np.float64) - lab_sample.astype(np.float64)
    return np.sqrt(np.sum(delta ** 2, axis=1))


def ciede94(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """CIE94のΔE（グラフィックアーツ標準）。"""
    L1, a1, b1 = lab_sample.astype(np.float64)
    L2 = lab_array[:, 0].astype(np.float64)
    a2 = lab_array[:, 1].astype(np.float64)
    b2 = lab_array[:, 2].astype(np.float64)

    deltaL = L1 - L2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    deltaC = C1 - C2

    deltaE_ab_sq = (L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
    deltaH_sq = np.maximum(0.0, deltaE_ab_sq - deltaC ** 2)

    kL = kC = kH = 1.0
    K1 = 0.045
    K2 = 0.015

    S_L = 1.0
    S_C = 1.0 + K1 * C1
    S_H = 1.0 + K2 * C1

    return np.sqrt((deltaL / (kL * S_L)) ** 2 + (deltaC / (kC * S_C)) ** 2 + (deltaH_sq / (kH * S_H) ** 2))


def ciede2000(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 between one sample and many targets.

    lab_sample: shape (3,)
    lab_array: shape (N, 3)
    """
    L1, a1, b1 = lab_sample.astype(np.float64)
    L2 = lab_array[:, 0].astype(np.float64)
    a2 = lab_array[:, 1].astype(np.float64)
    b2 = lab_array[:, 2].astype(np.float64)

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p))
    h1p = np.where(h1p < 0, h1p + 360, h1p)
    h2p = np.degrees(np.arctan2(b2, a2p))
    h2p = np.where(h2p < 0, h2p + 360, h2p)

    deltahp = h2p - h1p
    deltahp = np.where(deltahp > 180, deltahp - 360, deltahp)
    deltahp = np.where(deltahp < -180, deltahp + 360, deltahp)

    deltaLp = L2 - L1
    deltaCp = C2p - C1p
    deltaHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp) / 2.0)

    avg_Hp = (h1p + h2p) / 2.0
    avg_Hp = np.where(np.abs(h1p - h2p) > 180, avg_Hp + 180, avg_Hp)
    avg_Hp = np.where(avg_Hp >= 360, avg_Hp - 360, avg_Hp)

    T = (
        1
        - 0.17 * np.cos(np.radians(avg_Hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_Hp))
        + 0.32 * np.cos(np.radians(3 * avg_Hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_Hp - 63))
    )

    Sl = 1 + (0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T

    delta_theta = 30 * np.exp(-((avg_Hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7))
    Rt = -np.sin(2 * np.radians(delta_theta)) * Rc

    deltaE = np.sqrt(
        (deltaLp / Sl) ** 2 + (deltaCp / Sc) ** 2 + (deltaHp / Sh) ** 2 + Rt * (deltaCp / Sc) * (deltaHp / Sh)
    )
    return deltaE
