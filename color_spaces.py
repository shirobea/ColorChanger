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
