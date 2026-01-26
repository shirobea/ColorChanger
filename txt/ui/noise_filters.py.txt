"""ノイズ除去フィルタの定義。"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) が必要です。pip install opencv-python") from exc
    return cv2


def build_noise_filter_registry(size: int) -> dict[str, Callable[[Image.Image], Image.Image]]:
    cv2 = _require_cv2()
    return {
        "メディアン": lambda img: Image.fromarray(
            cv2.medianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), size)
        ),
        "ガウシアン": lambda img: Image.fromarray(
            cv2.GaussianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), (size, size), 0)
        ),
        "バイラテラル": lambda img: Image.fromarray(
            cv2.bilateralFilter(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                size,
                sigmaColor=float(size * 6),
                sigmaSpace=float(size * 2),
            )
        ),
        "非局所的平均": lambda img: Image.fromarray(
            cv2.fastNlMeansDenoisingColored(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                None,
                h=8.0,
                hColor=10.0,
                templateWindowSize=size,
                searchWindowSize=max(7, size * 3),
            )
        ),
    }
