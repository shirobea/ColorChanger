"""Image conversion pipeline: resize, quantize, map to bead palette."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import cv2
import numpy as np

from palette import BeadPalette
from color_spaces import rgb_to_lab, rgb_to_oklab, ciede2000

ProgressCb = Callable[[float], None]


def _report(progress_callback: ProgressCb | None, value: float) -> None:
    if progress_callback:
        progress_callback(value)


def _imread_unicode(path: str) -> np.ndarray | None:
    """Read image allowing Unicode paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _compute_resize(shape: Tuple[int, int], target: int | Tuple[int, int]) -> Tuple[int, int]:
    """Return new (width, height) preserving aspect ratio when target is an int."""
    h, w = shape
    if isinstance(target, Iterable):
        width, height = target  # type: ignore
        return int(width), int(height)
    short = float(target)
    if short <= 0:
        raise ValueError("短辺ピクセル数は正の値にしてください。")
    if w <= h:
        new_w = int(short)
        new_h = int(round(h / w * short))
    else:
        new_h = int(short)
        new_w = int(round(w / h * short))
    return max(1, new_w), max(1, new_h)


def _quantize(image_rgb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means quantization; returns centers and labels."""
    data = image_rgb.reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(data)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.clip(centers, 0, 255).astype(np.float32)
    labels = labels.flatten()
    return centers, labels


def _map_centers_to_palette(
    centers: np.ndarray, palette: BeadPalette, mode: str, progress_callback: ProgressCb | None = None
) -> np.ndarray:
    """Map quantized centers to nearest bead palette color."""
    mode_upper = mode.upper()
    mapping = np.zeros(len(centers), dtype=np.int32)
    total = len(centers)

    if mode_upper == "RGB":
        diff = centers[:, None, :] - palette.rgb_array[None, :, :]
        distances = np.sum(diff ** 2, axis=2)
        mapping = np.argmin(distances, axis=1)
    elif mode_upper == "OKLAB":
        center_oklab = rgb_to_oklab(centers)
        diff = center_oklab[:, None, :] - palette.oklab_array[None, :, :]
        distances = np.sum(diff ** 2, axis=2)
        mapping = np.argmin(distances, axis=1)
    else:  # Lab + CIEDE2000
        center_lab = rgb_to_lab(centers)
        for idx, lab_value in enumerate(center_lab):
            distances = ciede2000(lab_value, palette.lab_array)
            mapping[idx] = int(np.argmin(distances))
            if progress_callback:
                _report(progress_callback, 0.5 + 0.3 * (idx + 1) / total)
        return mapping

    return mapping


def convert_image(
    input_path: str,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    num_colors: int,
    progress_callback: ProgressCb | None = None,
) -> np.ndarray:
    """Convert an image into bead palette colors."""
    _report(progress_callback, 0.0)
    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        image_bgr = _imread_unicode(input_path)
    if image_bgr is None:
        raise ValueError("入力画像を読み込めませんでした。")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    new_w, new_h = _compute_resize((image_rgb.shape[0], image_rgb.shape[1]), output_size)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _report(progress_callback, 0.15)

    centers, labels = _quantize(resized, num_colors)
    _report(progress_callback, 0.45)

    center_to_palette = _map_centers_to_palette(centers, palette, mode, progress_callback)

    mapped_colors = palette.rgb_array[center_to_palette].astype(np.uint8)
    output = mapped_colors[labels].reshape(resized.shape).astype(np.uint8)
    _report(progress_callback, 1.0)
    return output
