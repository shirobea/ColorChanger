"""減色アルゴリズムとパレット対応付けを担うモジュール。"""

from __future__ import annotations

from typing import Callable, Tuple, TYPE_CHECKING
import threading

import cv2
import numpy as np

from color_spaces import ciede2000, rgb_to_lab, rgb_to_oklab
from palette import BeadPalette

if TYPE_CHECKING:
    from . import _PipelineConfig, ConversionCancelled

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う。"""
    if cancel_event and cancel_event.is_set():
        from . import ConversionCancelled  # 遅延インポートで循環を回避
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


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


def _quantize_wu(image_rgb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Xiaolin Wu 法による高速最適化減色。32^3の3Dヒストグラムから分割を繰り返す。"""
    pixels = image_rgb.reshape(-1, 3).astype(np.int32)
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    bins = (pixels >> 3) + 1  # range 1..32
    bin_indices = (bins[:, 0], bins[:, 1], bins[:, 2])

    shape = (34, 34, 34)  # 0..33 を安全に含む
    wt = np.zeros(shape, dtype=np.int32)
    mr = np.zeros(shape, dtype=np.float64)
    mg = np.zeros(shape, dtype=np.float64)
    mb = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)

    np.add.at(wt, bin_indices, 1)
    np.add.at(mr, bin_indices, pixels[:, 0])
    np.add.at(mg, bin_indices, pixels[:, 1])
    np.add.at(mb, bin_indices, pixels[:, 2])
    np.add.at(m2, bin_indices, pixels[:, 0] ** 2 + pixels[:, 1] ** 2 + pixels[:, 2] ** 2)

    wt = wt.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mr = mr.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mg = mg.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mb = mb.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    m2 = m2.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    class Box:
        """累積和テーブルに対する半開区間 [r0, r1) のキューブ。"""

        def __init__(self, r0: int, r1: int, g0: int, g1: int, b0: int, b1: int) -> None:
            self.r0, self.r1 = r0, r1
            self.g0, self.g1 = g0, g1
            self.b0, self.b1 = b0, b1

    def vol(table: np.ndarray, box: Box) -> float:
        return (
            table[box.r1, box.g1, box.b1]
            - table[box.r1, box.g1, box.b0]
            - table[box.r1, box.g0, box.b1]
            + table[box.r1, box.g0, box.b0]
            - table[box.r0, box.g1, box.b1]
            + table[box.r0, box.g1, box.b0]
            + table[box.r0, box.g0, box.b1]
            - table[box.r0, box.g0, box.b0]
        )

    def variance(box: Box) -> float:
        dr = vol(mr, box)
        dg = vol(mg, box)
        db = vol(mb, box)
        xx = vol(m2, box)
        wt_box = vol(wt, box)
        if wt_box == 0:
            return 0.0
        return xx - (dr * dr + dg * dg + db * db) / wt_box

    def maximize(box: Box, direction: int, first: int, last: int) -> tuple[float, int]:
        cut_at = -1
        max_var = -1.0
        for i in range(first, last):
            if direction == 0:
                box1 = Box(box.r0, i, box.g0, box.g1, box.b0, box.b1)
                box2 = Box(i, box.r1, box.g0, box.g1, box.b0, box.b1)
            elif direction == 1:
                box1 = Box(box.r0, box.r1, box.g0, i, box.b0, box.b1)
                box2 = Box(box.r0, box.r1, i, box.g1, box.b0, box.b1)
            else:
                box1 = Box(box.r0, box.r1, box.g0, box.g1, box.b0, i)
                box2 = Box(box.r0, box.r1, box.g0, box.g1, i, box.b1)
            w1 = vol(wt, box1)
            w2 = vol(wt, box2)
            if w1 == 0 or w2 == 0:
                continue
            mean1 = (vol(mr, box1), vol(mg, box1), vol(mb, box1))
            mean2 = (vol(mr, box2), vol(mg, box2), vol(mb, box2))
            var = (
                np.sum(np.square(mean1)) / w1
                + np.sum(np.square(mean2)) / w2
            )
            if var > max_var:
                max_var = var
                cut_at = i
        return max_var, cut_at

    def split(box: Box) -> tuple[Box, Box]:
        max_r, cut_r = maximize(box, 0, box.r0 + 1, box.r1)
        max_g, cut_g = maximize(box, 1, box.g0 + 1, box.g1)
        max_b, cut_b = maximize(box, 2, box.b0 + 1, box.b1)
        if max_r >= max_g and max_r >= max_b:
            return Box(box.r0, cut_r, box.g0, box.g1, box.b0, box.b1), Box(cut_r, box.r1, box.g0, box.g1, box.b0, box.b1)
        if max_g >= max_r and max_g >= max_b:
            return Box(box.r0, box.r1, box.g0, cut_g, box.b0, box.b1), Box(box.r0, box.r1, cut_g, box.g1, box.b0, box.b1)
        return Box(box.r0, box.r1, box.g0, box.g1, box.b0, cut_b), Box(box.r0, box.r1, box.g0, box.g1, cut_b, box.b1)

    def create_box_mask(boxes: list[Box]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.int32)
        for idx_box, box in enumerate(boxes):
            mask[box.r0:box.r1, box.g0:box.g1, box.b0:box.b1] = idx_box
        return mask

    # 実データ範囲は 1..32（0 は累積和の基点）。半開区間で上端33ならインデックスは最大32で安全。
    boxes = [Box(1, 33, 1, 33, 1, 33)]
    while len(boxes) < k:
        variances = [variance(b) for b in boxes]
        idx_max = int(np.argmax(variances))
        box_to_split = boxes[idx_max]
        box1, box2 = split(box_to_split)
        boxes[idx_max] = box1
        boxes.append(box2)

    box_mask = create_box_mask(boxes)
    wt_nonzero = max(vol(wt, boxes[0]), 1)
    centers = np.zeros((len(boxes), 3), dtype=np.float32)
    for box_idx, box in enumerate(boxes):
        w = vol(wt, box)
        if w == 0:
            w = wt_nonzero
        centers[box_idx, 0] = vol(mr, box) / w
        centers[box_idx, 1] = vol(mg, box) / w
        centers[box_idx, 2] = vol(mb, box) / w

    labels = box_mask[bin_indices]  # 0-based
    labels = np.clip(labels, 0, len(centers) - 1)
    centers = np.clip(centers, 0, 255)
    return centers.astype(np.float32), labels.astype(np.int32)


def _map_centers_to_palette(
    centers: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.5, 0.8),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """量子化後の中心色をビーズパレットへ最近傍対応付けする。"""
    mode_upper = mode.upper()
    mapping = np.zeros(len(centers), dtype=np.int32)
    total = len(centers)
    start, end = progress_range
    span = max(0.0, end - start)

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
            _report(progress_callback, start + span * (idx + 1) / total, cancel_event)
        return mapping

    _report(progress_callback, end, cancel_event)
    return mapping


class _PaletteQuantizer:
    """減色とパレットマッピングをまとめて行うヘルパー。"""

    def __init__(self, config: "_PipelineConfig") -> None:
        self.config = config
        self.quantize_lower = config.quantize_method.lower()

    def quantize_and_map(self, img: np.ndarray, progress_base: float, progress_after_map: float) -> np.ndarray:
        """指定画像を減色し、ビーズパレットに割り当てて返す。"""
        _report(self.config.progress_callback, progress_base, self.config.cancel_event)
        if self.config.cancel_event and self.config.cancel_event.is_set():
            from . import ConversionCancelled
            raise ConversionCancelled()

        if self.quantize_lower == "wu":
            centers, labels = _quantize_wu(img, self.config.num_colors)
        else:
            centers, labels = _quantize(img, self.config.num_colors)

        center_to_palette = _map_centers_to_palette(
            centers,
            self.config.palette,
            self.config.mode,
            self.config.progress_callback,
            progress_range=(progress_base, progress_after_map),
            cancel_event=self.config.cancel_event,
        )
        mapped_colors = self.config.palette.rgb_array[center_to_palette].astype(np.uint8)
        return mapped_colors[labels].reshape(img.shape).astype(np.uint8)
