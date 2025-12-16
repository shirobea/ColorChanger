"""パレットへの最短距離マッピングを担う軽量ヘルパー。"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple
import threading

import numpy as np

from color_spaces import rgb_to_lab, rgb_to_oklab
from palette import BeadPalette

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
CHUNK_SIZE = 2048  # 距離行列のメモリ肥大化を防ぐチャンクサイズ


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う。"""
    if cancel_event and cancel_event.is_set():
        from . import ConversionCancelled  # 遅延インポートで循環を回避
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


def _chunk_ranges(length: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    """0-length区間を返さないチャンク分割のイテレータ。"""
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        yield start, end


def _ciede2000_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000の距離行列をベクトル化して返す。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    avg_l = (l1 + l2) * 0.5
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_c = (c1 + c2) * 0.5
    g = 0.5 * (1 - np.sqrt((avg_c ** 7) / (avg_c ** 7 + 25.0 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p ** 2 + b1 ** 2)
    c2p = np.sqrt(a2p ** 2 + b2 ** 2)
    avg_cp = (c1p + c2p) * 0.5

    h1p = np.degrees(np.arctan2(b1, a1p))
    h2p = np.degrees(np.arctan2(b2, a2p))
    h1p = np.where(h1p < 0, h1p + 360.0, h1p)
    h2p = np.where(h2p < 0, h2p + 360.0, h2p)

    deltahp = h2p - h1p
    deltahp = np.where(deltahp > 180.0, deltahp - 360.0, deltahp)
    deltahp = np.where(deltahp < -180.0, deltahp + 360.0, deltahp)

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    delta_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(deltahp) * 0.5)

    avg_hp = (h1p + h2p) * 0.5
    avg_hp = np.where(np.abs(h1p - h2p) > 180.0, avg_hp + 180.0, avg_hp)
    avg_hp = np.where(avg_hp >= 360.0, avg_hp - 360.0, avg_hp)

    t = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )

    sl = 1 + (0.015 * (avg_l - 50.0) ** 2) / np.sqrt(20.0 + (avg_l - 50.0) ** 2)
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t

    delta_theta = 30.0 * np.exp(-((avg_hp - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt((avg_cp ** 7) / (avg_cp ** 7 + 25.0 ** 7))
    rt = -np.sin(2.0 * np.radians(delta_theta)) * rc

    return np.sqrt(
        (delta_lp / sl) ** 2
        + (delta_cp / sc) ** 2
        + (delta_hp / sh) ** 2
        + rt * (delta_cp / sc) * (delta_hp / sh)
    )


def _ciede94_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE94距離の行列計算（グラフィックアーツ標準）。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    delta_l = l1 - l2
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    delta_c = c1 - c2

    delta_e_sq = (l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
    delta_h_sq = np.maximum(0.0, delta_e_sq - delta_c ** 2)

    k1 = 0.045
    k2 = 0.015
    s_l = 1.0
    s_c = 1.0 + k1 * c1
    s_h = 1.0 + k2 * c1

    return np.sqrt((delta_l / s_l) ** 2 + (delta_c / s_c) ** 2 + (delta_h_sq / (s_h ** 2)))


def _ciede76_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76距離の行列計算。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    diff = lab2[None, :, :] - lab1[:, None, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _cmc_matrix(lab1: np.ndarray, lab2: np.ndarray, l_weight: float, c_weight: float) -> np.ndarray:
    """CMC(l:c)をサンプル×パレットで一括計算。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l_w = max(l_weight, 1e-6)
    c_w = max(c_weight, 1e-6)

    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)

    delta_l = l1 - l2
    delta_c = c1 - c2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_h_sq = np.maximum(0.0, delta_a ** 2 + delta_b ** 2 - delta_c ** 2)

    h1 = np.degrees(np.arctan2(b1, a1))
    h1 = np.where(h1 < 0, h1 + 360.0, h1)
    t = np.where(
        (h1 >= 164.0) & (h1 <= 345.0),
        0.56 + np.abs(0.2 * np.cos(np.radians(h1 + 168.0))),
        0.36 + np.abs(0.4 * np.cos(np.radians(h1 + 35.0))),
    )

    denom_f = c1 ** 4 + 1900.0
    f = np.sqrt((c1 ** 4) / denom_f, where=denom_f > 0, out=np.zeros_like(denom_f))

    s_l = np.where(l1 < 16.0, 0.511, (0.040975 * l1) / (1 + 0.01765 * l1))
    s_c = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    s_h = s_c * (f * t + 1 - f)

    return np.sqrt((delta_l / (l_w * s_l)) ** 2 + (delta_c / (c_w * s_c)) ** 2 + delta_h_sq / (s_h ** 2))


def _compute_lab_distances(
    centers_lab: np.ndarray, palette_lab: np.ndarray, metric: str
) -> np.ndarray:
    """Lab系距離をチャンク単位で計算し、最小インデックスを返す。"""
    centers_lab = centers_lab.astype(np.float64, copy=False)
    palette_lab = palette_lab.astype(np.float64, copy=False)
    metric_upper = metric.upper()
    if metric_upper == "CIE76":
        dist_func = _ciede76_matrix
    elif metric_upper == "CIE94":
        dist_func = _ciede94_matrix
    else:
        dist_func = _ciede2000_matrix
    distances = dist_func(centers_lab, palette_lab)
    return np.argmin(distances, axis=1)


def _compute_cmc_distances(
    centers_lab: np.ndarray, palette_lab: np.ndarray, l_weight: float, c_weight: float
) -> np.ndarray:
    """CMC(l:c)距離をチャンク単位で計算し、最小インデックスを返す。"""
    distances = _cmc_matrix(
        centers_lab.astype(np.float64, copy=False),
        palette_lab.astype(np.float64, copy=False),
        l_weight,
        c_weight,
    )
    return np.argmin(distances, axis=1)


def _map_centers_to_palette(
    centers: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.5, 0.8),
    cancel_event: CancelEvent | None = None,
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lab_metric: str = "CIEDE2000",
) -> np.ndarray:
    """中心色をビーズパレットへ最近傍対応付けする。"""
    mode_upper = mode.upper()
    mapping = np.zeros(len(centers), dtype=np.int32)
    total = len(centers)
    start, end = progress_range
    span = max(0.0, end - start)

    if mode_upper == "RGB":
        w_r, w_g, w_b = (float(w) for w in rgb_weights)
        diff = centers[:, None, :] - palette.rgb_array[None, :, :]
        weights = np.array([max(w_r, 1e-6), max(w_g, 1e-6), max(w_b, 1e-6)], dtype=np.float32)
        weighted = diff * weights[None, None, :]
        distances = np.sum(weighted ** 2, axis=2)
        mapping = np.argmin(distances, axis=1)
    elif mode_upper == "OKLAB":
        center_oklab = rgb_to_oklab(centers)
        diff = center_oklab[:, None, :] - palette.oklab_array[None, :, :]
        distances = np.sum(diff ** 2, axis=2)
        mapping = np.argmin(distances, axis=1)
    elif mode_upper.startswith("CMC"):
        center_lab = rgb_to_lab(centers)
        l_weight = max(float(cmc_l), 1e-6)
        c_weight = max(float(cmc_c), 1e-6)
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_cmc_distances(chunk, palette.lab_array, l_weight, c_weight)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    elif mode_upper.startswith("LAB"):
        center_lab = rgb_to_lab(centers)
        metric = lab_metric.upper()
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_lab_distances(chunk, palette.lab_array, metric)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    else:  # Lab + CIEDE2000 (後方互換)
        center_lab = rgb_to_lab(centers)
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_lab_distances(chunk, palette.lab_array, "CIEDE2000")
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping

    _report(progress_callback, end, cancel_event)
    return mapping
