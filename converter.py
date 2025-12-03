"""Image conversion pipeline: resize, quantize, map to bead palette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple
import threading

import cv2
import numpy as np

from palette import BeadPalette
from color_spaces import rgb_to_lab, rgb_to_oklab, ciede2000

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


class _PaletteQuantizer:
    """減色とパレットマッピングをまとめて行うヘルパー。"""

    def __init__(self, config: _PipelineConfig) -> None:
        self.config = config
        self.quantize_lower = config.quantize_method.lower()

    def quantize_and_map(self, img: np.ndarray, progress_base: float, progress_after_map: float) -> np.ndarray:
        """指定画像を減色し、ビーズパレットに割り当てて返す。"""
        _report(self.config.progress_callback, progress_base, self.config.cancel_event)
        if self.config.cancel_event and self.config.cancel_event.is_set():
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


def _compute_saliency_map(image_rgb: np.ndarray) -> np.ndarray:
    """元画像に対してサリエンシーマップを計算（リサイズや減色前を必須とする）。"""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    saliency: np.ndarray | None = None

    # OpenCVのサリエンシーモジュールがあれば詳細版を優先利用
    try:
        if hasattr(cv2, "saliency"):
            try:
                fine = cv2.saliency.StaticSaliencyFineGrained_create()
                ok, saliency_map = fine.computeSaliency(img_bgr)
                if ok:
                    saliency = saliency_map.astype(np.float32)
            except Exception:
                saliency = None
            if saliency is None:
                spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
                ok, saliency_map = spectral.computeSaliency(img_bgr)
                if ok:
                    saliency = saliency_map.astype(np.float32)
    except Exception:
        saliency = None

    # フォールバック: Laplacianの強度をサリエンシーとして扱う
    if saliency is None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        saliency = np.abs(lap)

    # 0-1に正規化して軽くぼかし、極端なノイズを抑える
    saliency = saliency.astype(np.float32)
    min_v, max_v = float(saliency.min()), float(saliency.max())
    if max_v > min_v:
        saliency = (saliency - min_v) / (max_v - min_v)
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=1.0)
    return saliency


def _apply_saliency_enhancement(image_rgb: np.ndarray, saliency: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """重要度の高い領域だけ輪郭を強調する（パレット外の色は使わずに保持）。"""
    h, w = image_rgb.shape[:2]
    saliency_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    saliency_resized = np.clip(saliency_resized, 0.0, 1.0)[..., None]

    img_f = image_rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.2)
    high_freq = img_f - blurred  # 高周波成分

    # サリエンシーに応じてシャープ量を増やす（ベースは控えめに設定）
    boost = 0.35 + strength * saliency_resized
    enhanced = img_f + high_freq * boost
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _face_saliency_mask(image_rgb: np.ndarray) -> np.ndarray:
    """顔検出に基づくマスクを生成し、目・口をより強調する。ヒットなしなら全域0。"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # OpenCV付属の正面顔カスケードを利用（同梱パスに依存）
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return np.zeros_like(gray, dtype=np.float32)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    mask = np.zeros_like(gray, dtype=np.float32)
    for (x, y, w, h) in faces:
        # 顔全体: 基本重み
        mask[y : y + h, x : x + w] = 0.6
        # 目周辺: 強調
        eye_y0, eye_y1 = y + int(h * 0.18), y + int(h * 0.45)
        eye_x0, eye_x1 = x + int(w * 0.1), x + int(w * 0.9)
        mask[eye_y0:eye_y1, eye_x0:eye_x1] = 1.0
        # 口周辺: 強調
        mouth_y0, mouth_y1 = y + int(h * 0.62), y + int(h * 0.88)
        mouth_x0, mouth_x1 = x + int(w * 0.2), x + int(w * 0.8)
        mask[mouth_y0:mouth_y1, mouth_x0:mouth_x1] = 0.9

    if mask.max() > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0)
        mask = mask / max(mask.max(), 1e-5)
    return mask


def _apply_saliency_if_needed(
    base_rgb: np.ndarray,
    use_saliency: bool,
    saliency_strength: float,
    progress_callback: ProgressCb | None,
    cancel_event: CancelEvent | None,
) -> np.ndarray:
    """サリエンシー設定が有効な場合のみ強調処理を適用する。"""
    if not use_saliency or saliency_strength <= 0:
        return base_rgb

    saliency = _compute_saliency_map(base_rgb)
    face_mask = _face_saliency_mask(base_rgb)
    if face_mask.max() > 0:
        saliency = np.clip(0.5 * face_mask + 0.5 * saliency, 0.0, 1.0)

    _report(progress_callback, 0.08, cancel_event)
    enhanced = _apply_saliency_enhancement(base_rgb, saliency, strength=saliency_strength)
    _report(progress_callback, 0.12, cancel_event)
    return enhanced


def _imread_unicode(path: str) -> np.ndarray | None:
    """Read image allowing Unicode paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _load_image_rgb(input_path: str) -> np.ndarray:
    """入力パスからRGB画像を読み込む。Unicodeパスも許容。"""
    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        image_bgr = _imread_unicode(input_path)
    if image_bgr is None:
        raise ValueError("入力画像を読み込めませんでした。")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _compute_resize(
    shape: Size, target: int | Size, keep_aspect: bool
) -> Size:
    """指定サイズへリサイズし、keep_aspectがTrueなら縦横比を維持する。"""
    h, w = shape
    if isinstance(target, Iterable):
        width, height = target  # type: ignore
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("幅・高さは1以上にしてください。")
        if keep_aspect:
            scale = min(width / w, height / h)
            # 丸めの結果が0にならないよう最低1ピクセルにする
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
        else:
            new_w, new_h = width, height
        return new_w, new_h

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


def _compute_hybrid_size(
    shape: Size, target_size: Size, long_side_cap: int = 1600
) -> Size:
    """ハイブリッド用の中間リサイズサイズを算出（長辺を上限値まで縮小しつつ出力よりは大きく）。"""
    h, w = shape
    target_w, target_h = target_size
    long_side = max(w, h)
    if long_side <= long_side_cap:
        return w, h
    scale = long_side_cap / long_side
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
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


def _quantize_wu(image_rgb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Xiaolin Wu 法による高速最適化減色。32^3の3Dヒストグラムから分割を繰り返す。"""
    pixels = image_rgb.reshape(-1, 3).astype(np.int32)
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    # 5bit/チャネルに量子化（0は累積和用の余白に確保するので+1オフセット）
    bins = (pixels >> 3) + 1  # range 1..32
    idx = (bins[:, 0], bins[:, 1], bins[:, 2])

    shape = (33, 33, 33)
    wt = np.zeros(shape, dtype=np.int32)
    mr = np.zeros(shape, dtype=np.float64)
    mg = np.zeros(shape, dtype=np.float64)
    mb = np.zeros(shape, dtype=np.float64)
    m2 = np.zeros(shape, dtype=np.float64)

    np.add.at(wt, idx, 1)
    np.add.at(mr, idx, pixels[:, 0])
    np.add.at(mg, idx, pixels[:, 1])
    np.add.at(mb, idx, pixels[:, 2])
    np.add.at(m2, idx, pixels[:, 0] ** 2 + pixels[:, 1] ** 2 + pixels[:, 2] ** 2)

    # 累積和を三方向に取り、任意の直方体の総和をO(1)で求められるようにする
    wt = wt.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mr = mr.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mg = mg.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    mb = mb.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    m2 = m2.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    def _volume(cube: tuple[int, int, int, int, int, int], moment: np.ndarray) -> float:
        """累積テーブルから直方体領域の総和を取り出す。"""
        r0, r1, g0, g1, b0, b1 = cube
        return (
            moment[r1, g1, b1]
            - moment[r1, g1, b0]
            - moment[r1, g0, b1]
            - moment[r0, g1, b1]
            + moment[r1, g0, b0]
            + moment[r0, g1, b0]
            + moment[r0, g0, b1]
            - moment[r0, g0, b0]
        )

    def _partial(moment: np.ndarray, cube: tuple[int, int, int, int, int, int], direction: str, cut: int) -> float:
        """指定軸でcutまでの部分和を計算。方向毎に公式を入れ替える。"""
        r0, r1, g0, g1, b0, b1 = cube
        if direction == "r":
            return (
                moment[cut, g1, b1]
                - moment[cut, g1, b0]
                - moment[cut, g0, b1]
                + moment[cut, g0, b0]
                - moment[r0, g1, b1]
                + moment[r0, g1, b0]
                + moment[r0, g0, b1]
                - moment[r0, g0, b0]
            )
        if direction == "g":
            return (
                moment[r1, cut, b1]
                - moment[r1, cut, b0]
                - moment[r0, cut, b1]
                + moment[r0, cut, b0]
                - moment[r1, g0, b1]
                + moment[r1, g0, b0]
                + moment[r0, g0, b1]
                - moment[r0, g0, b0]
            )
        # direction == "b"
        return (
            moment[r1, g1, cut]
            - moment[r1, g0, cut]
            - moment[r0, g1, cut]
            + moment[r0, g0, cut]
            - moment[r1, g1, b0]
            + moment[r1, g0, b0]
            + moment[r0, g1, b0]
            - moment[r0, g0, b0]
        )

    def _variance(cube: tuple[int, int, int, int, int, int]) -> float:
        """箱内の分散（誤差）を評価する。"""
        w = _volume(cube, wt)
        if w == 0:
            return 0.0
        r = _volume(cube, mr)
        g = _volume(cube, mg)
        b = _volume(cube, mb)
        m2_val = _volume(cube, m2)
        return m2_val - (r * r + g * g + b * b) / w

    def _cut(cube: tuple[int, int, int, int, int, int]) -> tuple[str | None, int, tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int]]:
        """分散減少が最大となる軸と位置を探し、2つの箱を返す。"""
        whole_w = _volume(cube, wt)
        if whole_w == 0:
            return None, -1, cube, cube
        whole_r = _volume(cube, mr)
        whole_g = _volume(cube, mg)
        whole_b = _volume(cube, mb)

        best_score = 0.0
        best_dir: str | None = None
        best_cut = -1

        def _try_direction(direction: str, lo: int, hi: int) -> None:
            nonlocal best_score, best_dir, best_cut
            for cut in range(lo + 1, hi):
                w0 = _partial(wt, cube, direction, cut)
                w1 = whole_w - w0
                if w0 == 0 or w1 == 0:
                    continue
                r0 = _partial(mr, cube, direction, cut)
                g0 = _partial(mg, cube, direction, cut)
                b0 = _partial(mb, cube, direction, cut)
                r1 = whole_r - r0
                g1 = whole_g - g0
                b1 = whole_b - b0
                score = (r0 * r0 + g0 * g0 + b0 * b0) / w0 + (r1 * r1 + g1 * g1 + b1 * b1) / w1
                if score > best_score:
                    best_score = score
                    best_dir = direction
                    best_cut = cut

        r0, r1, g0, g1, b0, b1 = cube
        _try_direction("r", r0, r1)
        _try_direction("g", g0, g1)
        _try_direction("b", b0, b1)

        if best_dir is None:
            return None, -1, cube, cube

        if best_dir == "r":
            c1 = (r0, best_cut, g0, g1, b0, b1)
            c2 = (best_cut, r1, g0, g1, b0, b1)
        elif best_dir == "g":
            c1 = (r0, r1, g0, best_cut, b0, b1)
            c2 = (r0, r1, best_cut, g1, b0, b1)
        else:
            c1 = (r0, r1, g0, g1, b0, best_cut)
            c2 = (r0, r1, g0, g1, best_cut, b1)

        return best_dir, best_cut, c1, c2

    max_colors = min(k, 256)  # 安全上限
    # 初期キューブは全域 (0,32]（0はプレフィックス用の余白）
    cubes: list[tuple[int, int, int, int, int, int]] = [(0, 32, 0, 32, 0, 32)]

    for _ in range(1, max_colors):
        # 分散が最大の箱を選び分割
        variances = [(_variance(cube), idx) for idx, cube in enumerate(cubes)]
        variances.sort(reverse=True)
        _, split_idx = variances[0]
        split_cube = cubes[split_idx]
        direction, cut_pos, first, second = _cut(split_cube)
        if direction is None or cut_pos < 0:
            break  # これ以上有効に分割できない
        cubes[split_idx] = first
        cubes.append(second)

    centers = np.zeros((len(cubes), 3), dtype=np.float32)
    lookup = np.zeros((33, 33, 33), dtype=np.int32)

    for idx_cube, cube in enumerate(cubes):
        w = _volume(cube, wt)
        if w == 0:
            # 空の箱は近似値として境界中点を入れる
            centers[idx_cube] = np.array(
                [
                    (cube[0] + cube[1]) * 4.0,
                    (cube[2] + cube[3]) * 4.0,
                    (cube[4] + cube[5]) * 4.0,
                ],
                dtype=np.float32,
            )
        else:
            centers[idx_cube, 0] = _volume(cube, mr) / w
            centers[idx_cube, 1] = _volume(cube, mg) / w
            centers[idx_cube, 2] = _volume(cube, mb) / w

        # 各ヒストグラムビンをどの箱が担当するかマッピング
        r0, r1, g0, g1, b0, b1 = cube
        lookup[r0 + 1 : r1 + 1, g0 + 1 : g1 + 1, b0 + 1 : b1 + 1] = idx_cube

    labels = lookup[idx].astype(np.int32)
    return centers, labels


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


def _run_block_pipeline(image_rgb: np.ndarray, config: _PipelineConfig) -> np.ndarray:
    """ブロック単位の最多色をとる減色パス。色数指定は無視する。"""
    return _dominant_block_image(
        image_rgb=image_rgb,
        target_size=config.target_size,
        palette=config.palette,
        mode=config.mode,
        progress_callback=config.progress_callback,
        cancel_event=config.cancel_event,
    )


def _map_centers_to_palette(
    centers: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.5, 0.8),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """Map quantized centers to nearest bead palette color."""
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


def _dominant_block_image(
    image_rgb: np.ndarray,
    target_size: Size,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """指定サイズのグリッドに分け、各ブロックの最多色で塗りつぶした後にパレットへ写像する。"""

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("幅・高さは1以上にしてください。")

    orig_h, orig_w = image_rgb.shape[:2]
    # 出力解像度より小さい場合でもブロックが空にならないよう事前に拡大する
    work_w = max(orig_w, target_w)
    work_h = max(orig_h, target_h)
    if (work_w, work_h) != (orig_w, orig_h):
        working = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    else:
        working = image_rgb

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
                # 念のため1ピクセルだけ拾う（ここには通常来ない）
                fallback_px = working[min(y0, work_h - 1), min(x0, work_w - 1)]
                dominant_color = fallback_px
            else:
                flat = block.reshape(-1, 3)
                values, counts = np.unique(flat, axis=0, return_counts=True)
                dominant_color = values[counts.argmax()]
            dominant[yi, xi] = dominant_color
            processed += 1

        if total_blocks:
            _report(progress_callback, 0.8 * processed / total_blocks, cancel_event)

    # パレットへマッピング（ここで色空間選択を反映）
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


def convert_image(
    input_path: str,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    num_colors: int,
    quantize_method: str = "kmeans",
    keep_aspect: bool = True,
    pipeline: str = "resize_first",
    use_saliency: bool = False,
    saliency_strength: float = 0.9,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """Convert an image into bead palette colors.

    quantize_method: "kmeans" / "wu" / "block" を指定。"block" の場合は色数指定を無視する。
    use_saliency: Trueなら元画像のサリエンシーマップを使って目・口などの重要部分をシャープに保つ。
    saliency_strength: サリエンシー強調の強さ（0で無効、1.0前後が標準、上げるほど強調）。
    """
    _report(progress_callback, 0.0, cancel_event)
    image_rgb = _load_image_rgb(input_path)
    original_rgb = image_rgb.copy()

    image_rgb = _apply_saliency_if_needed(
        original_rgb, use_saliency, saliency_strength, progress_callback, cancel_event
    )

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

    pipeline_lower = pipeline.lower()
    if config.quantize_method.lower() == "block":
        return _run_block_pipeline(image_rgb, config)

    quantizer = _PaletteQuantizer(config)
    pipeline_map = {
        "quantize_first": _run_quantize_first_pipeline,
        "hybrid": _run_hybrid_pipeline,
        "resize_first": _run_resize_first_pipeline,
    }
    runner = pipeline_map.get(pipeline_lower, _run_resize_first_pipeline)
    return runner(image_rgb, quantizer, config)
