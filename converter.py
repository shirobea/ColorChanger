"""Image conversion pipeline: resize, quantize, map to bead palette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple
from pathlib import Path
import tempfile
import shutil
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

@dataclass(frozen=True)
class ImportanceWeights:
    """重要度マップを構成する各要素の重み。"""

    w_saliency: float = 0.35
    w_face: float = 0.50
    w_parts: float = 0.70
    w_eye: float = 0.90
    w_skin: float = 0.25
    w_center: float = 0.15


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


def compute_saliency_map(image_rgb: np.ndarray) -> np.ndarray:
    """外部から再利用できるサリエンシーマップ計算の公開関数。"""
    return _compute_saliency_map(image_rgb)


def _normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """0-1に正規化し、浮動小数へ変換するユーティリティ。"""
    sal = saliency.astype(np.float32)
    if sal.ndim == 3:
        sal = sal[..., 0]
    min_v, max_v = float(sal.min()), float(sal.max())
    if max_v > min_v:
        sal = (sal - min_v) / (max_v - min_v)
    return np.clip(sal, 0.0, 1.0)


def _compute_center_mask(shape: tuple[int, int]) -> np.ndarray:
    """画像中心を1.0、周縁を0.0に近づけるバイアスマスク。"""
    h, w = shape
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx**2 + yy**2)
    mask = np.exp(-(dist**2) / 0.45)  # 中心付近を緩やかに強調
    return np.clip(mask, 0.0, 1.0)


def _compute_skin_mask(image_rgb: np.ndarray, face_mask: np.ndarray | None = None) -> np.ndarray:
    """肌色に基づく簡易マスク。顔検出結果があればその領域を優先的に1.0へ近づける。"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.float32)
    # 2レンジの肌色判定（0-25度 と 160-180度付近）
    mask1 = ((h <= 25) | (h >= 160)) & (s >= 30) & (s <= 180) & (v >= 50)
    mask2 = (h >= 0) & (h <= 50) & (s >= 20) & (v >= 50)
    mask = np.logical_or(mask1, mask2).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0)
    mask = _normalize_saliency(mask)
    if face_mask is not None and face_mask.max() > 0:
        mask = np.clip(mask + 0.6 * face_mask, 0.0, 1.0)
    return mask


def _validate_faces(gray: np.ndarray, faces: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """顔矩形の面積・アスペクト比・位置をチェックして不自然な検出を除外する。"""
    h, w = gray.shape
    if h == 0 or w == 0:
        return []
    img_area = h * w
    valid: list[tuple[int, int, int, int]] = []
    for (x, y, fw, fh) in faces:
        # 範囲外やゼロサイズは無効
        if fw <= 0 or fh <= 0 or x < 0 or y < 0 or x + fw > w or y + fh > h:
            continue
        area = fw * fh
        area_ratio = area / float(img_area)
        if area_ratio < 0.01 or area_ratio > 0.60:  # 小さすぎ/大きすぎ
            continue
        aspect = fw / float(fh)
        if aspect < 0.6 or aspect > 1.8:  # 極端な横長/縦長を除外
            continue
        # 中心が画像外にはみ出す場合も除外
        cx, cy = x + fw * 0.5, y + fh * 0.5
        if cx < 0 or cx > w or cy < 0 or cy > h:
            continue
        valid.append((int(x), int(y), int(fw), int(fh)))
    return valid


def _to_face_list(faces: object) -> list[tuple[int, int, int, int]]:
    """detectMultiScale が返す ndarray/tuple を安全にリスト化する。"""
    if faces is None:
        return []
    if isinstance(faces, np.ndarray):
        return [tuple(map(int, box)) for box in faces.tolist()]
    if isinstance(faces, (list, tuple)):
        try:
            return [tuple(map(int, box)) for box in faces]
        except Exception:
            return []
    return []


def _load_cascade(cascade_path: str) -> cv2.CascadeClassifier | None:
    """Unicodeパスで読み込めない場合にテンポラリへコピーして読ませる安全ローダ。"""
    if not cascade_path:
        return None
    if not Path(cascade_path).exists():
        return None
    # 非ASCIIパスは先に一時ファイルへコピーして試す（OpenCVのUnicode対応が不安定なため）
    if not cascade_path.isascii():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
                shutil.copyfile(cascade_path, tmp.name)
                cascade_path = tmp.name
        except Exception:
            pass

    cas = cv2.CascadeClassifier(cascade_path)
    if not cas.empty():
        return cas
    # Unicodeパスで失敗した場合はASCIIな一時ファイルへコピー
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            shutil.copyfile(cascade_path, tmp.name)
            tmp_path = tmp.name
        cas_tmp = cv2.CascadeClassifier(tmp_path)
        return cas_tmp if not cas_tmp.empty() else None
    except Exception:
        return None


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """複数の検出器を試し、妥当性チェック後の顔矩形リストを返す。"""
    detectors: list[list[tuple[int, int, int, int]]] = []

    # 1) LBP（アニメ向け）を最優先。ローカルに置いた lbpcascade_animeface.xml をまず試す。
    try:
        lbp_path_candidates = [
            str(Path(__file__).resolve().parent / "lbpcascade_animeface.xml"),
            getattr(cv2.data, "haarcascades", "") + "lbpcascade_animeface.xml",
        ]
        lbp_path = next((p for p in lbp_path_candidates if Path(p).exists()), None)
        if lbp_path:
            lbp = _load_cascade(lbp_path)
            if lbp is not None:
                faces = lbp.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
                detectors.append(_to_face_list(faces))
    except Exception:
        pass

    # 2) Mediapipe が使える場合は次に試す（実写寄りでも対応）。
    try:
        import mediapipe as mp  # type: ignore

        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4) as fd:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            res = fd.process(rgb)
            boxes: list[tuple[int, int, int, int]] = []
            if res.detections:
                h, w = gray.shape
                for det in res.detections:
                    bbox = det.location_data.relative_bounding_box
                    x, y, bw, bh = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                    x0 = int(x * w)
                    y0 = int(y * h)
                    bw_px = int(bw * w)
                    bh_px = int(bh * h)
                    boxes.append((x0, y0, bw_px, bh_px))
            detectors.append(_to_face_list(boxes))
    except Exception:
        pass

    # 3) 最後にデフォルトのHaar
    try:
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        face_cascade = _load_cascade(cascade_path)
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            detectors.append(_to_face_list(faces))
    except Exception:
        pass

    # 妥当性チェックを通った最初の検出結果を採用
    for cand in detectors:
        valid = _validate_faces(gray, cand)
        if valid:
            return valid
    return []


def _compute_face_masks(
    image_rgb: np.ndarray,
    faces: list[tuple[int, int, int, int]] | None = None,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """顔検出から face_mask と facial_parts_mask を生成する。検出失敗時はゼロマスクを返す。"""
    if gray is None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    face_mask = np.zeros((h, w), dtype=np.float32)
    parts_mask = np.zeros((h, w), dtype=np.float32)
    if faces is None:
        faces = _detect_faces(gray)
    else:
        faces = _to_face_list(faces)
    if not faces:
        return face_mask, parts_mask, []

    max_box = 0
    for (x, y, fw, fh) in faces:
        max_box = max(max_box, max(fw, fh))
        # 顔ボックスを少し拡張して髪・輪郭を含める（矩形→ソフトマスク化は後段で実施）
        margin = int(0.08 * max(fw, fh))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + fw + margin)
        y1 = min(h, y + fh + margin)
        face_mask[y0:y1, x0:x1] = np.maximum(face_mask[y0:y1, x0:x1], 1.0)

        # 目・口の簡易推定（顔ボックス内の比率で矩形配置）
        eye_y0, eye_y1 = y0 + int(0.18 * fh), y0 + int(0.42 * fh)
        eye_x0, eye_x1 = x0 + int(0.12 * fw), x0 + int(0.44 * fw)
        eye_x2, eye_x3 = x0 + int(0.56 * fw), x0 + int(0.88 * fw)
        parts_mask[eye_y0:eye_y1, eye_x0:eye_x1] = 1.0
        parts_mask[eye_y0:eye_y1, eye_x2:eye_x3] = 1.0

        mouth_y0, mouth_y1 = y0 + int(0.62 * fh), y0 + int(0.82 * fh)
        mouth_x0, mouth_x1 = x0 + int(0.22 * fw), x0 + int(0.78 * fw)
        parts_mask[mouth_y0:mouth_y1, mouth_x0:mouth_x1] = 0.9

    # ボックスサイズに応じて強めにぼかし、矩形感を減らす
    sigma_face = max(3.0, 0.12 * max_box) if max_box > 0 else 3.0
    sigma_parts = max(2.5, 0.08 * max_box) if max_box > 0 else 2.5
    if face_mask.max() > 0:
        face_mask = cv2.GaussianBlur(face_mask, (0, 0), sigmaX=sigma_face, sigmaY=sigma_face)
        face_mask = _normalize_saliency(face_mask)
    if parts_mask.max() > 0:
        parts_mask = cv2.GaussianBlur(parts_mask, (0, 0), sigmaX=sigma_parts, sigmaY=sigma_parts)
        parts_mask = _normalize_saliency(parts_mask)
    return face_mask, parts_mask, faces


def _compute_eye_mask(
    image_rgb: np.ndarray,
    faces: list[tuple[int, int, int, int]] | None = None,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """目専用の重要度マスクを生成する。検出失敗時は幾何学的推定で補完する。"""
    if gray is None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.float32)

    if faces is None:
        faces = _detect_faces(gray)
    else:
        faces = _to_face_list(faces)
    if not faces:
        return mask

    max_box = 0
    eye_cascade: cv2.CascadeClassifier | None = None
    try:
        eye_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(eye_path)
        if eye_cascade.empty():
            eye_cascade = None
    except Exception:
        eye_cascade = None

    for (x, y, fw, fh) in faces:
        max_box = max(max_box, max(fw, fh))
        roi_gray = gray[y : y + fh, x : x + fw]
        detected: list[tuple[int, int, int, int]] = []
        if eye_cascade is not None:
            try:
                detected = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(max(12, fw // 8), max(10, fh // 10)),
                )
            except Exception:
                detected = []

        if detected:
            for (ex, ey, ew, eh) in detected:
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                axes = (max(2, int(ew * 0.6)), max(2, int(eh * 0.7)))
                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
            if len(detected) < 2:
                eye_y0, eye_y1 = y + int(0.25 * fh), y + int(0.55 * fh)
                eye_x0, eye_x1 = x + int(0.10 * fw), x + int(0.45 * fw)
                eye_x2, eye_x3 = x + int(0.55 * fw), x + int(0.90 * fw)
                mask[eye_y0:eye_y1, eye_x0:eye_x1] = np.maximum(mask[eye_y0:eye_y1, eye_x0:eye_x1], 0.8)
                mask[eye_y0:eye_y1, eye_x2:eye_x3] = np.maximum(mask[eye_y0:eye_y1, eye_x2:eye_x3], 0.8)
        else:
            eye_y0, eye_y1 = y + int(0.25 * fh), y + int(0.55 * fh)
            eye_x0, eye_x1 = x + int(0.10 * fw), x + int(0.45 * fw)
            eye_x2, eye_x3 = x + int(0.55 * fw), x + int(0.90 * fw)
            mask[eye_y0:eye_y1, eye_x0:eye_x1] = 1.0
            mask[eye_y0:eye_y1, eye_x2:eye_x3] = 1.0

    if mask.max() > 0:
        sigma_eye = max(2.4, 0.08 * max_box) if max_box > 0 else 2.4
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma_eye, sigmaY=sigma_eye)
        mask = _normalize_saliency(mask)
    return mask


def compute_importance_map(
    image_rgb: np.ndarray,
    saliency_map: np.ndarray | None = None,
    weights: ImportanceWeights | None = None,
    eye_importance_scale: float = 0.8,
) -> np.ndarray:
    """サリエンシー・顔・肌・中心バイアスを統合した重要度マップを生成する。"""
    try:
        if weights is None:
            weights = ImportanceWeights()
        h, w = image_rgb.shape[:2]
        sal = _normalize_saliency(saliency_map) if saliency_map is not None else _compute_saliency_map(image_rgb)
        sal_soft = cv2.GaussianBlur(sal, (0, 0), sigmaX=2.0)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        face_mask, parts_mask, faces = _compute_face_masks(image_rgb, faces=None, gray=gray)
        eye_mask = _compute_eye_mask(image_rgb, faces=faces, gray=gray)
        # サリエンシーに沿って顔・目マスクをなじませ、矩形感を抑える
        if face_mask.max() > 0:
            face_mask = np.clip(face_mask * (0.35 + 0.65 * sal_soft), 0.0, 1.0)
        if parts_mask.max() > 0:
            parts_mask = np.clip(parts_mask * (0.25 + 0.75 * sal_soft), 0.0, 1.0)
        if eye_mask.max() > 0 and eye_importance_scale > 0:
            eye_mask = np.clip(eye_mask * (0.30 + 0.70 * sal_soft), 0.0, 1.0)
            sal = np.maximum(sal, np.clip(eye_mask * float(eye_importance_scale), 0.0, 1.0))
        skin_mask = _compute_skin_mask(image_rgb, face_mask if face_mask.max() > 0 else None)
        center_mask = _compute_center_mask((h, w))

        importance = (
            weights.w_saliency * sal
            + weights.w_face * face_mask
            + weights.w_parts * parts_mask
            + weights.w_eye * eye_mask
            + weights.w_skin * skin_mask
            + weights.w_center * center_mask
        )
        importance = np.clip(importance, 0.0, None)
        max_v = float(importance.max())
        if max_v > 0:
            importance = importance / max_v
        return importance.astype(np.float32)
    except ValueError as err:
        # numpy配列の真偽値評価エラーなどが起きた場合は安全側でサリエンシーのみを返す
        sal_safe = _normalize_saliency(saliency_map) if saliency_map is not None else _compute_saliency_map(image_rgb)
        return sal_safe.astype(np.float32)



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


def _map_image_to_palette(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.4, 0.9),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """減色を行わずに元色をそのままパレットへ最短距離で写像する。"""
    start, end = progress_range
    _report(progress_callback, start, cancel_event)
    flat = image_rgb.reshape(-1, 3).astype(np.float32)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    mapping = _map_centers_to_palette(
        colors,
        palette,
        mode,
        progress_callback=progress_callback,
        progress_range=(start, end),
        cancel_event=cancel_event,
    )
    mapped = palette.rgb_array[mapping].astype(np.uint8)[inv].reshape(image_rgb.shape)
    _report(progress_callback, end, cancel_event)
    return mapped


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


def _run_block_pipeline(
    image_rgb: np.ndarray,
    config: _PipelineConfig,
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
) -> np.ndarray:
    """ブロック単位の最多色をとる減色パス。色数指定は無視する。"""
    return _dominant_block_image(
        image_rgb=image_rgb,
        target_size=config.target_size,
        palette=config.palette,
        mode=config.mode,
        progress_callback=config.progress_callback,
        cancel_event=config.cancel_event,
        saliency_map=saliency_map,
        contour_enhance=contour_enhance,
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
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
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
    saliency_work: np.ndarray | None = None
    if contour_enhance and saliency_map is not None:
        saliency_work = cv2.resize(np.clip(saliency_map.astype(np.float32), 0.0, 1.0), (work_w, work_h), interpolation=cv2.INTER_LINEAR)

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
                if contour_enhance and saliency_work is not None:
                    sal_block = saliency_work[y0:y1, x0:x1]
                    sal_flat = sal_block.reshape(-1).astype(np.float32)
                    if sal_flat.size and sal_flat.max() > 0:
                        sal_flat = sal_flat / sal_flat.max()
                    colors, inv = np.unique(flat, axis=0, return_inverse=True)
                    weights = np.zeros(len(colors), dtype=np.float32)
                    np.add.at(weights, inv, sal_flat if sal_flat.size == len(inv) else np.ones(len(inv), dtype=np.float32))
                    dominant_color = colors[weights.argmax()]
                else:
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


def _adaptive_block_image(
    image_rgb: np.ndarray,
    target_size: Size,
    palette: BeadPalette,
    mode: str,
    saliency_weight: float,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    importance_map: np.ndarray | None = None,
    fine_scale: int = 2,
    saliency_map: np.ndarray | None = None,
    contour_enhance: bool = False,
) -> np.ndarray:
    """重要度マップを用いて「細かい」or「通常」の2段階でブロック代表色を求める。"""

    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("幅・高さは1以上にしてください。")

    # 反映度0なら従来のブロック減色と同じ挙動に切り替える
    if saliency_weight <= 0:
        return _dominant_block_image(
            image_rgb=image_rgb,
            target_size=target_size,
            palette=palette,
            mode=mode,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    # 出力解像度に合わせた作業画像と重要度マップを用意する
    fine_scale = max(1, int(fine_scale))

    # 出力解像度に合わせた作業画像と重要度マップを用意する
    orig_h, orig_w = image_rgb.shape[:2]
    work_w = max(orig_w, target_w)
    work_h = max(orig_h, target_h)
    if (work_w, work_h) != (orig_w, orig_h):
        working = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    else:
        working = image_rgb

    if importance_map is None:
        importance_map = compute_importance_map(image_rgb)
    importance_resized = cv2.resize(importance_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    importance_blurred = cv2.GaussianBlur(importance_resized, (0, 0), sigmaX=1.2)
    importance_blurred = np.clip(importance_blurred.astype(np.float32), 0.0, 1.0)
    saliency_work: np.ndarray | None = None
    if contour_enhance and saliency_map is not None:
        saliency_work = cv2.resize(np.clip(saliency_map.astype(np.float32), 0.0, 1.0), (work_w, work_h), interpolation=cv2.INTER_LINEAR)

    weight = float(np.clip(saliency_weight, 0.0, 1.0))
    base_threshold = 0.50
    threshold = float(np.clip(base_threshold - weight * 0.25, 0.18, 0.72))
    fine_mask = (importance_blurred >= threshold).astype(np.uint8)

    # 通常ブロックの幅・高さ（通常のブロック減色と同じ解像度）
    x_edges = np.linspace(0, work_w, target_w + 1, dtype=int)
    y_edges = np.linspace(0, work_h, target_h + 1, dtype=int)

    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    total_blocks = target_w * target_h
    processed = 0

    # 動作確認: face_sample.png（中央に人物）で顔の目鼻が細かく、背景が通常ブロックになることを目視確認済み。
    for yi in range(target_h):
        y0, y1 = y_edges[yi], y_edges[yi + 1]
        norm_h = max(1, y1 - y0)
        for xi in range(target_w):
            x0, x1 = x_edges[xi], x_edges[xi + 1]
            norm_w = max(1, x1 - x0)

            if fine_mask[yi, xi]:
                # 重要度が高い領域は中心付近を細かいブロックとして扱う
                fine_w = max(1, norm_w // fine_scale)
                fine_h = max(1, norm_h // fine_scale)
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                fx0 = max(0, cx - fine_w // 2)
                fx1 = min(work_w, fx0 + fine_w)
                fy0 = max(0, cy - fine_h // 2)
                fy1 = min(work_h, fy0 + fine_h)
                block = working[fy0:fy1, fx0:fx1, :]
            else:
                # 重要度が低い領域でも通常ブロックサイズ以下にはしない
                block = working[y0:y1, x0:x1, :]

            if block.size == 0:
                fallback_px = working[min(y0, work_h - 1), min(x0, work_w - 1)]
                dominant_color = fallback_px
            else:
                flat = block.reshape(-1, 3)
                if contour_enhance and saliency_work is not None:
                    sal_block = saliency_work[y0:y1, x0:x1]
                    sal_flat = sal_block.reshape(-1).astype(np.float32)
                    if sal_flat.size and sal_flat.max() > 0:
                        sal_flat = sal_flat / sal_flat.max()
                    colors, inv = np.unique(flat, axis=0, return_inverse=True)
                    weights = np.zeros(len(colors), dtype=np.float32)
                    np.add.at(weights, inv, sal_flat if sal_flat.size == len(inv) else np.ones(len(inv), dtype=np.float32))
                    dominant_color = colors[weights.argmax()]
                else:
                    values, counts = np.unique(flat, axis=0, return_counts=True)
                    dominant_color = values[counts.argmax()]

            output[yi, xi] = dominant_color
            processed += 1
            if total_blocks:
                _report(progress_callback, 0.1 + 0.65 * processed / total_blocks, cancel_event)

    centers = output.reshape(-1, 3).astype(np.float32)
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
    contour_enhance: bool = False,
    eye_importance_scale: float = 0.8,
    adaptive_saliency_weight: float = 0.5,
    fine_scale: int = 2,
    saliency_map: np.ndarray | None = None,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """Convert an image into bead palette colors.

    quantize_method: "none" / "kmeans" / "wu" / "block" / "adaptive_block" を指定。"block"系および"none"の場合は色数指定を無視する。
    contour_enhance: Trueならサリエンシーマップを輪郭強調に用いてブロック代表色の重みに反映する。
    eye_importance_scale: 目を最低でもこの値まで底上げする強度（0.0～1.0推奨）。
    adaptive_saliency_weight: 適応型ブロック減色時にサリエンシーをどの程度反映するか（0.0～1.0）。
    fine_scale: 重要領域のブロックをどれだけ細かくするか（2なら1/2サイズ）。
    saliency_map: 事前計算済みのサリエンシーマップを渡す場合に利用（0～1の2D配列）。
    """
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
