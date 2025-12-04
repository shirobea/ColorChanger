"""サリエンシー計算・顔/肌/中心バイアスを扱うユーティリティ群。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import tempfile
import shutil

import cv2
import numpy as np

Size = Tuple[int, int]


@dataclass(frozen=True)
class ImportanceWeights:
    """重要度マップを構成する各要素の重み。"""

    w_saliency: float = 0.35
    w_face: float = 0.50
    w_parts: float = 0.70
    w_eye: float = 0.90
    w_skin: float = 0.25
    w_center: float = 0.15


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
    except ValueError:
        # numpy配列の真偽値評価エラーなどが起きた場合は安全側でサリエンシーのみを返す
        sal_safe = _normalize_saliency(saliency_map) if saliency_map is not None else _compute_saliency_map(image_rgb)
        return sal_safe.astype(np.float32)
