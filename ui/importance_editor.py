"""重要度マップの編集（ブラシ適用とUndo/Redo）を司るユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List
import numpy as np


@dataclass
class _StrokeRecord:
    """1回のストローク前後の差分を保持してUndo/Redoに使う。"""

    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    before: np.ndarray
    after: np.ndarray


class ImportanceEditor:
    """重要度マップへ筆圧風ブラシを重ねるシンプルな編集器。"""

    def __init__(self) -> None:
        self._map: Optional[np.ndarray] = None
        self._undo: list[_StrokeRecord] = []
        self._redo: list[_StrokeRecord] = []
        self._brush_cache: dict[int, np.ndarray] = {}
        # ストローク単位のUndo用バッファ
        self._stroke_before: Optional[np.ndarray] = None
        self._stroke_active: bool = False

    @property
    def current_map(self) -> Optional[np.ndarray]:
        """現在保持している重要度マップを返す。"""
        return self._map

    def load_map(self, importance_map: np.ndarray) -> np.ndarray:
        """編集用にマップを受け取り、Undoスタックをリセットする。"""
        self._map = np.clip(importance_map.astype(np.float32), 0.0, 1.0)
        self._undo.clear()
        self._redo.clear()
        self._stroke_before = None
        self._stroke_active = False
        return self._map

    def clear(self) -> None:
        """画像切替時などに状態を空にする。"""
        self._map = None
        self._undo.clear()
        self._redo.clear()
        self._stroke_before = None
        self._stroke_active = False

    def begin_stroke(self) -> bool:
        """ストローク開始。Undo用に全体のスナップショットを保存する。"""
        if self._map is None:
            return False
        self._stroke_before = self._map.copy()
        self._stroke_active = True
        return True

    def paint_live(
        self,
        points: Iterable[Tuple[float, float]],
        radius: int,
        strength: float,
        mode: str = "add",
    ) -> bool:
        """ストローク中にリアルタイムで描画（Undoはまだ積まない）。"""
        if self._map is None or not self._stroke_active:
            return False
        pts = self._densify(points)
        if pts.size == 0:
            return False
        h, w = self._map.shape[:2]
        r = max(1, int(radius))
        xs, ys = pts[:, 0], pts[:, 1]
        x0 = max(0, int(np.floor(xs.min()) - r))
        x1 = min(w, int(np.ceil(xs.max()) + r + 1))
        y0 = max(0, int(np.floor(ys.min()) - r))
        y1 = min(h, int(np.ceil(ys.max()) + r + 1))
        if x0 >= x1 or y0 >= y1:
            return False
        self._paint(pts, r, strength, mode, (x0, y0, x1, y1))
        return True

    def commit_stroke(self) -> bool:
        """ストローク終了時に1レコードとしてUndoスタックへ積む。"""
        if not self._stroke_active or self._map is None or self._stroke_before is None:
            self._stroke_active = False
            self._stroke_before = None
            return False
        before_map = self._stroke_before
        after_map = self._map
        diff = np.abs(after_map - before_map) > 1e-6
        if not diff.any():
            self._stroke_active = False
            self._stroke_before = None
            return False
        ys, xs = np.nonzero(diff)
        x0 = int(xs.min())
        x1 = int(xs.max() + 1)
        y0 = int(ys.min())
        y1 = int(ys.max() + 1)
        before = before_map[y0:y1, x0:x1].copy()
        after = after_map[y0:y1, x0:x1].copy()
        self._undo.append(_StrokeRecord((x0, y0, x1, y1), before, after))
        self._redo.clear()
        self._stroke_active = False
        self._stroke_before = None
        return True

    def undo(self) -> bool:
        """一つ前のストロークを取り消す。"""
        if not self._undo or self._map is None:
            return False
        record = self._undo.pop()
        x0, y0, x1, y1 = record.bbox
        self._redo.append(record)
        self._map[y0:y1, x0:x1] = record.before
        return True

    def redo(self) -> bool:
        """Undoしたストロークをやり直す。"""
        if not self._redo or self._map is None:
            return False
        record = self._redo.pop()
        x0, y0, x1, y1 = record.bbox
        self._undo.append(record)
        self._map[y0:y1, x0:x1] = record.after
        return True

    def reset_to(self, base_map: np.ndarray) -> np.ndarray:
        """初期状態のマップへ戻し、スタックも空にする。"""
        return self.load_map(base_map)

    def fill_all(self, value: float) -> bool:
        """マップ全体を指定値で塗りつぶす（Undo対応）。"""
        if self._map is None:
            return False
        # 進行中のストロークは無かったことにする
        self._stroke_active = False
        self._stroke_before = None

        v = float(np.clip(value, 0.0, 1.0))
        if np.allclose(self._map, v, atol=1e-6):
            return False

        before = self._map.copy()
        self._map[:, :] = v
        after = np.full_like(self._map, v, dtype=np.float32)
        h, w = self._map.shape[:2]
        self._undo.append(_StrokeRecord((0, 0, w, h), before, after))
        self._redo.clear()
        return True

    # --- 内部処理 ---
    def _densify(self, points: Iterable[Tuple[float, float]]) -> np.ndarray:
        """ドラッグ軌跡を1px間隔程度に補間して塗り残しを防ぐ。"""
        pts: List[Tuple[float, float]] = list(points)
        if not pts:
            return np.empty((0, 2), dtype=np.float32)
        if len(pts) == 1:
            return np.array(pts, dtype=np.float32)
        samples: list[np.ndarray] = []
        for a, b in zip(pts[:-1], pts[1:]):
            ax, ay = a
            bx, by = b
            steps = int(max(abs(bx - ax), abs(by - ay))) + 1
            if steps < 2:
                steps = 2
            xs = np.linspace(ax, bx, steps)
            ys = np.linspace(ay, by, steps)
            samples.append(np.stack([xs, ys], axis=1))
        return np.concatenate(samples, axis=0).astype(np.float32)

    def _get_brush(self, radius: int) -> np.ndarray:
        """半径ごとにソフト円形ブラシをキャッシュする。"""
        r = max(1, int(radius))
        if r in self._brush_cache:
            return self._brush_cache[r]
        grid = np.linspace(-1.0, 1.0, 2 * r + 1, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid)
        dist2 = xx * xx + yy * yy
        mask = np.exp(-4.0 * dist2)
        mask = mask / mask.max()
        mask[mask < 1e-3] = 0.0  # 外周は0に近いので省く
        self._brush_cache[r] = mask.astype(np.float32)
        return self._brush_cache[r]

    def _paint(
        self,
        pts: np.ndarray,
        radius: int,
        strength: float,
        mode: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """指定されたbbox内でブラシを重ねる。"""
        if self._map is None:
            return
        x0, y0, x1, y1 = bbox
        brush = self._get_brush(radius)
        r = brush.shape[0] // 2
        strength = float(max(0.0, strength))
        for x_f, y_f in pts:
            cx = int(round(x_f))
            cy = int(round(y_f))
            bx0 = max(x0, cx - r)
            bx1 = min(x1, cx + r + 1)
            by0 = max(y0, cy - r)
            by1 = min(y1, cy + r + 1)
            if bx0 >= bx1 or by0 >= by1:
                continue
            mask_x0 = bx0 - (cx - r)
            mask_x1 = mask_x0 + (bx1 - bx0)
            mask_y0 = by0 - (cy - r)
            mask_y1 = mask_y0 + (by1 - by0)
            brush_view = brush[mask_y0:mask_y1, mask_x0:mask_x1]
            if brush_view.size == 0:
                continue
            patch = self._map[by0:by1, bx0:bx1]
            delta = strength * brush_view
            if mode == "erase":
                patch -= delta
            else:
                patch += delta
            np.clip(patch, 0.0, 1.0, out=patch)
            self._map[by0:by1, bx0:bx1] = patch
