"""UIで使用する入力パラメータのデータモデル定義。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversionRequest:
    """UIから取得した変換パラメータ一式。"""

    width: int
    height: int
    num_colors: int
    mode: str
    quantize_method: str
    division_method: str
    keep_aspect: bool
    pipeline: str
    contour_enhance: bool
    adaptive_weight: float
