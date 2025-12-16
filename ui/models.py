"""UIで使用する入力パラメータのデータモデル定義。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversionRequest:
    """UIから取得した変換パラメータ一式。"""

    width: int
    height: int
    mode: str
    lab_metric: str
    cmc_l: float
    cmc_c: float
    keep_aspect: bool
    resize_method: str
    rgb_weights: tuple[float, float, float]
