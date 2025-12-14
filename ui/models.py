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
    cmc_l: float
    cmc_c: float
    quantize_method: str
    keep_aspect: bool
    pipeline: str
    adaptive_weight: float
    hybrid_scale: float  # ハイブリッド時の縮小率(0.1-1.0)
    resize_method: str
    rgb_weights: tuple[float, float, float]
    edge_enhance: bool
    edge_strength: float
    edge_thickness: float
    edge_gain: float
    edge_gamma: float
    edge_saliency_weight: float
