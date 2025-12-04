"""Image conversion pipeline: resize, quantize, map to bead palette."""

from __future__ import annotations

from .pipeline import (
    convert_image,
    ConversionCancelled,
    _PipelineConfig,
    _report,
    _apply_saliency_enhancement,
    _run_quantize_first_pipeline,
    _run_resize_first_pipeline,
    _run_hybrid_pipeline,
    _run_map_only_resize_first,
    _run_map_only_quantize_first,
    _run_map_only_hybrid,
    ProgressCb,
    CancelEvent,
    Size,
)
from .io_utils import _compute_hybrid_size, _compute_resize, _imread_unicode, _load_image_rgb
from .saliency import (
    ImportanceWeights,
    _compute_center_mask,
    _compute_eye_mask,
    _compute_face_masks,
    _compute_saliency_map,
    _compute_skin_mask,
    _normalize_saliency,
    compute_importance_map,
    compute_saliency_map,
)
from .quantize import _PaletteQuantizer, _map_centers_to_palette, _quantize, _quantize_wu
from .block_methods import (
    _adaptive_block_image,
    _dominant_block_image,
    _map_image_to_palette,
    _run_block_pipeline,
)
from palette import BeadPalette

__all__ = [
    "convert_image",
    "ConversionCancelled",
    "_PipelineConfig",
    "ProgressCb",
    "CancelEvent",
    "Size",
    "_apply_saliency_enhancement",
    "_run_quantize_first_pipeline",
    "_run_resize_first_pipeline",
    "_run_hybrid_pipeline",
    "_run_map_only_resize_first",
    "_run_map_only_quantize_first",
    "_run_map_only_hybrid",
    "_compute_hybrid_size",
    "_compute_resize",
    "_imread_unicode",
    "_load_image_rgb",
    "ImportanceWeights",
    "_compute_center_mask",
    "_compute_eye_mask",
    "_compute_face_masks",
    "_compute_saliency_map",
    "_compute_skin_mask",
    "_normalize_saliency",
    "compute_importance_map",
    "compute_saliency_map",
    "_PaletteQuantizer",
    "_map_centers_to_palette",
    "_quantize",
    "_quantize_wu",
    "_adaptive_block_image",
    "_dominant_block_image",
    "_map_image_to_palette",
    "_run_block_pipeline",
    "BeadPalette",
]
