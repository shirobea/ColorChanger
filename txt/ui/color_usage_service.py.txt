"""Color usage analysis helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from palette import BeadPalette


def build_color_usage_rows(
    image: np.ndarray,
    palette: BeadPalette,
    require_in_palette: bool,
) -> tuple[bool, list[dict]]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return False, []
    palette_map: dict[tuple[int, int, int], dict[str, str]] = {}
    for color in palette:
        rgb = tuple(int(round(v)) for v in color.rgb)
        palette_map[rgb] = {"color_id": color.color_id, "name": color.name}
    if not palette_map:
        return False, []
    flat = image.reshape(-1, 3)
    if require_in_palette:
        palette_codes = np.array(
            [(rgb[0] << 16) | (rgb[1] << 8) | rgb[2] for rgb in palette_map.keys()],
            dtype=np.uint32,
        )
        if palette_codes.size == 0:
            return False, []
        total = flat.shape[0]
        chunk_size = 200_000
        for start in range(0, total, chunk_size):
            chunk = flat[start : start + chunk_size]
            codes = (
                (chunk[:, 0].astype(np.uint32) << 16)
                | (chunk[:, 1].astype(np.uint32) << 8)
                | chunk[:, 2].astype(np.uint32)
            )
            if not np.isin(codes, palette_codes).all():
                return False, []
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    rows: list[dict] = []
    for rgb_arr, count in zip(colors, counts):
        rgb = (int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2]))
        info = palette_map.get(rgb)
        if not info:
            if require_in_palette:
                return False, []
            continue
        rows.append(
            {
                "color_id": info["color_id"],
                "name": info["name"],
                "count": int(count),
                "rgb": rgb,
            }
        )
    rows.sort(key=lambda r: int(r.get("count", 0)), reverse=True)
    return True, rows
