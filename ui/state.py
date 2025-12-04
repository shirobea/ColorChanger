"""ウィンドウ位置・サイズの保存/復元など状態管理をまとめるMixin。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import tkinter as tk

if TYPE_CHECKING:
    from .app import BeadsApp


class StateMixin:
    def _on_window_configure(self: "BeadsApp", event: tk.Event) -> None:
        """ウィンドウが動いたりリサイズされたら現在位置を覚えておく。"""
        if event.widget is not self.root:
            return
        self._last_geometry = (
            self.root.winfo_width(),
            self.root.winfo_height(),
            self.root.winfo_x(),
            self.root.winfo_y(),
        )

    def _load_window_state(self: "BeadsApp") -> bool:
        """前回終了時のウィンドウ配置を読み込む。"""
        try:
            if not self._window_state_path.exists():
                return False
            data = json.loads(self._window_state_path.read_text(encoding="utf-8"))
            width = int(data.get("width", 0))
            height = int(data.get("height", 0))
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            if width > 0 and height > 0:
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                self._last_geometry = (width, height, x, y)
                return True
        except Exception:
            return False
        return False

    # --- 設定の永続化 ---
    def _load_settings(self: "BeadsApp") -> None:
        """前回変換時の設定差分を復元する。"""
        try:
            if not hasattr(self, "_settings_path"):
                return
            if not self._settings_path.exists():
                return
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
            self.last_settings = data.get("last_settings")
            self.prev_settings = data.get("prev_settings")
        except Exception:
            # 復元に失敗しても起動は続ける
            self.last_settings = None
            self.prev_settings = None

    def _save_settings(self: "BeadsApp") -> None:
        """現在の設定差分を保存する。"""
        try:
            if not hasattr(self, "_settings_path"):
                return
            payload = {
                "last_settings": self.last_settings,
                "prev_settings": self.prev_settings,
            }
            self._settings_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            # 保存失敗は致命的ではないので無視
            pass

    def _save_window_state(self: "BeadsApp") -> None:
        """直近に記録したウィンドウ配置をファイルへ保存する。"""
        if self._last_geometry is None:
            self._last_geometry = (
                self.root.winfo_width(),
                self.root.winfo_height(),
                self.root.winfo_x(),
                self.root.winfo_y(),
            )
        width, height, x, y = self._last_geometry
        payload = {
            "width": int(width),
            "height": int(height),
            "x": int(x),
            "y": int(y),
        }
        try:
            self._window_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self: "BeadsApp") -> None:
        """終了時に位置・サイズを保存してから閉じる。"""
        self._save_window_state()
        self.root.destroy()
