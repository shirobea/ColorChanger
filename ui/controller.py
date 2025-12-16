"""変換実行のワーカー管理を担うコントローラ。"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import converter
from palette import BeadPalette
from .models import ConversionRequest


class ConversionRunner:
    """UIとは疎結合に変換スレッドを管理する。"""

    def __init__(
        self,
        schedule_ui: Callable[[int, Callable[..., None], object], None],
        is_closing: Callable[[], bool],
    ) -> None:
        self._schedule_ui = schedule_ui
        self._is_closing = is_closing
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(
        self,
        request: ConversionRequest,
        input_path: str,
        palette: BeadPalette,
        on_progress: Callable[[float], None],
        on_success: Callable[[object], None],
        on_cancelled: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> bool:
        """変換スレッドを開始。すでに実行中ならFalseを返す。"""
        if self.is_running:
            return False
        self._cancel_event = threading.Event()

        def _progress_cb(value: float) -> None:
            self._schedule_ui(0, on_progress, value)

        def _worker() -> None:
            try:
                result = converter.convert_image(
                    input_path=input_path,
                    output_size=(request.width, request.height),
                    mode=request.mode,
                    lab_metric=request.lab_metric,
                    cmc_l=request.cmc_l,
                    cmc_c=request.cmc_c,
                    palette=palette,
                    keep_aspect=request.keep_aspect,
                    resize_method=request.resize_method,
                    rgb_weights=request.rgb_weights,
                    progress_callback=_progress_cb,
                    cancel_event=self._cancel_event,
                )
            except converter.ConversionCancelled:
                self._schedule_ui(0, on_cancelled)
                return
            except Exception as exc:  # pragma: no cover - GUI依存
                self._schedule_ui(0, on_error, exc)
                return

            if self._is_closing():
                return
            self._schedule_ui(0, on_success, result)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def cancel_and_wait(self, timeout: float = 2.0) -> None:
        self.cancel()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._cancel_event = None
