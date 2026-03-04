#!/usr/bin/env python3
"""
LazyGPUSessionPool — 유휴 타임아웃 기반 GPU 세션 관리.

기존 gpu_session() context manager 패턴을 대체하여:
- 첫 요청 시 GPU 모델 로드 (lazy)
- 마지막 요청 후 idle_timeout 경과 시 자동 언로드
- 연속 batch 처리 시 GPU 재로드 없음 (기존 대비 성능 우수)
- 캐시 히트 시 GPU 로드 불필요
"""

import threading
import time
from typing import Optional


class LazyGPUSessionPool:
    """HeaRo_TTS_Handler의 GPU 세션을 lazy 로딩 + 아이들 타임아웃으로 관리."""

    def __init__(self, tts_handler, idle_timeout_sec: float = 30.0, logger=None):
        """
        Args:
            tts_handler: HeaRo_TTS_Handler 인스턴스
            idle_timeout_sec: 유휴 시 GPU 모델 자동 언로드 시간 (초)
            logger: ROS2 logger (optional)
        """
        self._handler = tts_handler
        self._idle_timeout = idle_timeout_sec
        self._logger = logger

        self._lock = threading.Lock()
        self._loaded = False
        self._idle_timer: Optional[threading.Timer] = None
        self._last_use_time: float = 0.0

    def _log(self, msg: str, level: str = "info"):
        if self._logger:
            getattr(self._logger, level, self._logger.info)(msg)
        else:
            print(msg, flush=True)

    def ensure_loaded(self):
        """GPU 모델이 로드되어 있지 않으면 로드. 이미 로드 상태면 idle timer 리셋."""
        with self._lock:
            self._cancel_idle_timer()

            if not self._loaded:
                self._log("[LazyGPUPool] Loading GPU models...")
                start = time.time()
                try:
                    if hasattr(self._handler, '_load_gpu_models'):
                        self._handler._load_gpu_models()
                    elif hasattr(self._handler, 'gpu_session'):
                        self._handler._gpu_loaded = True
                    self._loaded = True
                    elapsed = time.time() - start
                    self._log(f"[LazyGPUPool] GPU models loaded in {elapsed:.2f}s")
                except Exception as e:
                    self._log(f"[LazyGPUPool] GPU load failed: {e}", "error")
                    raise

            self._last_use_time = time.time()
            self._start_idle_timer()

    def mark_done(self):
        """현재 요청 완료 후 호출. idle timer를 (재)시작."""
        with self._lock:
            self._last_use_time = time.time()
            self._cancel_idle_timer()
            self._start_idle_timer()

    def release(self):
        """즉시 GPU 모델 언로드."""
        with self._lock:
            self._cancel_idle_timer()
            self._do_release()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _start_idle_timer(self):
        """idle_timeout 후 자동 언로드 타이머 시작."""
        self._idle_timer = threading.Timer(self._idle_timeout, self._on_idle_timeout)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_timer(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _on_idle_timeout(self):
        """idle timeout 경과 → GPU 모델 언로드."""
        with self._lock:
            elapsed = time.time() - self._last_use_time
            if elapsed >= self._idle_timeout and self._loaded:
                self._log(f"[LazyGPUPool] Idle timeout ({elapsed:.1f}s). Releasing GPU models...")
                self._do_release()

    def _do_release(self):
        """실제 GPU 모델 언로드 수행."""
        if not self._loaded:
            return
        try:
            if hasattr(self._handler, '_release_gpu_models'):
                self._handler._release_gpu_models()
            elif hasattr(self._handler, 'release_tts_gpu'):
                self._handler.release_tts_gpu()

            import gc
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

            self._loaded = False
            self._log("[LazyGPUPool] GPU models released.")
        except Exception as e:
            self._log(f"[LazyGPUPool] GPU release error: {e}", "error")
