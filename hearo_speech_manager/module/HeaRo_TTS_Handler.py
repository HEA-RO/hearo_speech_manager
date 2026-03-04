#!/usr/bin/env python3
"""
HeaRo TTS Handler Module
HeaRo TTS 핸들러 모듈

Unified handler for TTS operations supporting multiple backends.
Provides automatic backend selection and fallback mechanisms.
Similar architecture to HeaRo_LLM_Distributor for consistency.

다중 백엔드를 지원하는 TTS 작업을 위한 통합 핸들러입니다.
자동 백엔드 선택 및 폴백 메커니즘을 제공합니다.
일관성을 위해 HeaRo_LLM_Distributor와 유사한 아키텍처를 사용합니다.
"""

import os
import threading
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any

# Import base class
from .Base_TTS_Handler import BaseTTSHandler

# Import specific handlers
from .OpenAI_TTS_Handler import OpenAI_TTS_Handler

# Try to import Zonos handler (optional, may not be available)
try:
    from .Zonos_TTS_Handler import Zonos_TTS_Handler

    ZONOS_AVAILABLE = True
except ImportError:
    ZONOS_AVAILABLE = False

# Import configuration
from hearo_speech_manager.config.speech_config import (
    CHUNK_DURATION_MS,
    DEFAULT_TTS_MODEL,
    TTS_MODEL_OPENAI,
    TTS_MODEL_AUTO,
    OPENAI_API_KEY,
    get_tts_config,
)


class HeaRo_TTS_Handler:
    """
    Unified TTS Handler for HeaRo system.
    HeaRo 시스템을 위한 통합 TTS 핸들러입니다.

    Similar to HeaRo_LLM_Distributor, this class provides:
    HeaRo_LLM_Distributor와 유사하게, 이 클래스는 다음을 제공합니다:

    - Automatic TTS backend selection based on configuration
      설정 기반 자동 TTS 백엔드 선택
    - Support for multiple TTS backends (OpenAI, future: Zonos)
      다중 TTS 백엔드 지원 (OpenAI, 향후: Zonos)
    - Audio chunking with configurable duration
      설정 가능한 지속 시간의 오디오 청킹
    - Fallback mechanism when preferred backend unavailable
      선호 백엔드를 사용할 수 없을 때의 폴백 메커니즘
    - Model name tracking for database storage
      데이터베이스 저장을 위한 모델 이름 추적

    Backend Types:
    백엔드 타입:
    - 'openai': OpenAI TTS API (online)
    - 'zonos_transformer': Zonos Transformer model (local, future)
    - 'zonos_hybrid': Zonos Hybrid model (local, future)
    - 'auto': Automatic selection based on priority
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chunk_duration_ms: Optional[int] = None,
        default_backend: Optional[str] = None,
    ):
        """
        Initialize HeaRo TTS Handler.

        Args:
            openai_api_key: OpenAI API key (if None, uses config)
            chunk_duration_ms: Chunk duration in milliseconds (default from config)
            default_backend: Default backend to use (if None, uses config)
        """
        # Configuration
        self.chunk_duration_ms = (
            chunk_duration_ms if chunk_duration_ms is not None else CHUNK_DURATION_MS
        )
        self.default_backend = default_backend or DEFAULT_TTS_MODEL
        self.openai_api_key = openai_api_key or OPENAI_API_KEY

        # Handler registry
        self.handlers: Dict[str, BaseTTSHandler] = {}

        # Track last used backend for model name retrieval
        self._last_used_backend: Optional[str] = None

        # GPU Session: scoped lifecycle management via context manager
        self._gpu_lock = threading.Lock()
        self._active_sessions: int = 0

        # GPU Inference Lock: serializes actual model.generate() calls
        # across concurrent threads (ReentrantCallbackGroup safety).
        # Separate from _gpu_lock which only protects load/unload lifecycle.
        self._inference_lock = threading.Lock()

        # Initialize handlers
        self._initialize_handlers()

    def _initialize_handlers(self):
        """
        Initialize TTS handlers based on configuration.
        설정에 기반하여 TTS 핸들러를 초기화합니다.
        """
        # Get TTS configuration
        tts_config = get_tts_config()

        # Initialize OpenAI TTS handler
        try:
            openai_config = tts_config.get("openai", {})
            self.handlers["openai"] = OpenAI_TTS_Handler(
                api_key=self.openai_api_key,
                model=openai_config.get("model"),
                voice=openai_config.get("voice"),
                speed=openai_config.get("speed"),
                response_format=openai_config.get("response_format"),
            )
            if self.handlers["openai"].is_available():
                print(f"[HeaRo_TTS_Handler] ✓ OpenAI TTS handler initialized", flush=True)
            else:
                print(
                    f"[HeaRo_TTS_Handler] ⚠ OpenAI TTS handler initialized but not available",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[HeaRo_TTS_Handler] ✗ Failed to initialize OpenAI TTS handler: {str(e)}",
                flush=True,
            )

        # Initialize Zonos handlers
        self._initialize_zonos_handlers(tts_config)

    def _initialize_zonos_handlers(self, tts_config: dict):
        """
        Initialize Zonos TTS handlers in lazy mode (no GPU model loading).
        Zonos TTS 핸들러를 lazy 모드로 초기화합니다 (GPU 모델 로딩 없음).

        Handlers are created with lazy=True so the model is NOT loaded on GPU.
        Call load_tts() to actually load the model when TTS is needed.

        핸들러는 lazy=True로 생성되므로 모델이 GPU에 로드되지 않습니다.
        TTS가 필요할 때 load_tts()를 호출하여 실제로 모델을 로드하세요.

        Args:
            tts_config: TTS configuration dictionary
        """
        if not ZONOS_AVAILABLE:
            print(
                f"[HeaRo_TTS_Handler] ⚠ Zonos TTS handler not available (package not installed)",
                flush=True,
            )
            return

        from hearo_speech_manager.config.speech_config import (
            DEFAULT_TTS_MODEL,
            TTS_MODEL_ZONOS_TRANSFORMER,
            TTS_MODEL_ZONOS_HYBRID,
        )

        # Create Zonos handler in lazy mode (no GPU model loading at init)
        # lazy 모드로 Zonos 핸들러 생성 (init 시 GPU 모델 로딩 없음)

        if DEFAULT_TTS_MODEL == TTS_MODEL_ZONOS_TRANSFORMER:
            try:
                zonos_transformer_config = tts_config.get("zonos_transformer", {})
                self.handlers["zonos_transformer"] = Zonos_TTS_Handler(
                    model_type="transformer",
                    model_name=zonos_transformer_config.get("model_name"),
                    device=zonos_transformer_config.get("device"),
                    speaker_id=zonos_transformer_config.get("speaker_id"),
                    speed=zonos_transformer_config.get("speed"),
                    pitch=zonos_transformer_config.get("pitch"),
                    emotion=zonos_transformer_config.get("emotion"),
                    use_reference=zonos_transformer_config.get("use_reference"),
                    reference_audio_path=zonos_transformer_config.get("reference_audio_path"),
                    lazy=True,
                )
                print(
                    f"[HeaRo_TTS_Handler] ✓ Zonos Transformer TTS handler registered (lazy, not loaded on GPU)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[HeaRo_TTS_Handler] ✗ Failed to register Zonos Transformer TTS handler: {str(e)}",
                    flush=True,
                )

        elif DEFAULT_TTS_MODEL == TTS_MODEL_ZONOS_HYBRID:
            try:
                zonos_hybrid_config = tts_config.get("zonos_hybrid", {})
                self.handlers["zonos_hybrid"] = Zonos_TTS_Handler(
                    model_type="hybrid",
                    model_name=zonos_hybrid_config.get("model_name"),
                    device=zonos_hybrid_config.get("device"),
                    speaker_id=zonos_hybrid_config.get("speaker_id"),
                    speed=zonos_hybrid_config.get("speed"),
                    pitch=zonos_hybrid_config.get("pitch"),
                    emotion=zonos_hybrid_config.get("emotion"),
                    use_reference=zonos_hybrid_config.get("use_reference"),
                    reference_audio_path=zonos_hybrid_config.get("reference_audio_path"),
                    lazy=True,
                )
                print(
                    f"[HeaRo_TTS_Handler] ✓ Zonos Hybrid TTS handler registered (lazy, not loaded on GPU)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[HeaRo_TTS_Handler] ✗ Failed to register Zonos Hybrid TTS handler: {str(e)}",
                    flush=True,
                )
        else:
            print(
                f"[HeaRo_TTS_Handler] ℹ Zonos not registered (DEFAULT_TTS_MODEL={DEFAULT_TTS_MODEL})",
                flush=True,
            )

    @contextmanager
    def gpu_session(self):
        """
        Context manager for scoped GPU model lifecycle.
        GPU 모델 생명주기를 구조적으로 보장하는 Context Manager입니다.

        토픽 처리 단위로 사용하여 정확히 1회 로드 / 1회 해제를 보장합니다.
        중첩(nested) 또는 동시(concurrent) 세션을 안전하게 지원합니다.

        Usage:
            with tts_handler.gpu_session():
                # 모델 로드됨 (이미 로드 상태면 no-op)
                result = tts_handler.text_to_speech(...)
            # 여기서 반드시 해제 (예외 발생해도)
        """
        with self._gpu_lock:
            self._active_sessions += 1
            if self._active_sessions == 1:
                self._load_default_model()
        try:
            yield
        finally:
            with self._gpu_lock:
                self._active_sessions -= 1
                if self._active_sessions == 0:
                    self._unload_all_zonos()

    def _load_default_model(self):
        """
        Load the DEFAULT_TTS_MODEL to GPU.
        DEFAULT_TTS_MODEL을 GPU에 로드합니다.

        Called internally by gpu_session().__enter__ when first session starts.
        첫 번째 세션이 시작될 때 gpu_session().__enter__에서 내부적으로 호출됩니다.
        """
        from hearo_speech_manager.config.speech_config import (
            get_default_tts_model,
            TTS_MODEL_ZONOS_TRANSFORMER,
            TTS_MODEL_ZONOS_HYBRID,
        )
        from hearo_speech_manager.config.speech_utility import print_color

        current_model = get_default_tts_model()

        if current_model == TTS_MODEL_OPENAI or current_model == TTS_MODEL_AUTO:
            return

        backend_key = None
        if current_model == TTS_MODEL_ZONOS_TRANSFORMER:
            backend_key = "zonos_transformer"
        elif current_model == TTS_MODEL_ZONOS_HYBRID:
            backend_key = "zonos_hybrid"

        if backend_key and backend_key in self.handlers:
            handler = self.handlers[backend_key]
            if not handler._initialized or handler.model is None:
                success = handler.load_model()
                if not success:
                    print_color(
                        f"[TTS] GPU Session: {backend_key} model loading FAILED", color="RED"
                    )
        elif backend_key and backend_key not in self.handlers:
            tts_config = get_tts_config()
            handler_config = tts_config.get(backend_key, {})
            model_type = "hybrid" if "hybrid" in backend_key else "transformer"
            try:
                self.handlers[backend_key] = Zonos_TTS_Handler(
                    model_type=model_type,
                    model_name=handler_config.get("model_name"),
                    device=handler_config.get("device"),
                    speaker_id=handler_config.get("speaker_id"),
                    speed=handler_config.get("speed"),
                    pitch=handler_config.get("pitch"),
                    emotion=handler_config.get("emotion"),
                    use_reference=handler_config.get("use_reference"),
                    reference_audio_path=handler_config.get("reference_audio_path"),
                    lazy=False,
                )
            except Exception as e:
                print(f"[HeaRo_TTS_Handler] Failed to create {backend_key}: {str(e)}", flush=True)

    def _unload_all_zonos(self):
        """
        Unload all Zonos TTS models from GPU to free VRAM.
        모든 Zonos TTS 모델을 GPU에서 언로드하여 VRAM을 해제합니다.

        Called internally by gpu_session().__exit__ when last session ends.
        마지막 세션이 종료될 때 gpu_session().__exit__에서 내부적으로 호출됩니다.
        """
        for key, handler in self.handlers.items():
            if key.startswith("zonos") and hasattr(handler, "unload_model"):
                if handler._initialized or (
                    hasattr(handler, "model") and handler.model is not None
                ):
                    handler.unload_model()

    def release_tts_gpu(self):
        """
        Force-release all GPU resources (for node shutdown).
        GPU 자원을 강제 해제합니다 (노드 종료 시 사용).

        Unlike gpu_session().__exit__, this ignores active session count
        and unconditionally unloads all models.
        gpu_session().__exit__와 달리, 활성 세션 수를 무시하고
        모든 모델을 무조건 언로드합니다.
        """
        with self._gpu_lock:
            self._active_sessions = 0
            for key, handler in self.handlers.items():
                if key.startswith("zonos") and hasattr(handler, "unload_model"):
                    if handler._initialized or (
                        hasattr(handler, "model") and handler.model is not None
                    ):
                        handler.unload_model()

    def register_handler(self, backend_name: str, handler: BaseTTSHandler) -> bool:
        """
        Register a custom TTS handler.
        커스텀 TTS 핸들러를 등록합니다.

        Args:
            backend_name: Name to identify this backend
            handler: Handler instance (must inherit from BaseTTSHandler)

        Returns:
            True if registered successfully, False otherwise
        """
        if not isinstance(handler, BaseTTSHandler):
            print(f"[HeaRo_TTS_Handler] Handler must inherit from BaseTTSHandler", flush=True)
            return False

        self.handlers[backend_name] = handler

        # Add to priority if not exists
        if backend_name not in self.backend_priority:
            self.backend_priority.append(backend_name)

        print(f"[HeaRo_TTS_Handler] Registered handler: {backend_name}", flush=True)
        return True

    def _select_backend(self, requested_backend: str) -> Optional[str]:
        """
        Select an available backend based on request.
        요청에 따라 사용 가능한 백엔드를 선택합니다.

        Args:
            requested_backend: Requested backend type or 'auto'

        Returns:
            Selected backend name or None if no backend available
        """
        # gpu_session() guarantees the model is loaded before this is called.
        # No auto-loading here — if model isn't loaded, fall back to available backend.
        # gpu_session()이 이 메서드 호출 전에 모델 로드를 보장합니다.
        # 자동 로딩 없음 — 모델이 로드되지 않은 경우 사용 가능한 백엔드로 폴백합니다.

        if requested_backend != "auto" and requested_backend != TTS_MODEL_AUTO:
            if requested_backend in self.handlers:
                handler = self.handlers[requested_backend]
                if handler.is_available():
                    return requested_backend
                print(
                    f"[HeaRo_TTS_Handler] Requested backend '{requested_backend}' not available, trying fallback...",
                    flush=True,
                )
            else:
                print(
                    f"[HeaRo_TTS_Handler] Unknown backend '{requested_backend}', trying fallback...",
                    flush=True,
                )

        from hearo_speech_manager.config.speech_config import DEFAULT_TTS_MODEL

        if DEFAULT_TTS_MODEL in self.handlers:
            handler = self.handlers[DEFAULT_TTS_MODEL]
            if handler.is_available():
                return DEFAULT_TTS_MODEL

        # Fallback: try any available backend
        for backend_name, handler in self.handlers.items():
            if handler.is_available():
                print(f"[HeaRo_TTS_Handler] Using fallback backend: {backend_name}", flush=True)
                return backend_name

        return None

    def text_to_speech(
        self,
        text: str,
        backend: str = "auto",
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bool, str, bytes, str]:
        """
        Convert text to speech.
        텍스트를 음성으로 변환합니다.

        Args:
            text: Text to convert to speech
            backend: Backend to use ('openai', 'zonos_transformer', 'zonos_hybrid', 'auto')
            voice: Voice name (backend-specific, if None uses config default)
            speed: Speech speed (backend-specific, if None uses config default)
            **kwargs: Additional backend-specific parameters

        Returns:
            Tuple of (success, error_message, audio_data, model_used)
            - model_used: Name of the TTS model used (for database storage)
        """
        try:
            # Select backend
            selected_backend = self._select_backend(backend)

            if not selected_backend:
                return False, "No TTS backend available", b"", ""

            self._last_used_backend = selected_backend
            handler = self.handlers[selected_backend]

            # Serialize GPU inference across concurrent threads
            with self._inference_lock:
                success, error_msg, audio_data = handler.text_to_speech(
                    text=text, voice=voice, speed=speed, **kwargs
                )

            # Get model name for database storage
            model_used = self._get_model_name(selected_backend)

            return success, error_msg, audio_data, model_used

        except Exception as e:
            error_message = f"TTS failed: {str(e)}"
            return False, error_message, b"", ""

    def text_to_speech_chunked(
        self,
        text: str,
        backend: str = "auto",
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        chunk_duration_ms: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, str, List[bytes], int, float, str]:
        """
        Convert text to speech and split into chunks.
        텍스트를 음성으로 변환하고 청크로 분할합니다.

        Args:
            text: Text to convert to speech
            backend: Backend to use ('openai', 'zonos_transformer', 'zonos_hybrid', 'auto')
            voice: Voice name (backend-specific, if None uses config default)
            speed: Speech speed (backend-specific, if None uses config default)
            chunk_duration_ms: Chunk duration in milliseconds
            **kwargs: Additional backend-specific parameters

        Returns:
            Tuple of (success, error_message, audio_chunks, n_chunks, total_duration_sec, model_used)
            - model_used: Name of the TTS model used (for database storage)
        """
        try:
            # Select backend
            selected_backend = self._select_backend(backend)

            if not selected_backend:
                return False, "No TTS backend available", [], 0, 0.0, ""

            self._last_used_backend = selected_backend
            handler = self.handlers[selected_backend]

            # Use provided chunk duration or default
            chunk_ms = chunk_duration_ms if chunk_duration_ms else self.chunk_duration_ms

            # Serialize GPU inference across concurrent threads
            with self._inference_lock:
                success, error_msg, chunks, n_chunks, duration = handler.text_to_speech_chunked(
                    text=text, voice=voice, speed=speed, chunk_duration_ms=chunk_ms, **kwargs
                )

            # Get model name for database storage
            model_used = self._get_model_name(selected_backend)

            return success, error_msg, chunks, n_chunks, duration, model_used

        except Exception as e:
            error_message = f"TTS chunking failed: {str(e)}"
            return False, error_message, [], 0, 0.0, ""

    def _get_model_name(self, backend_name: str) -> str:
        """
        Get the model name for a given backend.
        주어진 백엔드의 모델 이름을 반환합니다.

        Args:
            backend_name: Backend name

        Returns:
            Model name string for database storage (uses constants from configure.py)
        """
        # Return the backend name directly (matches TTS_MODEL_* constants in configure.py)
        # These are: 'openai', 'zonos_transformer', 'zonos_hybrid'
        return backend_name

    def get_last_used_model(self) -> str:
        """
        Get the model name of the last TTS operation.
        마지막 TTS 작업의 모델 이름을 반환합니다.

        Returns:
            Model name string
        """
        if self._last_used_backend:
            return self._get_model_name(self._last_used_backend)
        return ""

    def get_available_backends(self) -> List[str]:
        """
        Get list of available TTS backends.
        사용 가능한 TTS 백엔드 목록을 반환합니다.

        Returns:
            List of available backend names
        """
        available = []
        for backend_name, handler in self.handlers.items():
            if handler.is_available():
                available.append(backend_name)
        return available

    def get_all_backends(self) -> List[str]:
        """
        Get list of all registered backends (available or not).
        등록된 모든 백엔드 목록을 반환합니다 (사용 가능 여부 무관).

        Returns:
            List of all backend names
        """
        return list(self.handlers.keys())

    def get_handler(self, backend: str) -> Optional[BaseTTSHandler]:
        """
        Get a specific handler by backend name.
        백엔드 이름으로 특정 핸들러를 반환합니다.

        Args:
            backend: Backend name

        Returns:
            Handler instance or None if not found
        """
        return self.handlers.get(backend)

    def get_handler_info(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about TTS handler(s).
        TTS 핸들러에 대한 정보를 반환합니다.

        Args:
            backend: Specific backend name or None for all

        Returns:
            Dictionary with handler information
        """
        if backend and backend in self.handlers:
            return self.handlers[backend].get_handler_info()

        # Return info for all handlers
        return {
            "default_backend": self.default_backend,
            "backend_priority": self.backend_priority,
            "available_backends": self.get_available_backends(),
            "chunk_duration_ms": self.chunk_duration_ms,
            "handlers": {
                backend_name: handler.get_handler_info()
                for backend_name, handler in self.handlers.items()
            },
        }

    def set_default_backend(self, backend: str) -> bool:
        """
        Set the default backend.
        기본 백엔드를 설정합니다.

        Args:
            backend: Backend name

        Returns:
            True if successful, False if backend not available
        """
        if backend in self.handlers:
            self.default_backend = backend
            return True
        return False

    def is_available(self) -> bool:
        """
        Check if at least one TTS backend is available.
        최소 하나의 TTS 백엔드가 사용 가능한지 확인합니다.

        Returns:
            True if at least one backend is available
        """
        return len(self.get_available_backends()) > 0
