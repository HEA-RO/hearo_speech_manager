#!/usr/bin/env python3
"""
Base TTS Handler Module
기본 TTS 핸들러 모듈

Abstract base class for all TTS handlers.
Defines the common interface that all TTS backends must implement.

모든 TTS 핸들러를 위한 추상 기본 클래스입니다.
모든 TTS 백엔드가 구현해야 하는 공통 인터페이스를 정의합니다.
"""

import io
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from pydub import AudioSegment

# Import configuration
from hearo_speech_manager.config.speech_config import CHUNK_DURATION_MS
from hearo_speech_manager.config.speech_utility import print_color


class BaseTTSHandler(ABC):
    """
    Abstract Base Class for TTS Handlers.
    TTS 핸들러를 위한 추상 기본 클래스입니다.

    All TTS backend handlers should inherit from this class
    and implement the required abstract methods.

    모든 TTS 백엔드 핸들러는 이 클래스를 상속받고
    필수 추상 메서드를 구현해야 합니다.
    """

    # Backend identifier (to be overridden by subclasses)
    BACKEND_NAME: str = "base"
    BACKEND_TYPE: str = "abstract"  # 'online' or 'local'

    def __init__(self, sample_rate: int = 24000, chunk_duration_ms: int = None):
        """
        Initialize Base TTS Handler.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Chunk duration in milliseconds (default from config)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = (
            chunk_duration_ms if chunk_duration_ms is not None else CHUNK_DURATION_MS
        )
        self._initialized = False

    @abstractmethod
    def _initialize(self) -> bool:
        """
        Initialize the TTS backend.
        TTS 백엔드를 초기화합니다.

        Must be implemented by subclasses.
        하위 클래스에서 반드시 구현해야 합니다.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def _text_to_speech_impl(
        self, text: str, voice: Optional[str] = None, speed: Optional[float] = None, **kwargs
    ) -> Tuple[bool, str, bytes]:
        """
        Convert text to speech (backend implementation).
        텍스트를 음성으로 변환합니다 (백엔드 구현).

        Must be implemented by subclasses.
        하위 클래스에서 반드시 구현해야 합니다.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (backend-specific)
            speed: Speech speed multiplier
            **kwargs: Additional backend-specific parameters

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        pass

    def text_to_speech(
        self, text: str, voice: Optional[str] = None, speed: Optional[float] = None, **kwargs
    ) -> Tuple[bool, str, bytes]:
        """
        Convert text to speech with automatic timing and audio duration logging.
        자동 타이밍 및 오디오 길이 로깅과 함께 텍스트를 음성으로 변환합니다.

        Template Method: delegates to _text_to_speech_impl() and logs elapsed time.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (backend-specific)
            speed: Speech speed multiplier
            **kwargs: Additional backend-specific parameters

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        start = time.perf_counter()
        success, error_msg, audio_data = self._text_to_speech_impl(
            text=text, voice=voice, speed=speed, **kwargs
        )
        elapsed = time.perf_counter() - start

        if success and audio_data:
            audio_format = kwargs.get("response_format", "mp3")
            audio_duration = self.get_total_duration_from_audio(audio_data, audio_format)
            print_color(
                f"[TTS] {self.BACKEND_NAME} | "
                f"text: {len(text)} chars | "
                f"audio: {audio_duration:.2f}s | "
                f"elapsed: {elapsed:.3f}s",
                color="GREEN",
            )
        else:
            print_color(
                f"[TTS] {self.BACKEND_NAME} | " f"FAILED | elapsed: {elapsed:.3f}s | {error_msg}",
                color="GREEN",
            )

        return success, error_msg, audio_data

    def load(self) -> bool:
        """
        Load model to device (no-op by default, override for GPU-based backends).
        모델을 디바이스에 로드합니다 (기본 no-op, GPU 기반 백엔드에서 override).

        Returns:
            True if ready
        """
        return True

    def unload(self):
        """
        Unload model from device (no-op by default, override for GPU-based backends).
        모델을 디바이스에서 언로드합니다 (기본 no-op, GPU 기반 백엔드에서 override).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the handler is available and ready.
        핸들러가 사용 가능하고 준비되었는지 확인합니다.

        Must be implemented by subclasses.
        하위 클래스에서 반드시 구현해야 합니다.

        Returns:
            True if available, False otherwise
        """
        pass

    def create_audio_chunks(
        self, audio_data: bytes, audio_format: str = "mp3", chunk_duration_ms: Optional[int] = None
    ) -> Tuple[bool, str, List[bytes], int]:
        """
        Split audio data into chunks of specified duration.
        오디오 데이터를 지정된 지속 시간의 청크로 분할합니다.

        This is a common implementation shared by all handlers.
        모든 핸들러가 공유하는 공통 구현입니다.

        Args:
            audio_data: Audio data to split
            audio_format: Audio format (mp3, wav, etc.)
            chunk_duration_ms: Chunk duration in milliseconds (if None, uses default)

        Returns:
            Tuple of (success, error_message, audio_chunks, n_chunks)
        """
        try:
            # Use provided chunk duration or default
            chunk_ms = chunk_duration_ms if chunk_duration_ms else self.chunk_duration_ms

            # Load audio from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)

            # Calculate number of chunks
            total_duration_ms = len(audio)
            n_chunks = (total_duration_ms + chunk_ms - 1) // chunk_ms  # Ceiling division

            # Split audio into chunks
            chunks = []
            for i in range(n_chunks):
                start_ms = i * chunk_ms
                end_ms = min((i + 1) * chunk_ms, total_duration_ms)

                chunk = audio[start_ms:end_ms]

                # Export chunk to bytes
                chunk_buffer = io.BytesIO()
                chunk.export(chunk_buffer, format=audio_format)
                chunk_bytes = chunk_buffer.getvalue()

                chunks.append(chunk_bytes)

            return True, "", chunks, n_chunks

        except Exception as e:
            error_message = f"Audio chunking failed: {str(e)}"
            return False, error_message, [], 0

    def text_to_speech_chunked(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        chunk_duration_ms: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, str, List[bytes], int, float]:
        """
        Convert text to speech and split into chunks.
        텍스트를 음성으로 변환하고 청크로 분할합니다.

        This is a common implementation that uses text_to_speech and create_audio_chunks.
        text_to_speech와 create_audio_chunks를 사용하는 공통 구현입니다.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (backend-specific)
            speed: Speech speed multiplier
            chunk_duration_ms: Chunk duration in milliseconds (if None, uses default)
            **kwargs: Additional backend-specific parameters

        Returns:
            Tuple of (success, error_message, audio_chunks, n_chunks, total_duration_sec)
        """
        try:
            # Generate speech
            success, error_msg, audio_data = self.text_to_speech(
                text=text, voice=voice, speed=speed, **kwargs
            )

            if not success:
                return False, error_msg, [], 0, 0.0

            # Get audio format from kwargs or use default
            audio_format = kwargs.get("response_format", "mp3")

            # Split into chunks
            success, error_msg, chunks, n_chunks = self.create_audio_chunks(
                audio_data=audio_data,
                audio_format=audio_format,
                chunk_duration_ms=chunk_duration_ms,
            )

            if not success:
                return False, error_msg, [], 0, 0.0

            # Calculate total duration
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
            total_duration_sec = len(audio) / 1000.0

            return True, "", chunks, n_chunks, total_duration_sec

        except Exception as e:
            error_message = f"Chunked TTS failed: {str(e)}"
            return False, error_message, [], 0, 0.0

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about the current handler.
        현재 핸들러에 대한 정보를 반환합니다.

        Returns:
            Dictionary with handler information
        """
        return {
            "handler_type": "TTS",
            "backend": self.BACKEND_NAME,
            "type": self.BACKEND_TYPE,
            "available": self.is_available(),
            "sample_rate": self.sample_rate,
            "chunk_duration_ms": self.chunk_duration_ms,
        }

    def get_backend_name(self) -> str:
        """
        Get the backend name.
        백엔드 이름을 반환합니다.

        Returns:
            Backend name string
        """
        return self.BACKEND_NAME

    def get_total_duration_from_audio(self, audio_data: bytes, audio_format: str = "mp3") -> float:
        """
        Calculate total duration from audio data.
        오디오 데이터에서 전체 지속 시간을 계산합니다.

        Args:
            audio_data: Audio data bytes
            audio_format: Audio format

        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
            return len(audio) / 1000.0
        except Exception:
            return 0.0

    def calculate_n_chunks(
        self, total_duration_ms: int, chunk_duration_ms: Optional[int] = None
    ) -> int:
        """
        Calculate number of chunks needed for given duration.
        주어진 지속 시간에 필요한 청크 수를 계산합니다.

        Args:
            total_duration_ms: Total duration in milliseconds
            chunk_duration_ms: Chunk duration in milliseconds

        Returns:
            Number of chunks (ceiling division)
        """
        chunk_ms = chunk_duration_ms if chunk_duration_ms else self.chunk_duration_ms
        return (total_duration_ms + chunk_ms - 1) // chunk_ms
