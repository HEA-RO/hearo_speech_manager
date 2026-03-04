#!/usr/bin/env python3
"""
OpenAI TTS Handler Module
OpenAI TTS 핸들러 모듈

Handles text-to-speech conversion using OpenAI TTS API.
Inherits from BaseTTSHandler for common functionality.

OpenAI TTS API를 사용한 텍스트 음성 변환을 처리합니다.
공통 기능을 위해 BaseTTSHandler를 상속받습니다.
"""

import os
from typing import Tuple, Optional, Dict, Any
from openai import OpenAI

# Import base class
from .Base_TTS_Handler import BaseTTSHandler

# Import configuration
from hearo_speech_manager.config.speech_config import (
    CHUNK_DURATION_MS,
    OPENAI_API_KEY,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
    OPENAI_TTS_SPEED,
    OPENAI_TTS_SAMPLE_RATE,
    OPENAI_TTS_RESPONSE_FORMAT,
)


class OpenAI_TTS_Handler(BaseTTSHandler):
    """
    Handler for OpenAI TTS (Text-to-Speech) operations.
    OpenAI TTS (텍스트 음성 변환) 작업을 위한 핸들러입니다.

    Inherits from BaseTTSHandler and implements OpenAI-specific TTS logic.
    BaseTTSHandler를 상속받아 OpenAI 전용 TTS 로직을 구현합니다.

    Provides:
    제공 기능:
    - Text-to-speech conversion using OpenAI API
      OpenAI API를 사용한 텍스트 음성 변환
    - Audio chunking with configurable chunk duration (inherited)
      설정 가능한 청크 지속 시간의 오디오 청킹 (상속)
    - Multiple voice support (alloy, echo, fable, onyx, nova, shimmer)
      다중 음성 지원 (alloy, echo, fable, onyx, nova, shimmer)
    - Audio format conversion
      오디오 형식 변환
    """

    # Backend identifier
    BACKEND_NAME: str = "openai"
    BACKEND_TYPE: str = "online"

    # Supported voices
    SUPPORTED_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    # Supported audio formats
    SUPPORTED_FORMATS = ["mp3", "opus", "aac", "flac"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: Optional[str] = None,
    ):
        """
        Initialize OpenAI TTS Handler.

        Args:
            api_key: OpenAI API key (if None, uses environment variable or config)
            model: TTS model name (if None, uses config default)
            voice: Voice name (if None, uses config default)
            speed: Default speech speed (if None, uses config default)
            response_format: Audio format (if None, uses config default)
        """
        # Initialize base class
        super().__init__(sample_rate=OPENAI_TTS_SAMPLE_RATE, chunk_duration_ms=CHUNK_DURATION_MS)

        # OpenAI specific settings
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_TTS_MODEL
        self.voice = voice or OPENAI_TTS_VOICE
        self.speed = speed if speed is not None else OPENAI_TTS_SPEED
        self.response_format = response_format or OPENAI_TTS_RESPONSE_FORMAT

        # OpenAI client
        self.client: Optional[OpenAI] = None

        # Initialize
        self._initialized = self._initialize()

    def _initialize(self) -> bool:
        """
        Initialize OpenAI client.
        OpenAI 클라이언트를 초기화합니다.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")

            self.client = OpenAI(api_key=self.api_key)
            return True

        except Exception as e:
            print(f"[OpenAI_TTS_Handler] Failed to initialize: {str(e)}")
            return False

    def _text_to_speech_impl(
        self, text: str, voice: Optional[str] = None, speed: Optional[float] = None, **kwargs
    ) -> Tuple[bool, str, bytes]:
        """
        Convert text to speech using OpenAI TTS.
        OpenAI TTS를 사용하여 텍스트를 음성으로 변환합니다.

        Args:
            text: Text to convert to speech
            voice: Voice name (if None, uses default voice)
            speed: Speech speed (0.25 to 4.0, if None uses config default)
            **kwargs: Additional parameters
                - response_format: Audio format (mp3, opus, aac, flac)

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        if not self.client:
            return False, "OpenAI TTS client not initialized", b""

        try:
            # Use provided values or defaults
            voice_to_use = voice if voice else self.voice
            speed_to_use = speed if speed is not None else self.speed
            response_format = kwargs.get("response_format", self.response_format)

            # Validate voice
            if voice_to_use not in self.SUPPORTED_VOICES:
                print(
                    f"[OpenAI_TTS_Handler] Warning: Voice '{voice_to_use}' not in supported list, using anyway"
                )

            # Validate speed (0.25 to 4.0)
            speed_to_use = max(0.25, min(4.0, speed_to_use))

            # Validate format
            if response_format not in self.SUPPORTED_FORMATS:
                response_format = "mp3"

            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice_to_use,
                input=text,
                speed=speed_to_use,
                response_format=response_format,
            )

            # Get audio data
            audio_data = response.content

            return True, "", audio_data

        except Exception as e:
            error_message = f"OpenAI TTS API call failed: {str(e)}"
            return False, error_message, b""

    def is_available(self) -> bool:
        """
        Check if the handler is available and ready.
        핸들러가 사용 가능하고 준비되었는지 확인합니다.

        Returns:
            True if available, False otherwise
        """
        return self.client is not None and self._initialized

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about the current handler.
        현재 핸들러에 대한 정보를 반환합니다.

        Returns:
            Dictionary with handler information
        """
        # Get base info
        base_info = super().get_handler_info()

        # Add OpenAI specific info
        openai_info = {
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": self.response_format,
            "provider": "OpenAI",
            "supported_voices": self.SUPPORTED_VOICES,
            "supported_formats": self.SUPPORTED_FORMATS,
        }

        return {**base_info, **openai_info}

    def set_voice(self, voice: str) -> bool:
        """
        Set the default voice.
        기본 음성을 설정합니다.

        Args:
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            True if successful, False if invalid voice
        """
        if voice in self.SUPPORTED_VOICES:
            self.voice = voice
            return True
        else:
            print(f"[OpenAI_TTS_Handler] Invalid voice: {voice}")
            return False

    def set_speed(self, speed: float) -> bool:
        """
        Set the default speed.
        기본 속도를 설정합니다.

        Args:
            speed: Speech speed (0.25 to 4.0)

        Returns:
            True if valid speed, False otherwise
        """
        if 0.25 <= speed <= 4.0:
            self.speed = speed
            return True
        else:
            print(f"[OpenAI_TTS_Handler] Invalid speed: {speed} (must be 0.25-4.0)")
            return False

    def get_supported_voices(self) -> list:
        """
        Get list of supported voices.
        지원되는 음성 목록을 반환합니다.

        Returns:
            List of supported voice names
        """
        return self.SUPPORTED_VOICES.copy()
