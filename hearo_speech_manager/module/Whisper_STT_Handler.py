#!/usr/bin/env python3
"""
Whisper STT Handler Module
Whisper STT 핸들러 모듈

Huggingface Transformers의 Whisper pipeline을 사용한 음성-텍스트 변환을 처리합니다.
BaseSTTHandler를 상속받아 transcribe() 인터페이스를 구현합니다.

리샘플링, 모델 추론, max duration 제한이 모두 핸들러 내부에 캡슐화되어 있으며,
STTNode는 이러한 세부사항을 알 필요가 없습니다.
"""

import time
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import pipeline

from .Base_STT_Handler import BaseSTTHandler
from hearo_speech_manager.config.speech_config import (
    WHISPER_MODEL_NAME,
    WHISPER_LANGUAGE,
    WHISPER_SAMPLE_RATE,
    WHISPER_CHUNK_LENGTH_S,
    WHISPER_STRIDE_LENGTH_S,
    WHISPER_MAX_AUDIO_DURATION,
    get_stt_config,
)


class WhisperSTTHandler(BaseSTTHandler):
    """
    Huggingface Whisper 기반 STT 핸들러.

    오디오 리샘플링(입력 SR → 16kHz)과 Whisper pipeline 추론을
    캡슐화하여 transcribe(audio, sr) → str 인터페이스를 제공합니다.
    """

    BACKEND_NAME = "whisper"
    BACKEND_TYPE = "local"

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        cfg = config or get_stt_config("whisper")
        self.model_name = cfg.get("model_name", WHISPER_MODEL_NAME)
        self.language = cfg.get("language", WHISPER_LANGUAGE)
        self.target_sample_rate = cfg.get("sample_rate", WHISPER_SAMPLE_RATE)
        self.chunk_length_s = cfg.get("chunk_length_s", WHISPER_CHUNK_LENGTH_S)
        self.stride_length_s = cfg.get("stride_length_s", WHISPER_STRIDE_LENGTH_S)
        self.max_audio_duration = cfg.get("max_audio_duration", WHISPER_MAX_AUDIO_DURATION)

        self._pipe = None
        self._device = None
        self._initialized = self._initialize()

    def _initialize(self) -> bool:
        """Whisper pipeline 로드 (자동 청킹 지원)."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self._device,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=(self.stride_length_s, self.stride_length_s),
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            print(
                f"[WhisperSTTHandler] Model loaded: {self.model_name} "
                f"(device={self._device}, chunk={self.chunk_length_s}s, "
                f"stride={self.stride_length_s}s)",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"[WhisperSTTHandler] Failed to load model: {e}", flush=True)
            return False

    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        오디오를 텍스트로 변환합니다.

        1. 입력 SR → Whisper 요구 SR (16kHz) 리샘플링
        2. max duration 제한 적용
        3. Whisper pipeline 추론

        Args:
            audio_data: float32 numpy array (-1.0 ~ 1.0)
            sample_rate: 입력 샘플레이트 (예: 48000)

        Returns:
            변환된 텍스트

        Raises:
            RuntimeError: 모델 미초기화 시
        """
        if not self._initialized or self._pipe is None:
            raise RuntimeError("Whisper model not initialized")

        start_time = time.perf_counter()

        audio_resampled = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=self.target_sample_rate,
        )

        max_samples = self.max_audio_duration * self.target_sample_rate
        if len(audio_resampled) > max_samples:
            print(
                f"[WhisperSTTHandler] Audio exceeds {self.max_audio_duration}s, truncating",
                flush=True,
            )
            audio_resampled = audio_resampled[:max_samples]

        result = self._pipe(
            {"raw": audio_resampled, "sampling_rate": self.target_sample_rate},
            generate_kwargs={"language": "korean", "task": "transcribe"},
            return_timestamps=False,
        )
        transcription = result["text"].strip()

        elapsed = time.perf_counter() - start_time
        audio_duration = len(audio_data) / sample_rate
        print(
            f"[WhisperSTTHandler] {audio_duration:.2f}s audio → "
            f"'{transcription[:50]}{'...' if len(transcription) > 50 else ''}' "
            f"({elapsed:.3f}s)",
            flush=True,
        )

        return transcription

    def is_available(self) -> bool:
        """Whisper 모델 사용 가능 여부."""
        return self._initialized and self._pipe is not None
