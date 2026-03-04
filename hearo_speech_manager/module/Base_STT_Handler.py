#!/usr/bin/env python3
"""
Base STT Handler Module
기본 STT 핸들러 모듈

모든 STT 백엔드 핸들러를 위한 추상 기본 클래스입니다.
TTS의 BaseTTSHandler와 동일한 패턴으로, 각 STT 백엔드가
구현해야 하는 공통 인터페이스를 정의합니다.

배치(batch) 전용 — 발화 종료 후 한번에 변환하는 방식만 지원합니다.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np

from hearo_speech_manager.config.speech_config import STT_INPUT_SAMPLE_RATE


class BaseSTTHandler(ABC):
    """
    모든 STT 핸들러를 위한 추상 기본 클래스.

    구현체는 transcribe()에서 오디오 전처리(리샘플링 등)와
    모델 추론을 캡슐화해야 합니다.
    STTNode의 ROS2 토픽/제어 로직에 대한 의존 없이
    순수하게 audio → text 변환만 수행합니다.
    """

    BACKEND_NAME: str = "base"
    BACKEND_TYPE: str = "abstract"  # 'online' or 'local'

    def __init__(self, input_sample_rate: int = STT_INPUT_SAMPLE_RATE):
        self.input_sample_rate = input_sample_rate
        self._initialized = False

    @abstractmethod
    def _initialize(self) -> bool:
        """
        백엔드 초기화 (모델 로딩 또는 API 클라이언트 생성).

        Returns:
            초기화 성공 여부
        """
        pass

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        배치 transcription 수행.

        Args:
            audio_data: float32 numpy array (-1.0 ~ 1.0 범위)
            sample_rate: 입력 오디오 샘플레이트 (예: 48000)

        Returns:
            변환된 텍스트 문자열

        Raises:
            RuntimeError: 모델 미초기화 또는 추론 실패 시
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """백엔드 사용 가능 여부 확인."""
        pass

    def get_handler_info(self) -> Dict[str, Any]:
        """핸들러 상태 정보 반환."""
        return {
            "handler_type": "STT",
            "backend": self.BACKEND_NAME,
            "type": self.BACKEND_TYPE,
            "available": self.is_available(),
            "input_sample_rate": self.input_sample_rate,
        }

    def get_backend_name(self) -> str:
        """백엔드 식별자 반환."""
        return self.BACKEND_NAME


def create_stt_handler(
    backend: str, config: Optional[dict] = None
) -> BaseSTTHandler:
    """
    STT 백엔드명으로 핸들러 인스턴스를 생성하는 Factory Function.

    새 백엔드 추가 시 elif 분기 1개만 추가하면 됩니다.
    Lazy import를 사용하여 불필요한 의존성 로딩을 방지합니다.

    Args:
        backend: 백엔드 식별자 ("whisper", "clova", ...)
        config: 백엔드별 설정 딕셔너리 (None이면 speech_config에서 자동 로드)

    Returns:
        초기화된 BaseSTTHandler 구현체

    Raises:
        ValueError: 알 수 없는 백엔드 지정 시
    """
    if backend == "whisper":
        from .Whisper_STT_Handler import WhisperSTTHandler
        return WhisperSTTHandler(config=config)
    # elif backend == "clova":
    #     from .Clova_STT_Handler import ClovaSTTHandler
    #     return ClovaSTTHandler(config=config)
    else:
        available = ["whisper"]
        raise ValueError(
            f"Unknown STT backend: '{backend}'. Available: {available}"
        )
