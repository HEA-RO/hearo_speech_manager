#!/usr/bin/env python3
"""
HeaRo Speech Manager Configuration Module
HeaRo 음성 관리 시스템 설정 모듈

hearo_knowledge_manager/config/configure.py에서 TTS/Audio 관련 설정을 독립 복사하여
hearo_speech_manager가 자체 완결적으로 동작하도록 합니다.

configure.py 원본은 그대로 유지됩니다 (backward compatibility).
"""

import os
from typing import Optional

# =============================================================================
# ROS2 Configuration
# =============================================================================

QUEUE_SIZE = 150

# =============================================================================
# Audio Configuration
# =============================================================================

CHUNK_DURATION_MS = 200

AUDIO_SAMPLE_RATE_24KHZ = 24000
AUDIO_SAMPLE_RATE_48KHZ = 48000

CHUNK_SIZE_24KHZ = int(AUDIO_SAMPLE_RATE_24KHZ * CHUNK_DURATION_MS / 1000)
CHUNK_SIZE_48KHZ = int(AUDIO_SAMPLE_RATE_48KHZ * CHUNK_DURATION_MS / 1000)

# "Length_Chunking" | "Sentence_Chunking" | "None"
MODE_AUDIOCHUNK = "None"

# =============================================================================
# Database Configuration (for tb_audioCache direct access)
# =============================================================================

DB_HOST = os.getenv("HEARO_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("HEARO_DB_PORT", "3306"))
DB_NAME = os.getenv("HEARO_DB_NAME", "HeaRo")
DB_USER = os.getenv("HEARO_DB_USER", "HeaRo")
DB_PASSWORD = os.getenv("HEARO_DB_PASSWORD", "yslim_005064")
DB_CHARSET = "utf8mb4"

# =============================================================================
# OpenAI Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "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",
)

# =============================================================================
# TTS Configuration (Multi-Backend Support)
# =============================================================================

TTS_MODEL_OPENAI = "openai"
TTS_MODEL_ZONOS_TRANSFORMER = "zonos_transformer"
TTS_MODEL_ZONOS_HYBRID = "zonos_hybrid"
TTS_MODEL_AUTO = "auto"

DEFAULT_TTS_MODEL = TTS_MODEL_OPENAI

FORCE_TTS_REGENERATION = False

# --- OpenAI TTS ---
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "alloy"
OPENAI_TTS_SPEED = 1.0
OPENAI_TTS_SAMPLE_RATE = AUDIO_SAMPLE_RATE_24KHZ
OPENAI_TTS_RESPONSE_FORMAT = "mp3"

# --- Zonos TTS ---
ZONOS_TTS_MODEL_TYPE = "transformer"
ZONOS_TTS_MODEL_NAME = "Zyphra/Zonos-v0.1-transformer"
ZONOS_TTS_HYBRID_MODEL_NAME = "Zyphra/Zonos-v0.1-hybrid"
ZONOS_TTS_DEVICE = "cuda"
ZONOS_TTS_SPEAKER_ID = 0
ZONOS_TTS_SPEED = 1.15
ZONOS_TTS_PITCH = 1.0
ZONOS_TTS_EMOTION = "happy"
ZONOS_TTS_SAMPLE_RATE = AUDIO_SAMPLE_RATE_24KHZ
ZONOS_TTS_USE_REFERENCE = True
ZONOS_TTS_REFERENCE_AUDIO_PATH = "resource/audio_reference/voice_reference.wav"

# Zonos generation parameters
ZONOS_TTS_MAX_NEW_TOKENS = 6000
ZONOS_TTS_CFG_SCALE = 2.0
ZONOS_TTS_LANGUAGE = "ko"

# Long text handling
ZONOS_TTS_AUTO_SPLIT_LONG_TEXT = True
ZONOS_TTS_MAX_TEXT_LENGTH = 120
ZONOS_TTS_SPLIT_SENTENCES = True
ZONOS_TTS_CROSSFADE_MS = 0
ZONOS_TTS_SEGMENT_GAP_MS = 20
ZONOS_TTS_SPEAKER_CONTINUITY = True

# Context-Aware Segmenter (Stage 1)
ZONOS_TTS_OVERLAP_ENABLED = True
ZONOS_TTS_OVERLAP_SENTENCES = 1
ZONOS_TTS_MIN_SEGMENT_LENGTH = 30

# Signal-Aware Prefix Trim (Stage 2)
ZONOS_TTS_TRIM_SEARCH_WINDOW = 0.20
ZONOS_TTS_TRIM_FRAME_MS = 20
ZONOS_TTS_TRIM_MIN_VALLEY_MS = 60
ZONOS_TTS_PUNCTUATION_WEIGHTS = {
    ".": 3.0, "!": 3.0, "?": 3.0,
    "。": 3.0, "！": 3.0, "？": 3.0,
    ",": 1.5, "、": 1.5,
    " ": 0.3,
}

# Spectral Assembler (Stage 3)
ZONOS_TTS_SPECTRAL_CROSSFADE_ENABLED = True
ZONOS_TTS_SPECTRAL_CROSSFADE_MS = 50
ZONOS_TTS_RMS_NORMALIZE = True

# Emotion vector: [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
ZONOS_TTS_EMOTION_VECTOR = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8]
ZONOS_TTS_USE_EMOTION_VECTOR = True

ZONOS_EMOTION_PRESETS = {
    "neutral":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "happy":     [0.7, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1],
    "sad":       [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
    "angry":     [0.0, 0.0, 0.1, 0.0, 0.0, 0.7, 0.1, 0.1],
    "surprised": [0.1, 0.0, 0.0, 0.1, 0.7, 0.0, 0.1, 0.0],
    "fearful":   [0.0, 0.1, 0.0, 0.7, 0.1, 0.0, 0.1, 0.0],
    "disgusted": [0.0, 0.0, 0.7, 0.0, 0.0, 0.1, 0.1, 0.1],
    "excited":   [0.4, 0.0, 0.0, 0.0, 0.4, 0.0, 0.1, 0.1],
    "calm":      [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8],
}


# =============================================================================
# STT Configuration (Multi-Backend Support)
# =============================================================================

STT_BACKEND_WHISPER = "whisper"
STT_BACKEND_CLOVA = "clova"

DEFAULT_STT_BACKEND = os.getenv("HEARO_STT_BACKEND", STT_BACKEND_WHISPER)

# STT Common
STT_INPUT_SAMPLE_RATE = 48000

# --- Whisper STT ---
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
WHISPER_LANGUAGE = "ko"
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_LENGTH_S = 25
WHISPER_STRIDE_LENGTH_S = 5
WHISPER_MAX_AUDIO_DURATION = 300
WHISPER_MIN_AUDIO_DURATION = 0.3

# --- CLOVA Speech STT (placeholder) ---
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY", "")
CLOVA_API_URL = os.getenv("CLOVA_API_URL", "")
CLOVA_LANGUAGE = "ko"

# =============================================================================
# Helper Functions
# =============================================================================


def get_chunk_duration_ms() -> int:
    """오디오 청크 지속 시간 (ms) 반환."""
    return CHUNK_DURATION_MS


def get_chunk_size(sample_rate: int) -> int:
    """주어진 샘플 레이트에 대한 청크 크기 (samples) 계산."""
    return int(sample_rate * CHUNK_DURATION_MS / 1000)


def get_n_chunks(total_duration_ms: int, chunk_duration_ms: Optional[int] = None) -> int:
    """전체 지속 시간에 필요한 청크 수 계산 (올림 나눗셈)."""
    chunk_ms = chunk_duration_ms if chunk_duration_ms else CHUNK_DURATION_MS
    return (total_duration_ms + chunk_ms - 1) // chunk_ms


def get_database_config() -> dict:
    """DB 접속 설정 딕셔너리 반환 (tb_audioCache 접근용)."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "database": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "charset": DB_CHARSET,
        "autocommit": True,
    }


def get_tts_config(backend: Optional[str] = None) -> dict:
    """TTS 설정 딕셔너리 반환. backend 미지정 시 전체 설정."""
    common_config = {"chunk_duration_ms": CHUNK_DURATION_MS, "default_model": DEFAULT_TTS_MODEL}

    openai_config = {
        "backend": "openai",
        "model": OPENAI_TTS_MODEL,
        "voice": OPENAI_TTS_VOICE,
        "speed": OPENAI_TTS_SPEED,
        "sample_rate": OPENAI_TTS_SAMPLE_RATE,
        "response_format": OPENAI_TTS_RESPONSE_FORMAT,
    }

    zonos_base = {
        "device": ZONOS_TTS_DEVICE,
        "speaker_id": ZONOS_TTS_SPEAKER_ID,
        "speed": ZONOS_TTS_SPEED,
        "pitch": ZONOS_TTS_PITCH,
        "emotion": ZONOS_TTS_EMOTION,
        "sample_rate": ZONOS_TTS_SAMPLE_RATE,
        "use_reference": ZONOS_TTS_USE_REFERENCE,
        "reference_audio_path": ZONOS_TTS_REFERENCE_AUDIO_PATH,
        "max_new_tokens": ZONOS_TTS_MAX_NEW_TOKENS,
        "cfg_scale": ZONOS_TTS_CFG_SCALE,
        "language": ZONOS_TTS_LANGUAGE,
        "emotion_vector": ZONOS_TTS_EMOTION_VECTOR,
        "use_emotion_vector": ZONOS_TTS_USE_EMOTION_VECTOR,
        "auto_split_long_text": ZONOS_TTS_AUTO_SPLIT_LONG_TEXT,
        "max_text_length": ZONOS_TTS_MAX_TEXT_LENGTH,
        "crossfade_ms": ZONOS_TTS_CROSSFADE_MS,
        "segment_gap_ms": ZONOS_TTS_SEGMENT_GAP_MS,
        "speaker_continuity": ZONOS_TTS_SPEAKER_CONTINUITY,
        "overlap_enabled": ZONOS_TTS_OVERLAP_ENABLED,
        "overlap_sentences": ZONOS_TTS_OVERLAP_SENTENCES,
        "min_segment_length": ZONOS_TTS_MIN_SEGMENT_LENGTH,
        "trim_search_window": ZONOS_TTS_TRIM_SEARCH_WINDOW,
        "trim_frame_ms": ZONOS_TTS_TRIM_FRAME_MS,
        "trim_min_valley_ms": ZONOS_TTS_TRIM_MIN_VALLEY_MS,
        "spectral_crossfade_enabled": ZONOS_TTS_SPECTRAL_CROSSFADE_ENABLED,
        "spectral_crossfade_ms": ZONOS_TTS_SPECTRAL_CROSSFADE_MS,
        "rms_normalize": ZONOS_TTS_RMS_NORMALIZE,
    }

    zonos_transformer_config = {
        **zonos_base,
        "backend": "zonos_transformer",
        "model_type": "transformer",
        "model_name": ZONOS_TTS_MODEL_NAME,
    }

    zonos_hybrid_config = {
        **zonos_base,
        "backend": "zonos_hybrid",
        "model_type": "hybrid",
        "model_name": ZONOS_TTS_HYBRID_MODEL_NAME,
    }

    if backend == "openai":
        return {**common_config, **openai_config}
    elif backend == "zonos_transformer":
        return {**common_config, **zonos_transformer_config}
    elif backend == "zonos_hybrid":
        return {**common_config, **zonos_hybrid_config}
    else:
        return {
            **common_config,
            "openai": openai_config,
            "zonos_transformer": zonos_transformer_config,
            "zonos_hybrid": zonos_hybrid_config,
        }


def get_zonos_emotion_preset(preset_name: str) -> list:
    """프리셋 이름으로 감정 벡터 반환."""
    return ZONOS_EMOTION_PRESETS.get(preset_name.lower(), ZONOS_TTS_EMOTION_VECTOR)


def validate_emotion_vector(emotion_vector: list) -> bool:
    """감정 벡터 형식 및 값 검증. 8개 요소, 각 0.0~1.0, 합 ~1.0."""
    if not isinstance(emotion_vector, list) or len(emotion_vector) != 8:
        return False
    if not all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in emotion_vector):
        return False
    total = sum(emotion_vector)
    if not (0.99 <= total <= 1.01):
        print(f"⚠️ Emotion vector sum is {total:.3f}, recommended to be 1.0")
    return True


def set_default_tts_model(model: str) -> bool:
    """기본 TTS 모델 변경."""
    global DEFAULT_TTS_MODEL
    valid_models = [TTS_MODEL_OPENAI, TTS_MODEL_ZONOS_TRANSFORMER, TTS_MODEL_ZONOS_HYBRID, TTS_MODEL_AUTO]
    if model not in valid_models:
        print(f"⚠️ Invalid TTS model: {model}. Valid models: {valid_models}")
        return False
    DEFAULT_TTS_MODEL = model
    return True


def set_force_tts_regeneration(enable: bool) -> bool:
    """TTS 강제 재생성 플래그 설정."""
    global FORCE_TTS_REGENERATION
    if not isinstance(enable, bool):
        return False
    FORCE_TTS_REGENERATION = enable
    return True


def get_default_tts_model() -> str:
    """현재 기본 TTS 모델 반환."""
    return DEFAULT_TTS_MODEL


def get_force_tts_regeneration() -> bool:
    """TTS 강제 재생성 플래그 반환."""
    return FORCE_TTS_REGENERATION


# =============================================================================
# STT Helper Functions
# =============================================================================


def get_stt_config(backend: Optional[str] = None) -> dict:
    """STT 설정 딕셔너리 반환. backend 미지정 시 전체 설정."""
    common_config = {
        "input_sample_rate": STT_INPUT_SAMPLE_RATE,
        "default_backend": DEFAULT_STT_BACKEND,
    }

    whisper_config = {
        "backend": STT_BACKEND_WHISPER,
        "model_name": WHISPER_MODEL_NAME,
        "language": WHISPER_LANGUAGE,
        "sample_rate": WHISPER_SAMPLE_RATE,
        "chunk_length_s": WHISPER_CHUNK_LENGTH_S,
        "stride_length_s": WHISPER_STRIDE_LENGTH_S,
        "max_audio_duration": WHISPER_MAX_AUDIO_DURATION,
        "min_audio_duration": WHISPER_MIN_AUDIO_DURATION,
    }

    clova_config = {
        "backend": STT_BACKEND_CLOVA,
        "api_key": CLOVA_API_KEY,
        "api_url": CLOVA_API_URL,
        "language": CLOVA_LANGUAGE,
    }

    if backend == STT_BACKEND_WHISPER:
        return {**common_config, **whisper_config}
    elif backend == STT_BACKEND_CLOVA:
        return {**common_config, **clova_config}
    else:
        return {
            **common_config,
            STT_BACKEND_WHISPER: whisper_config,
            STT_BACKEND_CLOVA: clova_config,
        }


def get_default_stt_backend() -> str:
    """현재 기본 STT 백엔드 반환."""
    return DEFAULT_STT_BACKEND
