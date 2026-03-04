#!/usr/bin/env python3
"""
Zonos TTS Handler Module
Zonos TTS 핸들러 모듈

Handles text-to-speech conversion using Zonos TTS model (Zyphra).
Inherits from BaseTTSHandler for common functionality.
Supports both Transformer and Hybrid model architectures.

Zonos TTS 모델(Zyphra)을 사용한 텍스트 음성 변환을 처리합니다.
공통 기능을 위해 BaseTTSHandler를 상속받습니다.
Transformer 및 Hybrid 모델 아키텍처를 모두 지원합니다.

Requirements:
    - pip install -e . (from cloned Zonos repo)
    - pip install phonemizer
    - sudo apt install espeak-ng
    - CUDA GPU with sufficient memory
"""

import io
import os
import threading
from typing import Tuple, Optional, Dict, Any

# Import base class
from .Base_TTS_Handler import BaseTTSHandler

# Import configuration
from hearo_speech_manager.config.speech_config import (
    CHUNK_DURATION_MS,
    ZONOS_TTS_MODEL_TYPE,
    ZONOS_TTS_MODEL_NAME,
    ZONOS_TTS_DEVICE,
    ZONOS_TTS_SPEAKER_ID,
    ZONOS_TTS_SPEED,
    ZONOS_TTS_PITCH,
    ZONOS_TTS_SAMPLE_RATE,
    ZONOS_TTS_MAX_NEW_TOKENS,
    ZONOS_TTS_CFG_SCALE,
    ZONOS_TTS_LANGUAGE,
    ZONOS_TTS_AUTO_SPLIT_LONG_TEXT,
    ZONOS_TTS_MAX_TEXT_LENGTH,
    ZONOS_TTS_SPLIT_SENTENCES,
    ZONOS_TTS_EMOTION_VECTOR,
    ZONOS_TTS_USE_EMOTION_VECTOR,
    ZONOS_EMOTION_PRESETS,
    get_zonos_emotion_preset,
    validate_emotion_vector,
)

# Lazy imports for Zonos (heavy dependencies)
# Zonos 지연 임포트 (무거운 의존성)
_torch = None
_torchcodec_available = False
_torchaudio_transforms_available = False
_zonos_available = False


def _check_zonos_availability():
    """
    Check if Zonos and its dependencies are available.
    Zonos 및 관련 의존성의 사용 가능 여부를 확인합니다.

    I/O: torchcodec (AudioDecoder), Signal processing: torchaudio.transforms (Resample)
    I/O: torchcodec (AudioDecoder), 신호 처리: torchaudio.transforms (Resample)
    """
    global _torch, _torchcodec_available, _torchaudio_transforms_available, _zonos_available

    torch_ok = False
    torchcodec_ok = False
    torchaudio_transforms_ok = False
    zonos_ok = False

    # 1. torch (required / 필수)
    try:
        import torch

        _torch = torch
        torch_ok = True
    except ImportError as e:
        print(f"[Zonos_TTS_Handler] torch import error: {e}")
        return False

    # 2. torchcodec - AudioDecoder for audio I/O (required / 필수)
    # 오디오 I/O를 위한 torchcodec AudioDecoder
    try:
        from torchcodec.decoders import AudioDecoder

        _torchcodec_available = True
        torchcodec_ok = True
    except ImportError as e:
        print(f"[Zonos_TTS_Handler] torchcodec import error: {e}")
        return False

    # 3. torchaudio.transforms - Resample for signal processing (required / 필수)
    # 신호 처리를 위한 torchaudio.transforms Resample
    try:
        from torchaudio.transforms import Resample

        _torchaudio_transforms_available = True
        torchaudio_transforms_ok = True
    except ImportError as e:
        print(f"[Zonos_TTS_Handler] torchaudio.transforms import error: {e}")
        return False

    # 4. Zonos model (required / 필수)
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict

        _zonos_available = True
        zonos_ok = True
    except ImportError as e:
        print(f"[Zonos_TTS_Handler] Zonos import error: {e}")
        import traceback

        traceback.print_exc()
        _zonos_available = False
    except Exception as e:
        print(f"[Zonos_TTS_Handler] Zonos unexpected error: {e}")
        import traceback

        traceback.print_exc()
        _zonos_available = False

    if not zonos_ok:
        print(
            f"[Zonos_TTS_Handler] Dependency status: torch={torch_ok}, torchcodec={torchcodec_ok}, torchaudio.transforms={torchaudio_transforms_ok}, zonos={zonos_ok}"
        )
        return False

    return True


class Zonos_TTS_Handler(BaseTTSHandler):
    """
    Handler for Zonos TTS (Text-to-Speech) operations.
    Zonos TTS (텍스트 음성 변환) 작업을 위한 핸들러입니다.

    Inherits from BaseTTSHandler and implements Zonos-specific TTS logic.
    BaseTTSHandler를 상속받아 Zonos 전용 TTS 로직을 구현합니다.

    Provides:
    제공 기능:
    - Text-to-speech conversion using Zonos model (local GPU)
      Zonos 모델을 사용한 텍스트 음성 변환 (로컬 GPU)
    - Audio chunking with configurable chunk duration (inherited)
      설정 가능한 청크 지속 시간의 오디오 청킹 (상속)
    - Voice conditioning with speaking_rate, emotion
      speaking_rate, emotion을 통한 음성 조절

    Supported Models:
    지원 모델:
    - Zyphra/Zonos-v0.1-transformer (default, faster)
    - Zyphra/Zonos-v0.1-hybrid (higher quality)
    """

    # Backend identifier
    BACKEND_NAME: str = "zonos"
    BACKEND_TYPE: str = "local"
    SUPPORTED_FORMATS: list = ["mp3", "wav", "opus", "aac", "flac"]  # Supported output formats

    # Model variants
    MODEL_TRANSFORMER = "Zyphra/Zonos-v0.1-transformer"
    MODEL_HYBRID = "Zyphra/Zonos-v0.1-hybrid"

    # Default Zonos sample rate (44.1kHz output)
    ZONOS_OUTPUT_SAMPLE_RATE = 44100

    def __init__(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        emotion: Optional[str] = None,
        emotion_vector: Optional[list] = None,
        use_reference: Optional[bool] = None,
        reference_audio_path: Optional[str] = None,
        lazy: bool = False,
    ):
        """
        Initialize Zonos TTS Handler.

        Args:
            model_type: Model type ('transformer' or 'hybrid', if None uses config)
            model_name: Full model name (if None, derived from model_type)
            device: Device to use ('cuda' or 'cpu', if None uses config)
            speaker_id: Speaker ID for voice selection (not used in Zonos v0.1)
            speed: Speaking rate multiplier (if None uses config)
            pitch: Pitch multiplier for audio post-processing (0.5~2.0, 1.0=normal)
            emotion: Emotion preset name ('happy', 'sad', 'neutral', etc.)
            emotion_vector: Custom emotion vector [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
            use_reference: Whether to use reference audio
            reference_audio_path: Path to reference audio file
            lazy: If True, skip model loading at init (load later via load_model())
        """
        # Initialize base class
        super().__init__(sample_rate=ZONOS_TTS_SAMPLE_RATE, chunk_duration_ms=CHUNK_DURATION_MS)

        # Zonos specific settings from config
        self.model_type = model_type or ZONOS_TTS_MODEL_TYPE
        self.device = device or ZONOS_TTS_DEVICE
        self.speaker_id = speaker_id if speaker_id is not None else ZONOS_TTS_SPEAKER_ID
        self.speed = speed if speed is not None else ZONOS_TTS_SPEED
        self.pitch = pitch if pitch is not None else ZONOS_TTS_PITCH

        # Emotion vector configuration
        # Priority: emotion_vector > emotion preset > config default
        if emotion_vector is not None:
            if validate_emotion_vector(emotion_vector):
                self.emotion_vector = emotion_vector
                print(f"[Zonos_TTS_Handler] Using custom emotion vector: {emotion_vector}")
            else:
                print(f"[Zonos_TTS_Handler] ⚠️ Invalid emotion vector, using default")
                self.emotion_vector = ZONOS_TTS_EMOTION_VECTOR
        elif emotion is not None:
            self.emotion_vector = get_zonos_emotion_preset(emotion)
            print(f"[Zonos_TTS_Handler] Using emotion preset '{emotion}': {self.emotion_vector}")
        else:
            self.emotion_vector = ZONOS_TTS_EMOTION_VECTOR

        self.use_emotion_vector = ZONOS_TTS_USE_EMOTION_VECTOR

        # Determine model name
        if model_name:
            self.model_name = model_name
        elif self.model_type == "hybrid":
            self.model_name = self.MODEL_HYBRID
        else:
            self.model_name = ZONOS_TTS_MODEL_NAME or self.MODEL_TRANSFORMER

        # Update backend name based on model type
        self.BACKEND_NAME = f"zonos_{self.model_type}"

        # Zonos model instance
        self.model = None
        self.torch = None
        self._AudioDecoder = None
        self._Resample = None

        # Deep-defense lock: serializes model.generate() + autoencoder.decode()
        # even if the upper-level HeaRo_TTS_Handler._inference_lock is bypassed.
        self._inference_lock = threading.Lock()

        # Reference audio / speaker embedding
        self.speaker_embedding = None
        self.use_reference = use_reference if use_reference is not None else False
        self.reference_audio_path = reference_audio_path

        # Initialize (skip if lazy mode for on-demand GPU loading)
        if not lazy:
            self._initialized = self._initialize()

    def load_model(self) -> bool:
        """
        Load Zonos model to GPU (on-demand).
        Zonos 모델을 GPU에 로드합니다 (on-demand).

        Returns:
            True if model is loaded and ready, False otherwise
        """
        if self._initialized and self.model is not None:
            return True
        self._initialized = self._initialize()
        return self._initialized

    def unload_model(self):
        """
        Unload Zonos model from GPU and free VRAM.
        Zonos 모델을 GPU에서 언로드하고 VRAM을 해제합니다.

        NOTE: Do NOT call self.model.cpu() — Zonos uses meta tensors internally
        which cause NotImplementedError on .cpu()/.to(). Instead, delete all
        references and let GC + empty_cache() reclaim GPU memory.
        """
        import gc

        if self.model is not None:
            # Clear lazily-created sub-models that hold their own GPU tensors
            if hasattr(self.model, "spk_clone_model"):
                del self.model.spk_clone_model
            del self.model
            self.model = None

        self.speaker_embedding = None
        self._initialized = False

        # Force two rounds of GC to break circular references in PyTorch modules
        gc.collect()
        gc.collect()

        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()

    def load(self) -> bool:
        """BaseTTSHandler interface: load model to device."""
        return self.load_model()

    def unload(self):
        """BaseTTSHandler interface: unload model from device."""
        self.unload_model()

    def _initialize(self) -> bool:
        """
        Initialize Zonos model.
        Zonos 모델을 초기화합니다.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check availability
            if not _check_zonos_availability():
                print(f"[Zonos_TTS_Handler] Required dependencies not available")
                return False

            # Import dependencies - I/O via torchcodec, signal processing via torchaudio
            # 의존성 임포트 - I/O는 torchcodec, 신호 처리는 torchaudio
            import torch
            from torchcodec.decoders import AudioDecoder
            from torchaudio.transforms import Resample
            from zonos.model import Zonos

            self.torch = torch
            self._AudioDecoder = AudioDecoder
            self._Resample = Resample

            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                print(f"[Zonos_TTS_Handler] CUDA not available, falling back to CPU")
                self.device = "cpu"

            # Load model
            print(f"[Zonos_TTS_Handler] Loading model: {self.model_name} on {self.device}...")
            self.model = Zonos.from_pretrained(self.model_name, device=self.device)
            print(f"[Zonos_TTS_Handler] ✓ Model loaded successfully")

            # Load reference audio if specified
            if self.use_reference and self.reference_audio_path:
                self._load_speaker_embedding()

            return True

        except Exception as e:
            print(f"[Zonos_TTS_Handler] Failed to initialize: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _load_speaker_embedding(self) -> bool:
        """
        Load speaker embedding from reference audio.
        참조 오디오에서 화자 임베딩을 로드합니다.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.reference_audio_path:
                print(f"[Zonos_TTS_Handler] Reference audio path not specified")
                return False

            # Convert relative path to absolute path
            # 상대 경로를 절대 경로로 변환
            audio_path = self.reference_audio_path
            if not os.path.isabs(audio_path):
                # Get package directory (hearo_knowledge_manager package root)
                # 패키지 디렉터리 경로 획득 (hearo_knowledge_manager 패키지 루트)
                import hearo_knowledge_manager

                package_dir = os.path.dirname(os.path.dirname(hearo_knowledge_manager.__file__))
                audio_path = os.path.join(package_dir, "hearo_knowledge_manager", audio_path)
                print(f"[Zonos_TTS_Handler] Converted relative path to absolute: {audio_path}")

            if not os.path.exists(audio_path):
                print(f"[Zonos_TTS_Handler] Reference audio not found: {audio_path}")
                return False

            # Load audio file
            decoder = self._AudioDecoder(audio_path)
            result = decoder.get_all_samples()
            audio, sr = result.data, result.sample_rate

            # Create speaker embedding using Zonos
            self.speaker_embedding = self.model.make_speaker_embedding(audio, sr)
            print(f"[Zonos_TTS_Handler] ✓ Speaker embedding created from: {audio_path}")
            return True

        except Exception as e:
            print(f"[Zonos_TTS_Handler] Failed to load speaker embedding: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _text_to_speech_impl(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: str = "mp3",  # Default to mp3 to match OpenAI
        **kwargs,
    ) -> Tuple[bool, str, bytes]:
        """
        Convert text to speech using Zonos TTS.
        Zonos TTS를 사용하여 텍스트를 음성으로 변환합니다.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (not used in Zonos v0.1)
            speed: Speaking rate multiplier (if None uses config default)
            response_format: Audio format ('mp3', 'wav', 'opus', 'aac', 'flac')
            **kwargs: Additional parameters
                - language: Language code (default: from config)
                - cfg_scale: Classifier-free guidance scale (default: from config)
                - max_new_tokens: Maximum tokens to generate (default: from config)
                - auto_split: Auto split long text (default: from config)
                - max_text_length: Max chars per segment (default: from config)
                - emotion_vector: Custom emotion vector (default: from config/instance)
                - emotion: Emotion preset name ('happy', 'sad', etc.)
                - pitch: Pitch multiplier (0.5~2.0, default: from config/instance)

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        if not self.model:
            return False, "Zonos model not initialized", b""

        try:
            # Get parameters from kwargs or config
            language = kwargs.get("language", ZONOS_TTS_LANGUAGE)
            cfg_scale = kwargs.get("cfg_scale", ZONOS_TTS_CFG_SCALE)
            max_new_tokens = kwargs.get("max_new_tokens", ZONOS_TTS_MAX_NEW_TOKENS)
            auto_split = kwargs.get("auto_split", ZONOS_TTS_AUTO_SPLIT_LONG_TEXT)
            max_text_length = kwargs.get("max_text_length", ZONOS_TTS_MAX_TEXT_LENGTH)
            pitch = kwargs.get("pitch", None)  # Get pitch from kwargs

            # Emotion vector handling
            # Priority: kwargs emotion_vector > kwargs emotion preset > instance emotion_vector
            emotion_vector = kwargs.get("emotion_vector", None)
            if emotion_vector is None:
                emotion_preset = kwargs.get("emotion", None)
                if emotion_preset:
                    emotion_vector = get_zonos_emotion_preset(emotion_preset)
                else:
                    emotion_vector = self.emotion_vector if self.use_emotion_vector else None

            # Check if text needs to be split
            if auto_split and len(text) > max_text_length:
                print(
                    f"[Zonos_TTS_Handler] Text length ({len(text)}) exceeds max ({max_text_length}), splitting into segments..."
                )
                return self._text_to_speech_with_split(
                    text=text,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    response_format=response_format,
                    language=language,
                    cfg_scale=cfg_scale,
                    max_new_tokens=max_new_tokens,
                    max_text_length=max_text_length,
                    emotion_vector=emotion_vector,
                )

            # Single segment TTS
            return self._generate_single_segment(
                text=text,
                speed=speed,
                pitch=pitch,
                response_format=response_format,
                language=language,
                cfg_scale=cfg_scale,
                max_new_tokens=max_new_tokens,
                emotion_vector=emotion_vector,
            )

        except Exception as e:
            error_message = f"Zonos TTS generation failed: {str(e)}"
            import traceback

            traceback.print_exc()
            return False, error_message, b""

    def _generate_single_segment(
        self,
        text: str,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        response_format: str = "mp3",
        language: str = "ko",
        cfg_scale: float = 2.0,
        max_new_tokens: int = 8000,
        emotion_vector: Optional[list] = None,
        speaker_embedding=None,
    ) -> Tuple[bool, str, bytes]:
        """
        Generate TTS for a single text segment.
        단일 텍스트 세그먼트에 대한 TTS를 생성합니다.

        Args:
            text: Text to convert
            speed: Speaking rate multiplier
            pitch: Pitch multiplier (0.5~2.0, 1.0=normal)
            response_format: Audio format
            language: Language code
            cfg_scale: Classifier-free guidance scale
            max_new_tokens: Maximum audio tokens
            emotion_vector: Emotion vector [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
            speaker_embedding: Override speaker embedding (None uses self.speaker_embedding)

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        success, error_msg, audio_bytes, _ = self._generate_single_segment_with_tensor(
            text=text,
            speed=speed,
            pitch=pitch,
            response_format=response_format,
            language=language,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            emotion_vector=emotion_vector,
            speaker_embedding=speaker_embedding,
        )
        return success, error_msg, audio_bytes

    def _generate_single_segment_with_tensor(
        self,
        text: str,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        response_format: str = "mp3",
        language: str = "ko",
        cfg_scale: float = 2.0,
        max_new_tokens: int = 8000,
        emotion_vector: Optional[list] = None,
        speaker_embedding=None,
        trim_prefix_chars: int = 0,
        total_text_chars: int = 0,
        fmax: float = 22050.0,
        pitch_std: float = 20.0,
        core_text: str = "",
    ) -> Tuple[bool, str, bytes, any]:
        """
        Generate TTS for a single text segment, returning both audio bytes and raw tensor.
        단일 텍스트 세그먼트에 대한 TTS를 생성하고 오디오 바이트와 raw 텐서를 함께 반환합니다.

        Args:
            text: Text to convert
            speed: Speaking rate multiplier
            pitch: Pitch multiplier (0.5~2.0, 1.0=normal)
            response_format: Audio format
            language: Language code
            cfg_scale: Classifier-free guidance scale
            max_new_tokens: Maximum audio tokens
            emotion_vector: Emotion vector [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
            speaker_embedding: Override speaker embedding (None uses self.speaker_embedding)
            trim_prefix_chars: Number of prefix-context characters to trim from the
                               generated audio (0 = no trimming). Used by the quality
                               pipeline to remove overlapping context audio.
                               생성된 오디오에서 잘라낼 프리픽스 컨텍스트 글자 수 (0 = 트림 없음).
            total_text_chars: Total character count of *text* (for proportional trim
                              calculation). If 0, uses len(text).
                              *text*의 전체 글자 수 (비례 트림 계산용). 0이면 len(text) 사용.
            fmax: Maximum frequency for Zonos conditioning (22050 recommended for cloning).
                  Zonos conditioning용 최대 주파수 (클로닝 시 22050 권장).
            pitch_std: Pitch standard deviation for Zonos conditioning (20-45 normal).
                       Zonos conditioning용 피치 표준 편차 (20-45 일반).
            core_text: The actual core text (without prefix) for post-trim verification.
                       If provided and trim_prefix_chars > 0, the remaining audio
                       duration is checked against expected duration for core_text.
                       On mismatch, the segment is regenerated without overlap.
                       포스트 트림 검증을 위한 실제 core 텍스트 (prefix 제외).

        Returns:
            Tuple of (success, error_message, audio_data, audio_tensor)
            audio_tensor is the raw [channels, samples] tensor before format conversion (None on failure)
        """
        from zonos.conditioning import make_cond_dict

        speed_to_use = speed if speed is not None else self.speed
        speaking_rate = 15.0 * speed_to_use

        pitch_to_use = pitch if pitch is not None else self.pitch

        speaker_to_use = (
            speaker_embedding if speaker_embedding is not None else self.speaker_embedding
        )

        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_to_use,
            speaking_rate=speaking_rate,
            emotion=emotion_vector,
            fmax=fmax,
            pitch_std=pitch_std,
            device=self.device,
        )

        conditioning = self.model.prepare_conditioning(cond_dict)

        with self._inference_lock:
            with self.torch.no_grad():
                codes = self.model.generate(
                    prefix_conditioning=conditioning,
                    max_new_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    progress_bar=False,
                    disable_torch_compile=True,
                )

            audio_tensor = self.model.autoencoder.decode(codes).cpu()

        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if pitch_to_use != 1.0:
            audio_tensor = self._apply_pitch_shift(audio_tensor, pitch_to_use)

        if self.ZONOS_OUTPUT_SAMPLE_RATE != self.sample_rate:
            resampler = self._Resample(self.ZONOS_OUTPUT_SAMPLE_RATE, self.sample_rate)
            audio_tensor = resampler(audio_tensor)

        # --- Signal-based overlap prefix trimming ---
        # Detect the sentence-boundary pause via energy-valley analysis
        # and trim the prefix audio precisely instead of using character-ratio
        # 에너지 valley 분석을 통해 문장 경계 pause를 탐지하고
        # 글자 비율 대신 정확하게 프리픽스 오디오를 트림합니다
        if trim_prefix_chars > 0:
            char_count = total_text_chars if total_text_chars > 0 else len(text)
            if char_count > 0 and trim_prefix_chars < char_count:
                try:
                    from hearo_speech_manager.module.AudioBoundaryDetector import (
                        detect_sentence_boundary,
                        estimate_punctuation_aware_ratio,
                    )
                    from hearo_speech_manager.config.speech_config import (
                        ZONOS_TTS_TRIM_SEARCH_WINDOW,
                        ZONOS_TTS_TRIM_FRAME_MS,
                        ZONOS_TTS_TRIM_MIN_VALLEY_MS,
                        ZONOS_TTS_PUNCTUATION_WEIGHTS,
                    )

                    estimated_ratio = estimate_punctuation_aware_ratio(
                        text=text,
                        boundary_char_idx=trim_prefix_chars,
                        punctuation_weights=ZONOS_TTS_PUNCTUATION_WEIGHTS,
                    )
                    audio_1d = audio_tensor.squeeze().cpu().numpy()
                    trim_sample = detect_sentence_boundary(
                        audio=audio_1d,
                        sample_rate=self.sample_rate,
                        estimated_ratio=estimated_ratio,
                        search_window_ratio=ZONOS_TTS_TRIM_SEARCH_WINDOW,
                        frame_ms=ZONOS_TTS_TRIM_FRAME_MS,
                        min_valley_duration_ms=ZONOS_TTS_TRIM_MIN_VALLEY_MS,
                    )
                    if 0 < trim_sample < audio_tensor.shape[-1]:
                        audio_tensor = audio_tensor[:, trim_sample:]
                        print(
                            f"[Zonos_TTS_Handler] Signal-based trim at sample {trim_sample} "
                            f"(estimated_ratio={estimated_ratio:.2f}, "
                            f"{trim_prefix_chars}/{char_count} chars)"
                        )
                except ImportError:
                    estimated_ratio = trim_prefix_chars / char_count
                    total_samples = audio_tensor.shape[-1]
                    overshoot_ratio = min(estimated_ratio + 0.05, 0.95)
                    trim_samples = int(total_samples * overshoot_ratio)
                    if 0 < trim_samples < total_samples:
                        audio_tensor = audio_tensor[:, trim_samples:]
                        print(
                            f"[Zonos_TTS_Handler] Fallback trim at {trim_samples} samples "
                            f"(ratio={overshoot_ratio:.2f})"
                        )

        # --- Post-trim verification ---
        # Check trimmed audio duration against expected duration for core_text.
        # If the remaining audio is unreasonably short or long, the trim
        # likely hit the wrong valley — regenerate with core_text only.
        # 트림 후 남은 오디오 길이가 core_text 예상 길이와 합리적인지 검증합니다.
        verification_text = core_text if core_text else text
        if trim_prefix_chars > 0 and core_text:
            remaining_sec = audio_tensor.shape[-1] / self.sample_rate
            core_len = len(core_text)
            expected_min = core_len * 0.08
            expected_max = core_len * 0.25
            print(
                f"[Zonos_TTS_Handler] Trim verification: "
                f"remaining={remaining_sec:.2f}s, "
                f"expected=[{expected_min:.2f}~{expected_max:.2f}]s, "
                f"core_chars={core_len}"
            )
            if not (expected_min <= remaining_sec <= expected_max):
                print(
                    f"[Zonos_TTS_Handler] Trim verification FAILED — "
                    f"regenerating with core_text only (no overlap)"
                )
                return self._generate_single_segment_with_tensor(
                    text=core_text,
                    speed=speed,
                    pitch=pitch,
                    response_format=response_format,
                    language=language,
                    cfg_scale=cfg_scale,
                    max_new_tokens=max_new_tokens,
                    emotion_vector=emotion_vector,
                    speaker_embedding=speaker_embedding,
                    trim_prefix_chars=0,
                    total_text_chars=0,
                    fmax=fmax,
                    pitch_std=pitch_std,
                    core_text="",
                )

        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)

        if response_format != "wav":
            audio_bytes = self._convert_audio_format(wav_bytes, "wav", response_format)
        else:
            audio_bytes = wav_bytes

        return True, "", audio_bytes, audio_tensor

    def _split_text_into_segments(self, text: str, max_length: int) -> list:
        """
        Split text into segments based on sentence boundaries.
        문장 경계를 기준으로 텍스트를 세그먼트로 분할합니다.

        Args:
            text: Text to split
            max_length: Maximum characters per segment

        Returns:
            List of text segments
        """
        import re

        segments = []

        if not ZONOS_TTS_SPLIT_SENTENCES:
            # Simple chunking without sentence boundary
            for i in range(0, len(text), max_length):
                segments.append(text[i : i + max_length])
            return segments

        # Split by sentence boundaries (Korean/English)
        # 한국어: . ! ? 등 / English: . ! ?
        sentence_endings = r"[.!?。！？]\s*"
        sentences = re.split(f"({sentence_endings})", text)

        # Rejoin sentences with their endings
        sentences_with_endings = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentences_with_endings.append(sentences[i] + sentences[i + 1])
            else:
                sentences_with_endings.append(sentences[i])
        if len(sentences) % 2 == 1:
            sentences_with_endings.append(sentences[-1])

        # Group sentences into segments
        current_segment = ""
        for sentence in sentences_with_endings:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence exceeds max_length
            if current_segment and len(current_segment) + len(sentence) + 1 > max_length:
                # Save current segment and start new one
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                # Add to current segment
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence

        # Add last segment
        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def _text_to_speech_with_split(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        response_format: str = "mp3",
        language: str = "ko",
        cfg_scale: float = 2.0,
        max_new_tokens: int = 8000,
        max_text_length: int = 200,
        emotion_vector: Optional[list] = None,
    ) -> Tuple[bool, str, bytes]:
        """
        Convert long text to speech via the 3-Stage Quality Pipeline.
        3단계 품질 파이프라인을 통해 긴 텍스트를 음성으로 변환합니다.

        Stage 1 — ContextAwareSegmenter: text → SegmentPlans with overlap context
        Stage 2 — Continuity Generator:  SegmentPlans → raw audio arrays (with trimming)
        Stage 3 — SpectralAudioAssembler: raw arrays → seamless final audio

        Falls back to legacy pydub concatenation if any pipeline component fails.
        파이프라인 구성 요소가 실패하면 기존 pydub 연결 방식으로 자동 폴백합니다.

        Args:
            text: Text to convert
            voice: Voice identifier (not used)
            speed: Speaking rate multiplier
            pitch: Pitch multiplier
            response_format: Audio format
            language: Language code
            cfg_scale: Classifier-free guidance scale
            max_new_tokens: Maximum audio tokens per segment
            max_text_length: Maximum characters per segment
            emotion_vector: Emotion vector for all segments

        Returns:
            Tuple of (success, error_message, audio_data)
        """
        import numpy as np
        from hearo_speech_manager.config.speech_config import (
            ZONOS_TTS_CROSSFADE_MS,
            ZONOS_TTS_SPEAKER_CONTINUITY,
            ZONOS_TTS_SEGMENT_GAP_MS,
            ZONOS_TTS_OVERLAP_ENABLED,
            ZONOS_TTS_OVERLAP_SENTENCES,
            ZONOS_TTS_MIN_SEGMENT_LENGTH,
            ZONOS_TTS_SPLIT_SENTENCES,
            ZONOS_TTS_MAX_NEW_TOKENS,
            ZONOS_TTS_SPECTRAL_CROSSFADE_ENABLED,
            ZONOS_TTS_SPECTRAL_CROSSFADE_MS,
            ZONOS_TTS_RMS_NORMALIZE,
        )

        # ── Stage 1: Context-Aware Segmentation ──
        use_quality_pipeline = True
        try:
            from hearo_speech_manager.module.ContextAwareSegmenter import segment as ctx_segment

            segment_plans = ctx_segment(
                text=text,
                max_length=max_text_length,
                overlap_sentences=ZONOS_TTS_OVERLAP_SENTENCES,
                overlap_enabled=ZONOS_TTS_OVERLAP_ENABLED,
                min_segment_length=ZONOS_TTS_MIN_SEGMENT_LENGTH,
                split_sentences=ZONOS_TTS_SPLIT_SENTENCES,
            )
            print(
                f"[Zonos_TTS_Handler] Stage 1: Context-aware segmentation → "
                f"{len(segment_plans)} segments (overlap={ZONOS_TTS_OVERLAP_ENABLED})"
            )
        except Exception as seg_err:
            print(
                f"[Zonos_TTS_Handler] Stage 1 failed ({seg_err}), falling back to legacy segmenter"
            )
            use_quality_pipeline = False

        # Legacy fallback: use the old simple segmenter
        if not use_quality_pipeline:
            return self._text_to_speech_with_split_legacy(
                text=text,
                voice=voice,
                speed=speed,
                pitch=pitch,
                response_format=response_format,
                language=language,
                cfg_scale=cfg_scale,
                max_new_tokens=max_new_tokens,
                max_text_length=max_text_length,
                emotion_vector=emotion_vector,
            )

        # ── Stage 2: Continuity Generator ──
        raw_audio_arrays: list = []
        current_speaker_embedding = None
        fixed_pitch_std = 20.0
        fixed_fmax = 22050.0

        for plan in segment_plans:
            idx = plan.segment_index
            total = plan.total_segments
            print(
                f"[Zonos_TTS_Handler] Stage 2: Generating segment {idx+1}/{total}: "
                f"{plan.core_text[:50]}..."
            )

            # Adaptive max_new_tokens: proportional to text length
            tokens_per_char = ZONOS_TTS_MAX_NEW_TOKENS / max_text_length
            adaptive_tokens = int(len(plan.text) * tokens_per_char)
            adaptive_tokens = max(1500, min(adaptive_tokens, ZONOS_TTS_MAX_NEW_TOKENS))

            success, error_msg, audio_bytes, audio_tensor = (
                self._generate_single_segment_with_tensor(
                    text=plan.text,
                    speed=speed,
                    pitch=pitch,
                    response_format="wav",
                    language=language,
                    cfg_scale=cfg_scale,
                    max_new_tokens=adaptive_tokens,
                    emotion_vector=emotion_vector,
                    speaker_embedding=current_speaker_embedding,
                    trim_prefix_chars=plan.prefix_char_count,
                    total_text_chars=len(plan.text),
                    fmax=fixed_fmax,
                    pitch_std=fixed_pitch_std,
                    core_text=plan.core_text,
                )
            )

            if not success:
                return False, f"Failed to generate segment {idx+1}: {error_msg}", b""

            # Convert tensor to numpy for spectral assembler
            if audio_tensor is not None:
                np_audio = audio_tensor.squeeze().numpy().astype(np.float32)
                raw_audio_arrays.append(np_audio)
            else:
                return False, f"Segment {idx+1} returned None tensor", b""

            # Enhanced speaker embedding: extract from the stable centre of the segment
            if ZONOS_TTS_SPEAKER_CONTINUITY and audio_tensor is not None and idx < total - 1:
                try:
                    n_samples = audio_tensor.shape[-1]
                    quarter = n_samples // 4
                    centre_tensor = audio_tensor[:, quarter : quarter * 3]
                    if centre_tensor.shape[-1] < self.sample_rate // 4:
                        centre_tensor = audio_tensor
                    current_speaker_embedding = self.model.make_speaker_embedding(
                        centre_tensor, self.sample_rate
                    )
                    print(
                        f"[Zonos_TTS_Handler] Enhanced speaker embedding from centre 50% of segment {idx+1}"
                    )
                except Exception as emb_e:
                    print(
                        f"[Zonos_TTS_Handler] Speaker embedding extraction failed: {emb_e}, using default"
                    )
                    current_speaker_embedding = None

            if self.torch and self.device == "cuda":
                self.torch.cuda.empty_cache()

        # ── Stage 3: Spectral Assembly ──
        try:
            from hearo_speech_manager.module.SpectralAudioAssembler import (
                assemble as spectral_assemble,
            )

            print(
                f"[Zonos_TTS_Handler] Stage 3: Spectral assembly of {len(raw_audio_arrays)} segments "
                f"(crossfade={ZONOS_TTS_SPECTRAL_CROSSFADE_MS}ms, RMS={ZONOS_TTS_RMS_NORMALIZE})"
            )

            final_np = spectral_assemble(
                segments=raw_audio_arrays,
                sample_rate=self.sample_rate,
                crossfade_ms=ZONOS_TTS_SPECTRAL_CROSSFADE_MS,
                rms_normalize_enabled=ZONOS_TTS_RMS_NORMALIZE,
                spectral_crossfade_enabled=ZONOS_TTS_SPECTRAL_CROSSFADE_ENABLED,
            )

            # Convert numpy → tensor → wav bytes → final format
            final_tensor = self.torch.from_numpy(final_np).unsqueeze(0)
            wav_bytes = self._tensor_to_wav_bytes(final_tensor)

            if response_format != "wav":
                final_audio_bytes = self._convert_audio_format(wav_bytes, "wav", response_format)
            else:
                final_audio_bytes = wav_bytes

            print(
                f"[Zonos_TTS_Handler] ✓ Quality Pipeline complete: {len(final_audio_bytes)} bytes"
            )

        except Exception as asm_err:
            print(
                f"[Zonos_TTS_Handler] Stage 3 failed ({asm_err}), falling back to pydub concatenation"
            )
            final_audio_bytes = self._pydub_fallback_concat(
                raw_audio_arrays,
                response_format,
                ZONOS_TTS_SEGMENT_GAP_MS,
                ZONOS_TTS_CROSSFADE_MS,
            )
        return True, "", final_audio_bytes

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _text_to_speech_with_split_legacy(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        response_format: str = "mp3",
        language: str = "ko",
        cfg_scale: float = 2.0,
        max_new_tokens: int = 8000,
        max_text_length: int = 200,
        emotion_vector: Optional[list] = None,
    ) -> Tuple[bool, str, bytes]:
        """
        Legacy segment-and-concatenate pipeline (pre-Quality-Pipeline).
        품질 파이프라인 실패 시 사용되는 기존 세그먼트-연결 파이프라인.
        """
        from pydub import AudioSegment
        from hearo_speech_manager.config.speech_config import (
            ZONOS_TTS_CROSSFADE_MS,
            ZONOS_TTS_SPEAKER_CONTINUITY,
            ZONOS_TTS_SEGMENT_GAP_MS,
        )

        segments = self._split_text_into_segments(text, max_text_length)
        print(f"[Zonos_TTS_Handler][legacy] Split text into {len(segments)} segments")

        audio_segments = []
        current_speaker_embedding = None

        for i, segment_text in enumerate(segments):
            print(
                f"[Zonos_TTS_Handler][legacy] Generating segment {i+1}/{len(segments)}: {segment_text[:50]}..."
            )

            success, error_msg, audio_bytes, audio_tensor = (
                self._generate_single_segment_with_tensor(
                    text=segment_text,
                    speed=speed,
                    pitch=pitch,
                    response_format="wav",
                    language=language,
                    cfg_scale=cfg_scale,
                    max_new_tokens=max_new_tokens,
                    emotion_vector=emotion_vector,
                    speaker_embedding=current_speaker_embedding,
                )
            )

            if not success:
                return False, f"Failed to generate segment {i+1}: {error_msg}", b""

            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            audio_segments.append(audio_seg)

            if ZONOS_TTS_SPEAKER_CONTINUITY and audio_tensor is not None and i < len(segments) - 1:
                try:
                    current_speaker_embedding = self.model.make_speaker_embedding(
                        audio_tensor, self.sample_rate
                    )
                except Exception:
                    current_speaker_embedding = None

            if self.torch and self.device == "cuda":
                self.torch.cuda.empty_cache()

        combined_audio = audio_segments[0]
        for audio_seg in audio_segments[1:]:
            if ZONOS_TTS_SEGMENT_GAP_MS > 0:
                silence = AudioSegment.silent(
                    duration=ZONOS_TTS_SEGMENT_GAP_MS, frame_rate=self.sample_rate
                )
                combined_audio = combined_audio + silence + audio_seg
            elif ZONOS_TTS_CROSSFADE_MS > 0:
                crossfade = min(ZONOS_TTS_CROSSFADE_MS, len(combined_audio), len(audio_seg))
                combined_audio = combined_audio.append(audio_seg, crossfade=crossfade)
            else:
                combined_audio = combined_audio + audio_seg

        output_buffer = io.BytesIO()
        combined_audio.export(output_buffer, format=response_format)
        output_buffer.seek(0)
        final_audio_bytes = output_buffer.read()
        print(f"[Zonos_TTS_Handler][legacy] ✓ Combined audio: {len(final_audio_bytes)} bytes")
        return True, "", final_audio_bytes

    def _pydub_fallback_concat(
        self,
        raw_audio_arrays: list,
        response_format: str,
        gap_ms: int,
        crossfade_ms: int,
    ) -> bytes:
        """
        Concatenate raw numpy arrays via pydub as a fallback when spectral
        assembly fails. 스펙트럴 어셈블리 실패 시 pydub을 통한 연결 폴백.
        """
        import numpy as np
        from pydub import AudioSegment

        pydub_segments = []
        for arr in raw_audio_arrays:
            pcm16 = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                data=pcm16.tobytes(),
                sample_width=2,
                frame_rate=self.sample_rate,
                channels=1,
            )
            pydub_segments.append(seg)

        combined = pydub_segments[0]
        for seg in pydub_segments[1:]:
            if gap_ms > 0:
                silence = AudioSegment.silent(duration=gap_ms, frame_rate=self.sample_rate)
                combined = combined + silence + seg
            elif crossfade_ms > 0:
                xf = min(crossfade_ms, len(combined), len(seg))
                combined = combined.append(seg, crossfade=xf)
            else:
                combined = combined + seg

        buf = io.BytesIO()
        combined.export(buf, format=response_format)
        buf.seek(0)
        return buf.read()

    def _apply_pitch_shift(self, audio_tensor, pitch_multiplier: float):
        """
        Apply pitch shift to audio tensor using sample rate conversion.
        샘플 레이트 변환을 사용하여 오디오 텐서에 피치 변경을 적용합니다.

        Args:
            audio_tensor: Audio tensor [channels, samples]
            pitch_multiplier: Pitch multiplier (0.5=lower, 1.0=normal, 2.0=higher)

        Returns:
            Pitch-shifted audio tensor
        """
        if pitch_multiplier == 1.0:
            return audio_tensor

        try:
            # Calculate new sample rate for pitch shift
            # Higher pitch = faster playback = higher sample rate
            original_sr = self.ZONOS_OUTPUT_SAMPLE_RATE
            temp_sr = int(original_sr * pitch_multiplier)

            # Resample to temporary rate (this changes pitch)
            resample_to_temp = self._Resample(orig_freq=original_sr, new_freq=temp_sr)

            # Resample back to original rate (maintains new pitch)
            resample_to_original = self._Resample(orig_freq=temp_sr, new_freq=original_sr)

            # Apply transformations
            pitched_audio = resample_to_temp(audio_tensor)
            pitched_audio = resample_to_original(pitched_audio)

            return pitched_audio

        except Exception as e:
            print(f"[Zonos_TTS_Handler] ⚠️ Pitch shift failed: {e}, using original audio")
            return audio_tensor

    def _tensor_to_wav_bytes(self, audio_tensor) -> bytes:
        """
        Convert audio tensor to WAV bytes.
        오디오 텐서를 WAV 바이트로 변환합니다.

        Uses scipy.io.wavfile for WAV encoding (no torchaudio dependency).
        WAV 인코딩에 scipy.io.wavfile을 사용합니다 (torchaudio 의존성 없음).

        Args:
            audio_tensor: Audio tensor [channels, samples]

        Returns:
            Audio data as bytes (WAV format)
        """
        import numpy as np
        from scipy.io import wavfile

        # Ensure tensor is on CPU
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()

        # Normalize to [-1, 1] if needed
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()

        # Convert to numpy int16 for WAV format
        # scipy.io.wavfile expects [samples] for mono or [samples, channels] for stereo
        # torch tensor is [channels, samples], so transpose for multi-channel
        if audio_tensor.shape[0] == 1:
            audio_np = audio_tensor.squeeze(0).numpy()
        else:
            audio_np = audio_tensor.T.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Write WAV to buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_int16)
        buffer.seek(0)

        return buffer.read()

    def _convert_audio_format(self, audio_data: bytes, from_format: str, to_format: str) -> bytes:
        """
        Convert audio from one format to another using pydub.
        pydub을 사용하여 오디오를 다른 형식으로 변환합니다.

        Args:
            audio_data: Audio data in bytes
            from_format: Source audio format (e.g., 'wav')
            to_format: Target audio format (e.g., 'mp3')

        Returns:
            Converted audio data as bytes
        """
        from pydub import AudioSegment

        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=from_format)

        # Export to target format
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format=to_format)
        output_buffer.seek(0)

        return output_buffer.read()

    def text_to_speech_chunked(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        chunk_duration_ms: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, str, list, int, float]:
        """
        Override to handle format conversion properly.
        포맷 변환을 올바르게 처리하도록 오버라이드합니다.

        Zonos generates WAV internally but can output in any format to match OpenAI.
        Zonos는 내부적으로 WAV를 생성하지만 OpenAI와 동일하게 모든 형식으로 출력할 수 있습니다.
        """
        # Get requested format (default to mp3 to match OpenAI)
        response_format = kwargs.get("response_format", "mp3")

        # Call parent implementation with the requested format
        return super().text_to_speech_chunked(
            text=text, voice=voice, speed=speed, chunk_duration_ms=chunk_duration_ms, **kwargs
        )

    def is_available(self) -> bool:
        """
        Check if the handler is available and ready.
        핸들러가 사용 가능하고 준비되었는지 확인합니다.

        Returns:
            True if available, False otherwise
        """
        return self.model is not None and self._initialized

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about the current handler.
        현재 핸들러에 대한 정보를 반환합니다.

        Returns:
            Dictionary with handler information
        """
        # Get base info
        base_info = super().get_handler_info()

        # Add Zonos specific info
        zonos_info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "speaker_id": self.speaker_id,
            "speed": self.speed,
            "pitch": self.pitch,
            "has_speaker_embedding": self.speaker_embedding is not None,
            "emotion_vector": self.emotion_vector,
            "use_emotion_vector": self.use_emotion_vector,
            "provider": "Zyphra",
            "output_sample_rate": self.ZONOS_OUTPUT_SAMPLE_RATE,
        }

        return {**base_info, **zonos_info}

    def set_speed(self, speed: float) -> bool:
        """
        Set the speech speed.
        말하기 속도를 설정합니다.

        Args:
            speed: Speed multiplier (0.5 to 2.0 recommended)

        Returns:
            True if valid speed
        """
        if 0.1 <= speed <= 3.0:
            self.speed = speed
            return True
        else:
            print(f"[Zonos_TTS_Handler] Invalid speed: {speed} (recommended: 0.5-2.0)")
            return False

    def set_pitch(self, pitch: float) -> bool:
        """
        Set the pitch multiplier.
        피치 배율을 설정합니다.

        Args:
            pitch: Pitch multiplier (0.5 to 2.0 recommended, 1.0=normal)

        Returns:
            True if valid pitch
        """
        if 0.5 <= pitch <= 2.0:
            self.pitch = pitch
            return True
        else:
            print(f"[Zonos_TTS_Handler] Invalid pitch: {pitch} (recommended: 0.5-2.0)")
            return False

    def load_reference_audio(self, audio_path: str) -> bool:
        """
        Load reference audio for voice cloning.
        음성 복제를 위한 참조 오디오를 로드합니다.

        Args:
            audio_path: Path to reference audio file

        Returns:
            True if successful, False otherwise
        """
        self.reference_audio_path = audio_path
        self.use_reference = True
        return self._load_speaker_embedding()
