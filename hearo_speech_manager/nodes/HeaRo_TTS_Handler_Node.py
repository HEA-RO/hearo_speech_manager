#!/usr/bin/env python3
"""
HeaRo TTS Handler Node
HeaRo TTS 핸들러 노드

A ROS2 node that provides Text-to-Speech (TTS) services using multiple backends.
Supports OpenAI TTS and Zonos TTS models with automatic backend selection.

다중 백엔드를 사용하여 TTS(텍스트 음성 변환) 서비스를 제공하는 ROS2 노드입니다.
OpenAI TTS 및 Zonos TTS 모델을 자동 백엔드 선택과 함께 지원합니다.

Topics Subscribed:
- Request2TTS: TTS generation requests (from HeaRo_LLM_Distributor or other nodes)

Topics Published:
- Response2TTS: TTS generation response with metadata
- Response2TTS_Chunk: Audio chunks for streaming

Services Provided:
- Service_TTS_Generate: Synchronous TTS generation service
"""

import rclpy
from rclpy.node import Node
import json
import os
import base64
import time
from typing import List, Dict, Any, Optional, Tuple

# Import configuration
from hearo_speech_manager.config.speech_config import (
    CHUNK_DURATION_MS,
    OPENAI_TTS_SAMPLE_RATE,
    ZONOS_TTS_SAMPLE_RATE,
    DEFAULT_TTS_MODEL,
    TTS_MODEL_OPENAI,
    TTS_MODEL_ZONOS_TRANSFORMER,
    TTS_MODEL_ZONOS_HYBRID,
    get_tts_config,
)

# Import utility functions
from hearo_speech_manager.config.speech_utility import (
    load_error_codes,
    get_error_info,
    get_detailed_error_message,
    log_error_with_details,
    AnsiColor,
    GracefulShutdownMixin,
)

# Import message types
from std_msgs.msg import String

# Import custom audio chunk message
try:
    from hearo_message_handler.msg import AudioDataChunk

    AUDIO_CHUNK_AVAILABLE = True
except ImportError:
    AUDIO_CHUNK_AVAILABLE = False

# Import service definitions
from hearo_speech_manager.srv import TTSGenerate, GetOrCreateAudioId

# Import TTS handler
from hearo_speech_manager.module.HeaRo_TTS_Handler import HeaRo_TTS_Handler

# Import new modules for GetOrCreateAudioId support
from hearo_speech_manager.module.audio_cache_manager import AudioCacheManager
from hearo_speech_manager.module.lazy_gpu_session_pool import LazyGPUSessionPool


class HeaRo_TTS_Handler_Node(GracefulShutdownMixin, Node):
    """
    ROS2 Node for Text-to-Speech generation using multiple backends.
    다중 백엔드를 사용한 텍스트 음성 변환 ROS2 노드입니다.

    Provides:
    제공 기능:
    - Service-based TTS generation (Service_TTS_Generate)
      서비스 기반 TTS 생성
    - Topic-based TTS generation for backward compatibility
      하위 호환성을 위한 토픽 기반 TTS 생성
    - Audio chunking for streaming playback
      스트리밍 재생을 위한 오디오 청킹
    - Multiple backend support (OpenAI, Zonos)
      다중 백엔드 지원 (OpenAI, Zonos)
    """

    def __init__(self):
        super().__init__("HeaRo_TTS_Handler")
        self._init_graceful_shutdown()
        _init_start_time = time.time()

        # OpenAI API Key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # Initialize TTS handler
        self.tts_handler = None
        self._initialize_tts_handler()

        # Error codes dictionary
        self.error_codes = {}
        self._load_error_codes()

        # Audio cache manager (pymysql direct access to tb_audioCache)
        self._cache_manager = AudioCacheManager(logger=self.get_logger())

        # Lazy GPU session pool (idle timeout-based GPU lifecycle)
        self._gpu_pool = LazyGPUSessionPool(
            self.tts_handler,
            idle_timeout_sec=30.0,
            logger=self.get_logger(),
        )

        # Create GetOrCreateAudioId service
        self.audio_cache_service = self.create_service(
            GetOrCreateAudioId,
            "Service_GetOrCreateAudioId",
            self.handle_get_or_create_audio_id,
        )

        # Create TTS service
        self.tts_service = self.create_service(
            TTSGenerate, "Service_TTS_Generate", self.handle_tts_service_request
        )

        # Create subscriber for topic-based TTS requests (new topic)
        self.tts_request_sub = self.create_subscription(
            String, "Request2TTS", self.handle_tts_topic_request, 150
        )

        # Create subscriber for legacy TTS requests (backward compatibility with LLM Distributor topic)
        self.legacy_tts_request_sub = self.create_subscription(
            String, "Request2LLM_TTS", self.handle_legacy_tts_topic_request, 150
        )

        # Create publishers for legacy TTS responses (backward compatibility)
        self.legacy_tts_response_pub = self.create_publisher(String, "Response2LLM_TTS", 150)

        # Create publisher for legacy TTS audio chunks
        if AUDIO_CHUNK_AVAILABLE:
            self.legacy_tts_chunk_pub = self.create_publisher(
                AudioDataChunk, "Response2LLM_TTS_Chunk", 150
            )
        else:
            self.legacy_tts_chunk_pub = self.create_publisher(String, "Response2LLM_TTS_Chunk", 150)

        # Create publishers for TTS responses
        self.tts_response_pub = self.create_publisher(String, "Response2TTS", 150)

        # Create publisher for TTS audio chunks
        if AUDIO_CHUNK_AVAILABLE:
            self.tts_chunk_pub = self.create_publisher(AudioDataChunk, "Response2TTS_Chunk", 150)
            self._log_info("TTS chunk publishing with AudioDataChunk enabled")
        else:
            self.tts_chunk_pub = self.create_publisher(String, "Response2TTS_Chunk", 150)
            self._log_warn("Using legacy String format for TTS chunk publishing")

        self._log_info("HeaRo TTS Handler Node initialized")
        self._log_available_backends()

        # ── Publish ready status to HeaRo_Package_Monitor (optional) ─
        try:
            from hearo_knowledge_manager.module.logging_helper import publish_node_ready_status

            _tts_loaded = self.tts_handler is not None
            _available_backends = self.tts_handler.get_available_backends() if _tts_loaded else []
            _openai_loaded = any("openai" in b.lower() for b in _available_backends)
            publish_node_ready_status(
                self,
                "HeaRo_TTS_Handler",
                [
                    {"name": "TTS Handler", "loaded": _tts_loaded},
                    {"name": "OpenAI TTS Backend", "loaded": _openai_loaded},
                    {"name": "AudioDataChunk Support", "loaded": AUDIO_CHUNK_AVAILABLE},
                    {"name": "Service: TTS Generate", "loaded": True},
                    {"name": "Sub: Request2TTS", "loaded": True},
                    {"name": "Sub: Request2LLM_TTS (legacy)", "loaded": True},
                ],
                elapsed_sec=time.time() - _init_start_time,
            )
        except ImportError:
            self._log_info(f"Package monitor integration skipped (elapsed: {time.time() - _init_start_time:.2f}s)")

    # ========== Logging Helper Methods ==========

    def _log_info(self, message: str, color: str = None):
        """Log info message with optional color."""
        if color:
            try:
                color_code = AnsiColor[color.upper()].value
                reset_code = AnsiColor.RESET.value
                styled_message = f"{color_code}{message}{reset_code}"
            except (KeyError, AttributeError):
                styled_message = message
        else:
            styled_message = message
        self.get_logger().info(styled_message)

    def _log_warn(self, message: str, color: str = "YELLOW"):
        """Log warning message with color (default: YELLOW)."""
        try:
            color_code = AnsiColor[color.upper()].value
            reset_code = AnsiColor.RESET.value
            styled_message = f"{color_code}{message}{reset_code}"
        except (KeyError, AttributeError):
            styled_message = message
        self.get_logger().warn(styled_message)

    def _log_error(self, message: str, color: str = "RED"):
        """Log error message with color (default: RED)."""
        try:
            color_code = AnsiColor[color.upper()].value
            reset_code = AnsiColor.RESET.value
            styled_message = f"{color_code}{message}{reset_code}"
        except (KeyError, AttributeError):
            styled_message = message
        self.get_logger().error(styled_message)

    # ========== Initialization Methods ==========

    def _initialize_tts_handler(self):
        """Initialize TTS handler with all backends."""
        try:
            self._log_info("Initializing TTS handler...")
            self.tts_handler = HeaRo_TTS_Handler(
                openai_api_key=self.openai_api_key, chunk_duration_ms=CHUNK_DURATION_MS
            )

            available_backends = self.tts_handler.get_available_backends()
            if available_backends:
                self._log_info(f"TTS handler initialized with backends: {available_backends}")
            else:
                self._log_warn("TTS handler initialized but no backends available")

        except Exception as e:
            self._log_error(f"Failed to initialize TTS handler: {str(e)}")

    def _load_error_codes(self):
        """Load error codes from CSV file."""
        self.error_codes = load_error_codes()
        self._log_info(f"Loaded {len(self.error_codes)} error codes")

    def _log_available_backends(self):
        """Log available TTS backends."""
        if not self.tts_handler:
            self._log_warn("TTS handler not initialized")
            return

        self._log_info("=== Available TTS Backends ===")
        backends = self.tts_handler.get_available_backends()
        for backend in backends:
            self._log_info(f"  ✓ {backend}")

        all_backends = self.tts_handler.get_all_backends()
        for backend in all_backends:
            if backend not in backends:
                self._log_warn(f"  ✗ {backend} (not available)")

    # ========== Service Handler ==========

    def handle_tts_service_request(self, request, response):
        """
        Handle TTS generation service request.
        TTS 생성 서비스 요청을 처리합니다.

        Args:
            request: TTSGenerate request
            response: TTSGenerate response

        Returns:
            Populated response object
        """
        start_time = time.time()

        try:
            self._log_info("=" * 80, color="BRIGHT_CYAN")
            self._log_info("[TTS Handler] Received Service Request", color="BRIGHT_CYAN")
            self._log_info(
                (
                    f"  - Text: {request.text[:50]}..."
                    if len(request.text) > 50
                    else f"  - Text: {request.text}"
                ),
                color="BRIGHT_CYAN",
            )
            self._log_info(f"  - Backend: {request.backend}", color="BRIGHT_CYAN")
            self._log_info(f"  - Speed: {request.speed}", color="BRIGHT_CYAN")
            self._log_info(f"  - Language: {request.language}", color="BRIGHT_CYAN")
            self._log_info("=" * 80, color="BRIGHT_CYAN")

            # Check TTS handler
            if not self.tts_handler:
                response.success = False
                response.error_code = "TTS001"
                response.error_message = "TTS handler not initialized"
                return response

            with self.tts_handler.gpu_session():
                # Determine backend
                backend = request.backend if request.backend else "auto"

                # Determine speed
                speed = request.speed if request.speed > 0 else None

                # Generate TTS with chunking
                success, error_msg, chunks, n_chunks, duration, model_used = (
                    self.tts_handler.text_to_speech_chunked(
                        text=request.text,
                        backend=backend,
                        speed=speed,
                        voice=request.voice if request.voice else None,
                        language=request.language if request.language else "ko",
                    )
                )

                elapsed = time.time() - start_time

                if not success:
                    response.success = False
                    response.error_code = "TTS002"
                    response.error_message = f"TTS generation failed: {error_msg}"
                    self._log_error(f"TTS generation failed: {error_msg}")
                    return response

                # Build successful response
                response.success = True
                response.error_code = ""
                response.error_message = ""
                response.backend_used = model_used
                response.n_chunks = n_chunks
                response.n_sent_chunks = 1
                response.audio_duration_sec = float(duration)
                response.audio_id = 0
                response.audio_format = "mp3" if "openai" in model_used else "wav"

                if request.return_audio_data:
                    full_audio = b"".join(chunks)
                    response.audio_data = list(full_audio)
                else:
                    response.audio_data = []

                self._log_info("=" * 80, color="BRIGHT_CYAN")
                self._log_info("[TTS Handler] Sending Service Response", color="BRIGHT_CYAN")
                self._log_info(f"  - Success: {response.success}", color="BRIGHT_CYAN")
                self._log_info(f"  - Backend Used: {response.backend_used}", color="BRIGHT_CYAN")
                self._log_info(f"  - Chunks: {response.n_chunks}", color="BRIGHT_CYAN")
                self._log_info(
                    f"  - Duration: {response.audio_duration_sec:.2f}s", color="BRIGHT_CYAN"
                )
                self._log_info(f"  - Processing Time: {elapsed:.2f}s", color="BRIGHT_CYAN")
                self._log_info("=" * 80, color="BRIGHT_CYAN")

            return response

        except Exception as e:
            response.success = False
            response.error_code = "TTS003"
            response.error_message = f"TTS service error: {str(e)}"
            self._log_error(f"TTS service error: {str(e)}")
            import traceback

            traceback.print_exc()
            return response

    # ========== Topic Handler ==========

    def handle_tts_topic_request(self, msg):
        """
        Handle TTS request from topic (for backward compatibility).
        토픽으로부터의 TTS 요청을 처리합니다 (하위 호환성).

        Expected message format:
        {
            "Request": {
                "Topic": "TTS",
                "Contents": {
                    "Text": "안녕하세요",
                    "Backend": "auto",  # optional
                    "Speed": 1.0,       # optional
                    "Voice": "alloy"    # optional
                }
            }
        }
        """
        try:
            # Parse request message
            request_data = json.loads(msg.data)

            if "Request" not in request_data:
                self._log_error('Invalid TTS request: missing "Request" field')
                return

            request = request_data["Request"]

            if request.get("Topic") != "TTS":
                self._log_error(f'Invalid TTS request topic: {request.get("Topic")}')
                return

            if "Contents" not in request or "Text" not in request["Contents"]:
                self._log_error("Invalid TTS request: missing text content")
                return

            contents = request["Contents"]
            text = contents["Text"]
            backend = contents.get("Backend", "auto")
            speed = contents.get("Speed", None)
            voice = contents.get("Voice", None)
            language = contents.get("Language", "ko")

            self._log_info(
                f'Processing TTS topic request: "{text[:50]}..."'
                if len(text) > 50
                else f'Processing TTS topic request: "{text}"'
            )

            # Check TTS handler
            if not self.tts_handler:
                self._log_error("TTS handler not initialized")
                return

            with self.tts_handler.gpu_session():
                success, error_msg, chunks, n_chunks, duration, model_used = (
                    self.tts_handler.text_to_speech_chunked(
                        text=text, backend=backend, speed=speed, voice=voice, language=language
                    )
                )

                if not success:
                    self._log_error(f"TTS generation failed: {error_msg}")
                    self._publish_tts_error_response(request_data, error_msg)
                    return

                self._log_info(
                    f"TTS generated: {n_chunks} chunks, {duration:.2f}s duration, backend: {model_used}"
                )

                audio_format = "mp3" if "openai" in model_used else "wav"
                sample_rate = (
                    OPENAI_TTS_SAMPLE_RATE if "openai" in model_used else ZONOS_TTS_SAMPLE_RATE
                )

                response_msg = String()
                response_data = {
                    "Request": request_data["Request"],
                    "Response": {
                        "n_chunk": n_chunks,
                        "backend_used": model_used,
                        "audio_duration_sec": duration,
                        "audio_format": audio_format,
                    },
                }
                response_msg.data = json.dumps(response_data, ensure_ascii=False)
                self.tts_response_pub.publish(response_msg)

                self._log_info(f"Published TTS response with {n_chunks} chunks")

                self._publish_audio_chunks(chunks, n_chunks, audio_format, sample_rate)

                self._log_info("All TTS chunks published")

        except json.JSONDecodeError as e:
            self._log_error(f"Failed to parse TTS request JSON: {str(e)}")
        except Exception as e:
            self._log_error(f"TTS topic request handling failed: {str(e)}")
            import traceback

            traceback.print_exc()

    def _publish_tts_error_response(self, request_data: dict, error_msg: str):
        """Publish TTS error response."""
        response_msg = String()
        response_data = {
            "Request": request_data.get("Request", {}),
            "Response": {"success": False, "error": error_msg, "n_chunk": 0},
        }
        response_msg.data = json.dumps(response_data, ensure_ascii=False)
        self.tts_response_pub.publish(response_msg)

    def handle_legacy_tts_topic_request(self, msg):
        """
        Handle legacy TTS request from Request2LLM_TTS topic (for backward compatibility).
        기존 Request2LLM_TTS 토픽으로부터의 TTS 요청을 처리합니다 (하위 호환성).

        This handler publishes to Response2LLM_TTS and Response2LLM_TTS_Chunk
        to maintain compatibility with existing systems.
        """
        try:
            # Parse request message
            request_data = json.loads(msg.data)

            if "Request" not in request_data:
                self._log_error('Invalid legacy TTS request: missing "Request" field')
                return

            request = request_data["Request"]

            if request.get("Topic") != "TTS":
                self._log_error(f'Invalid legacy TTS request topic: {request.get("Topic")}')
                return

            if "Contents" not in request or "Text" not in request["Contents"]:
                self._log_error("Invalid legacy TTS request: missing text content")
                return

            contents = request["Contents"]
            text = contents["Text"]
            backend = contents.get("Backend", "auto")
            speed = contents.get("Speed", None)
            voice = contents.get("Voice", None)
            language = contents.get("Language", "ko")

            self._log_info(
                f'[Legacy] Processing TTS request: "{text[:50]}..."'
                if len(text) > 50
                else f'[Legacy] Processing TTS request: "{text}"'
            )

            # Check TTS handler
            if not self.tts_handler:
                self._log_error("TTS handler not initialized")
                return

            with self.tts_handler.gpu_session():
                success, error_msg, chunks, n_chunks, duration, model_used = (
                    self.tts_handler.text_to_speech_chunked(
                        text=text, backend=backend, speed=speed, voice=voice, language=language
                    )
                )

                if not success:
                    self._log_error(f"TTS generation failed: {error_msg}")
                    response_msg = String()
                    response_data = {
                        "Request": request_data["Request"],
                        "Response": {"success": False, "error": error_msg, "n_chunk": 0},
                    }
                    response_msg.data = json.dumps(response_data, ensure_ascii=False)
                    self.legacy_tts_response_pub.publish(response_msg)
                    return

                self._log_info(
                    f"[Legacy] TTS generated: {n_chunks} chunks, {duration:.2f}s duration, backend: {model_used}"
                )

                audio_format = "mp3" if "openai" in model_used else "wav"
                sample_rate = (
                    OPENAI_TTS_SAMPLE_RATE if "openai" in model_used else ZONOS_TTS_SAMPLE_RATE
                )

                response_msg = String()
                response_data = {
                    "Request": request_data["Request"],
                    "Response": {
                        "n_chunk": n_chunks,
                        "backend_used": model_used,
                        "audio_duration_sec": duration,
                        "audio_format": audio_format,
                    },
                }
                response_msg.data = json.dumps(response_data, ensure_ascii=False)
                self.legacy_tts_response_pub.publish(response_msg)

                self._log_info(f"[Legacy] Published TTS response with {n_chunks} chunks")

                self._publish_legacy_audio_chunks(chunks, n_chunks, audio_format, sample_rate)

                self._log_info("[Legacy] All TTS chunks published")

        except json.JSONDecodeError as e:
            self._log_error(f"Failed to parse legacy TTS request JSON: {str(e)}")
        except Exception as e:
            self._log_error(f"Legacy TTS topic request handling failed: {str(e)}")
            import traceback

            traceback.print_exc()

    def _publish_legacy_audio_chunks(
        self, chunks: List[bytes], n_chunks: int, audio_format: str, sample_rate: float
    ):
        """Publish audio chunks to legacy topic (Response2LLM_TTS_Chunk)."""
        if AUDIO_CHUNK_AVAILABLE:
            # Use AudioDataChunk message type
            for i, chunk in enumerate(chunks):
                chunk_msg = AudioDataChunk()
                chunk_msg.data = list(chunk)  # Convert bytes to list of uint8
                chunk_msg.chunk_index = i
                chunk_msg.total_chunks = n_chunks
                chunk_msg.audio_format = audio_format
                chunk_msg.sample_rate = float(sample_rate)
                chunk_msg.channels = 1  # Mono audio

                self.legacy_tts_chunk_pub.publish(chunk_msg)
        else:
            # Use legacy String/JSON format
            for i, chunk in enumerate(chunks):
                chunk_b64 = base64.b64encode(chunk).decode("utf-8")

                chunk_msg = String()
                chunk_data = {
                    "chunk_index": i,
                    "total_chunks": n_chunks,
                    "audio_data": chunk_b64,
                    "audio_format": audio_format,
                }
                chunk_msg.data = json.dumps(chunk_data, ensure_ascii=False)
                self.legacy_tts_chunk_pub.publish(chunk_msg)

    def _publish_audio_chunks(
        self, chunks: List[bytes], n_chunks: int, audio_format: str, sample_rate: float
    ):
        """Publish audio chunks to topic."""
        if AUDIO_CHUNK_AVAILABLE:
            # Use AudioDataChunk message type
            for i, chunk in enumerate(chunks):
                chunk_msg = AudioDataChunk()
                chunk_msg.data = list(chunk)  # Convert bytes to list of uint8
                chunk_msg.chunk_index = i
                chunk_msg.total_chunks = n_chunks
                chunk_msg.audio_format = audio_format
                chunk_msg.sample_rate = float(sample_rate)
                chunk_msg.channels = 1  # Mono audio

                self.tts_chunk_pub.publish(chunk_msg)

                self._log_info(
                    f"Published TTS chunk {i+1}/{n_chunks} ({len(chunk)} bytes, AudioDataChunk format)"
                )
        else:
            # Use legacy String/JSON format
            for i, chunk in enumerate(chunks):
                chunk_b64 = base64.b64encode(chunk).decode("utf-8")

                chunk_msg = String()
                chunk_data = {
                    "chunk_index": i,
                    "total_chunks": n_chunks,
                    "audio_data": chunk_b64,
                    "audio_format": audio_format,
                }
                chunk_msg.data = json.dumps(chunk_data, ensure_ascii=False)
                self.tts_chunk_pub.publish(chunk_msg)

                self._log_info(f"Published TTS chunk (legacy) {i+1}/{n_chunks}")

    # ========== Node Lifecycle ==========

    # ========== GetOrCreateAudioId Service Handler ==========

    def handle_get_or_create_audio_id(self, request, response):
        """
        Service_GetOrCreateAudioId 핸들러.
        캐시 조회 → (미스 시) TTS 생성 → DB 저장 → audio_id 반환.
        """
        text = request.text
        if not text or not text.strip():
            response.success = False
            response.error_code = "TTS002"
            response.error_message = "Empty text"
            return response

        try:
            force_regen = request.force_regeneration

            if not force_regen:
                existing_id = self._cache_manager.check_cache(text)
                if existing_id is not None:
                    self._log_info(f"[GetOrCreateAudioId] Cache hit: AudioID={existing_id}")
                    response.success = True
                    response.audio_id = existing_id
                    response.cache_hit = True
                    return response

            backend = request.backend if request.backend else "auto"
            speed = request.speed if request.speed > 0 else None
            voice = request.voice if request.voice else None
            language = request.language if request.language else "ko"

            self._gpu_pool.ensure_loaded()

            tts_result = self.tts_handler.text_to_speech(
                text=text,
                backend=backend,
                speed=speed,
                voice=voice,
                language=language,
            )

            if not tts_result or not tts_result[0]:
                error_msg = tts_result[1] if tts_result and len(tts_result) > 1 else "Unknown TTS error"
                response.success = False
                response.error_code = "TTS002"
                response.error_message = str(error_msg)
                self._gpu_pool.mark_done()
                return response

            success_flag = tts_result[0]
            audio_bytes = tts_result[1]
            model_used = tts_result[2] if len(tts_result) > 2 else backend

            if not audio_bytes:
                response.success = False
                response.error_code = "TTS002"
                response.error_message = "TTS produced empty audio"
                self._gpu_pool.mark_done()
                return response

            self._gpu_pool.mark_done()

            audio_id, n_chunks, duration_sec = self._cache_manager.save_to_cache(
                text, audio_bytes, str(model_used)
            )

            response.success = True
            response.audio_id = audio_id
            response.n_chunks = n_chunks
            response.audio_duration_sec = duration_sec
            response.backend_used = str(model_used)
            response.cache_hit = False

            self._log_info(
                f"[GetOrCreateAudioId] Generated: AudioID={audio_id}, "
                f"chunks={n_chunks}, duration={duration_sec:.2f}s, backend={model_used}"
            )
            return response

        except Exception as e:
            self._log_error(f"[GetOrCreateAudioId] Error: {e}")
            response.success = False
            response.error_code = "TTS002"
            response.error_message = str(e)
            return response

    # ========== Lifecycle ==========

    def run(self):
        """Main execution loop."""
        self._log_info("HeaRo TTS Handler is running")
        rclpy.spin(self)

    def _graceful_cleanup(self):
        """Release TTS GPU resources on Publish_HeaRo_Process_Kill."""
        if hasattr(self, '_gpu_pool'):
            self._gpu_pool.release()
        if self.tts_handler:
            self.tts_handler.release_tts_gpu()

    def destroy_node(self):
        """Clean up resources including GPU models."""
        if hasattr(self, '_gpu_pool'):
            self._gpu_pool.release()
        if self.tts_handler:
            self.tts_handler.release_tts_gpu()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = HeaRo_TTS_Handler_Node()
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
