#!/usr/bin/env python3
"""
STT Node - Huggingface Whisper 기반 음성-텍스트 변환 노드

Subscriber:
    - Publish2STT_Listen_Data (audio_common_msgs/AudioData): 오디오 청크 수신
    - Publish2Agent_STT (std_msgs/Bool): VAD 발화 종료 신호 (False일 때 STT 수행)

Publisher:
    - Publish_STT_Result (std_msgs/String): STT 결과 텍스트
"""

import json
import socket
import time

import librosa
import numpy as np
import rclpy
import torch
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from std_msgs.msg import Bool, String
from transformers import pipeline


class STTNode(Node):
    """
    Huggingface Whisper 기반 STT 노드
    오디오 청크를 수집하고 VAD 발화 종료 신호 시 텍스트로 변환
    """

    # 오디오 설정
    INPUT_SAMPLE_RATE = 48000  # 입력 샘플링 레이트 (VAD와 동일)
    WHISPER_SAMPLE_RATE = 16000  # Whisper 요구사항
    MODEL_NAME = "openai/whisper-large-v3"
    LANGUAGE = "ko"  # 한국어

    # 청킹 설정 (30초 이상 긴 오디오 처리용)
    CHUNK_LENGTH_S = 25  # 25초 청크
    STRIDE_LENGTH_S = 5  # 5초 오버랩 (앞뒤)
    MAX_AUDIO_DURATION = 300  # 최대 5분

    def __init__(self):
        super().__init__("stt_node")

        # 오디오 버퍼 초기화
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False

        # Google STT 제어 변수
        self._should_transcribe = None  # None: 대기, True: 진행, False: 스킵

        # 발화 종료 후 Google STT 제어 신호 대기 상태
        self._waiting_for_google_stt = False

        # Whisper 모델 로드
        self._load_model()

        # Subscriber 생성
        self.audio_subscriber = self.create_subscription(
            AudioData, "Publish2Server_Chunk", self._audio_callback, 10
        )

        self.vad_subscriber = self.create_subscription(
            Bool, "Publish2Agent_STT", self._vad_signal_callback, 10
        )

        # Google STT 제어 신호 구독 추가
        self.google_stt_control_subscriber = self.create_subscription(
            Bool, "Google_STT_Control", self._google_stt_control_callback, 10
        )

        # Publisher 생성
        self.result_publisher = self.create_publisher(String, "Publish_STT_Result", 10)

        # 타이머 생성 (Google STT 제어 신호 처리용)
        self.timer = self.create_timer(0.01, self._check_google_stt_signal)

        self.get_logger().info(f"STT Node initialized with model: {self.MODEL_NAME}")
        self.get_logger().info(f"Device: {self._device}, Language: {self.LANGUAGE}")
        self.get_logger().info(
            "Listening on: Publish2Server_Chunk, Publish2Agent_STT, Google_STT_Control"
        )
        self.get_logger().info("Publishing to: Publish_STT_Result")

    def _load_model(self):
        """Huggingface Whisper pipeline 로드 (자동 청킹 지원)"""
        self.get_logger().info(f"Loading Whisper model: {self.MODEL_NAME}...")

        # GPU/CPU 자동 선택
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Pipeline 생성 (자동 청킹 지원)
            # chunk_length_s와 stride_length_s로 긴 오디오를 자동 분할 처리
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_NAME,
                device=self._device,
                chunk_length_s=self.CHUNK_LENGTH_S,
                stride_length_s=(self.STRIDE_LENGTH_S, self.STRIDE_LENGTH_S),
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )

            self.get_logger().info(
                f"Whisper pipeline loaded successfully! "
                f"(chunk={self.CHUNK_LENGTH_S}s, stride={self.STRIDE_LENGTH_S}s)"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper model: {e}")
            raise

    def _get_local_ip(self) -> str:
        """로컬 IP 주소 자동 감지"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    def _audio_callback(self, msg: AudioData):
        """
        오디오 청크 수신 콜백
        수신된 오디오 데이터를 버퍼에 누적
        """
        try:
            # uint8 배열을 int16으로 변환 (16-bit PCM)
            audio_bytes = bytes(msg.data)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # int16을 float32로 변환 (-1.0 ~ 1.0 범위)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # 버퍼에 누적
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_float])

            # 디버그: 버퍼 상태 로그 (매 100번째 청크마다)
            # if not hasattr(self, '_chunk_count'):
            #     self._chunk_count = 0
            # self._chunk_count += 1
            # if self._chunk_count % 100 == 0:
            #     self.get_logger().info(f"[DEBUG] Audio buffer: {len(self._audio_buffer)} samples ({len(self._audio_buffer)/self.INPUT_SAMPLE_RATE:.2f}s)")

        except Exception as e:
            self.get_logger().error(f"Audio processing error: {e}")

    def _google_stt_control_callback(self, msg: Bool):
        """
        Google STT 제어 신호 수신 콜백
        True: 발화 감지됨 (Whisper transcribe 진행)
        False: 발화 없음 (Whisper skip, 빈 문자열 하드코딩 발행)

        Note: Static Waiting에서는 VAD False 신호 없이 이 신호만 올 수 있으므로
              _waiting_for_google_stt도 True로 설정하여 처리되도록 함
        """
        self._should_transcribe = msg.data
        self._waiting_for_google_stt = True  # Static Waiting 대응
        self.get_logger().info(
            f"Google STT Control: {'Transcribe' if msg.data else 'Skip (no speech)'}"
        )

    def _vad_signal_callback(self, msg: Bool):
        """
        VAD 발화 신호 콜백
        True: 발화 진행중, False: 발화 종료 (STT 수행)
        """
        if msg.data:
            # 발화 시작/진행중
            if not self._is_speaking:
                self._is_speaking = True
                # 새 발화 시작 시 이전 세션의 신호 리셋
                self._should_transcribe = None
                self.get_logger().info("Speech started - collecting audio...")
        else:
            # 발화 종료 - Google STT 제어 신호 대기 상태로 전환
            if self._is_speaking or len(self._audio_buffer) > 0:
                self._is_speaking = False
                # 발화 종료 시 이전 세션의 신호 리셋 후 대기
                self._should_transcribe = None
                self._waiting_for_google_stt = True
                self.get_logger().info(
                    "Speech ended - waiting for Google STT control signal..."
                )

    def _check_google_stt_signal(self):
        """
        타이머 콜백: Google STT 제어 신호 확인 및 처리
        """
        if not self._waiting_for_google_stt:
            return

        if self._should_transcribe is None:
            # 아직 신호 수신 안 됨
            return

        # 신호 수신됨 - 처리
        self._waiting_for_google_stt = False

        if self._should_transcribe is True:
            # 발화 감지됨 - Whisper로 transcribe 진행
            self.get_logger().info(
                "Google STT Control: Transcribe - Performing Whisper STT..."
            )
            self._perform_stt()

        elif self._should_transcribe is False:
            # 발화 없음 - Whisper 스킵하고 빈 문자열 하드코딩 발행
            self.get_logger().info(
                "Google STT Control: Skip - No speech detected, publishing empty string"
            )

            # 버퍼 초기화
            self._audio_buffer = np.array([], dtype=np.float32)

            # 빈 문자열을 STT 결과로 발행 (Whisper 모델 거치지 않음)
            result_msg = String()
            result_data = {
                "ResponseUrl": self._get_local_ip(),
                "resultSTT": "",  # 빈 문자열 하드코딩
            }
            result_msg.data = json.dumps(result_data, ensure_ascii=False)
            self.result_publisher.publish(result_msg)

            self.get_logger().info("Published empty STT result (no speech detected)")

        # 제어 신호 리셋
        self._should_transcribe = None

    def _perform_stt(self):
        """
        수집된 오디오에 대해 STT 수행 및 결과 발행
        Pipeline을 사용하여 긴 오디오도 자동 청킹으로 처리
        """
        if len(self._audio_buffer) == 0:
            self.get_logger().warn("No audio data to transcribe")
            return

        # 최소 오디오 길이 체크 (0.3초 이상)
        min_audio_duration = 0.3  # 초
        audio_duration = len(self._audio_buffer) / self.INPUT_SAMPLE_RATE

        if audio_duration < min_audio_duration:
            self.get_logger().warn(
                f"Audio too short ({audio_duration:.2f}s < {min_audio_duration}s), "
                "likely robot speech echo or noise - skipping STT"
            )
            # 버퍼 초기화
            self._audio_buffer = np.array([], dtype=np.float32)
            return

        try:
            # 48kHz -> 16kHz 리샘플링
            audio_resampled = librosa.resample(
                self._audio_buffer,
                orig_sr=self.INPUT_SAMPLE_RATE,
                target_sr=self.WHISPER_SAMPLE_RATE,
            )

            # 최대 길이 제한 (5분)
            max_samples = self.MAX_AUDIO_DURATION * self.WHISPER_SAMPLE_RATE
            if len(audio_resampled) > max_samples:
                self.get_logger().warn(
                    f"Audio exceeds max duration ({self.MAX_AUDIO_DURATION}s), truncating..."
                )
                audio_resampled = audio_resampled[:max_samples]

            resampled_duration = len(audio_resampled) / self.WHISPER_SAMPLE_RATE
            self.get_logger().info(
                f"Audio length: {audio_duration:.2f}s "
                f"-> {resampled_duration:.2f}s (resampled)"
            )

            # Pipeline으로 transcribe (자동 청킹)
            # 긴 오디오도 chunk_length_s와 stride_length_s 설정으로 자동 분할 처리
            result = self._pipe(
                {"raw": audio_resampled, "sampling_rate": self.WHISPER_SAMPLE_RATE},
                generate_kwargs={"language": "korean", "task": "transcribe"},
                return_timestamps=False,
            )
            transcription = result["text"].strip()

            # transcription을 초록색으로만 출력
            self.get_logger().info(f"STT Result: \033[92m{transcription}\033[0m")

            # 결과 발행 (Message Handler가 기대하는 JSON 형식)
            result_msg = String()
            result_data = {
                "ResponseUrl": self._get_local_ip(),  # 이 서버의 IP
                "resultSTT": transcription,
            }
            result_msg.data = json.dumps(result_data, ensure_ascii=False)
            self.result_publisher.publish(result_msg)

            self.get_logger().info(f"Published STT result to Publish_STT_Result")

        except Exception as e:
            self.get_logger().error(f"STT error: {e}")

        finally:
            # 버퍼 초기화
            self._audio_buffer = np.array([], dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)

    stt_node = STTNode()

    try:
        rclpy.spin(stt_node)
    except KeyboardInterrupt:
        pass
    finally:
        stt_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
