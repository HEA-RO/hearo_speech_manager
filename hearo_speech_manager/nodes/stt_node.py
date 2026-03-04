#!/usr/bin/env python3
"""
STT Node - 다중 백엔드 지원 음성-텍스트 변환 노드

오디오 버퍼링, VAD 신호 처리, Google STT 제어 등 오케스트레이션을 담당합니다.
실제 음성 변환은 BaseSTTHandler 구현체에 위임합니다.

Launch 시점에 HEARO_STT_BACKEND 환경변수로 백엔드를 선택합니다:
  - "whisper" (기본): Huggingface Whisper
  - "clova" (향후): CLOVA Speech

Subscriber:
    - Publish2Server_Chunk (audio_common_msgs/AudioData): 오디오 청크 수신
    - Publish2Agent_STT (std_msgs/Bool): VAD 발화 종료 신호
    - Google_STT_Control (std_msgs/Bool): Google STT 제어 신호

Publisher:
    - Publish_STT_Result (std_msgs/String): STT 결과 텍스트
"""

import json
import socket

import numpy as np
import rclpy
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from std_msgs.msg import Bool, String

from hearo_speech_manager.config.speech_config import (
    DEFAULT_STT_BACKEND,
    STT_INPUT_SAMPLE_RATE,
    WHISPER_MIN_AUDIO_DURATION,
)
from hearo_speech_manager.module.Base_STT_Handler import create_stt_handler


class STTNode(Node):
    """
    다중 백엔드 지원 STT 노드.
    오디오 수집 및 제어 로직(오케스트레이션)을 담당하며,
    모델 추론은 BaseSTTHandler 구현체에 위임합니다.
    """

    INPUT_SAMPLE_RATE = STT_INPUT_SAMPLE_RATE

    def __init__(self):
        super().__init__("stt_node")

        # STT 핸들러 생성 (환경변수로 백엔드 결정)
        self._handler = create_stt_handler(DEFAULT_STT_BACKEND)

        # 오디오 버퍼 초기화
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False

        # Google STT 제어 변수
        self._should_transcribe = None
        self._waiting_for_google_stt = False

        # Subscriber 생성
        self.audio_subscriber = self.create_subscription(
            AudioData, "Publish2Server_Chunk", self._audio_callback, 10
        )

        self.vad_subscriber = self.create_subscription(
            Bool, "Publish2Agent_STT", self._vad_signal_callback, 10
        )

        self.google_stt_control_subscriber = self.create_subscription(
            Bool, "Google_STT_Control", self._google_stt_control_callback, 10
        )

        # Publisher 생성
        self.result_publisher = self.create_publisher(String, "Publish_STT_Result", 10)

        # 타이머 생성 (Google STT 제어 신호 처리용)
        self.timer = self.create_timer(0.01, self._check_google_stt_signal)

        self.get_logger().info(
            f"STT Node initialized | backend: {self._handler.BACKEND_NAME} "
            f"({self._handler.BACKEND_TYPE})"
        )
        self.get_logger().info(
            "Listening on: Publish2Server_Chunk, Publish2Agent_STT, Google_STT_Control"
        )
        self.get_logger().info("Publishing to: Publish_STT_Result")

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
        """오디오 청크 수신 → 버퍼에 누적"""
        try:
            audio_bytes = bytes(msg.data)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_float])
        except Exception as e:
            self.get_logger().error(f"Audio processing error: {e}")

    def _google_stt_control_callback(self, msg: Bool):
        """
        Google STT 제어 신호 수신 콜백
        True: 발화 감지됨 (transcribe 진행)
        False: 발화 없음 (skip, 빈 문자열 발행)
        """
        self._should_transcribe = msg.data
        self._waiting_for_google_stt = True
        self.get_logger().info(
            f"Google STT Control: {'Transcribe' if msg.data else 'Skip (no speech)'}"
        )

    def _vad_signal_callback(self, msg: Bool):
        """
        VAD 발화 신호 콜백
        True: 발화 진행중, False: 발화 종료
        """
        if msg.data:
            if not self._is_speaking:
                self._is_speaking = True
                self._should_transcribe = None
                self.get_logger().info("Speech started - collecting audio...")
        else:
            if self._is_speaking or len(self._audio_buffer) > 0:
                self._is_speaking = False
                self._should_transcribe = None
                self._waiting_for_google_stt = True
                self.get_logger().info(
                    "Speech ended - waiting for Google STT control signal..."
                )

    def _check_google_stt_signal(self):
        """타이머 콜백: Google STT 제어 신호 확인 및 처리"""
        if not self._waiting_for_google_stt:
            return

        if self._should_transcribe is None:
            return

        self._waiting_for_google_stt = False

        if self._should_transcribe is True:
            self.get_logger().info(
                f"Google STT Control: Transcribe - Performing {self._handler.BACKEND_NAME} STT..."
            )
            self._perform_stt()

        elif self._should_transcribe is False:
            self.get_logger().info(
                "Google STT Control: Skip - No speech detected, publishing empty string"
            )
            self._audio_buffer = np.array([], dtype=np.float32)

            result_msg = String()
            result_data = {
                "ResponseUrl": self._get_local_ip(),
                "resultSTT": "",
            }
            result_msg.data = json.dumps(result_data, ensure_ascii=False)
            self.result_publisher.publish(result_msg)
            self.get_logger().info("Published empty STT result (no speech detected)")

        self._should_transcribe = None

    def _perform_stt(self):
        """수집된 오디오에 대해 STT 수행 및 결과 발행"""
        if len(self._audio_buffer) == 0:
            self.get_logger().warn("No audio data to transcribe")
            return

        audio_duration = len(self._audio_buffer) / self.INPUT_SAMPLE_RATE
        if audio_duration < WHISPER_MIN_AUDIO_DURATION:
            self.get_logger().warn(
                f"Audio too short ({audio_duration:.2f}s < {WHISPER_MIN_AUDIO_DURATION}s), "
                "likely robot speech echo or noise - skipping STT"
            )
            self._audio_buffer = np.array([], dtype=np.float32)
            return

        try:
            transcription = self._handler.transcribe(
                self._audio_buffer, self.INPUT_SAMPLE_RATE
            )

            self.get_logger().info(f"STT Result: \033[92m{transcription}\033[0m")

            result_msg = String()
            result_data = {
                "ResponseUrl": self._get_local_ip(),
                "resultSTT": transcription,
            }
            result_msg.data = json.dumps(result_data, ensure_ascii=False)
            self.result_publisher.publish(result_msg)

            self.get_logger().info("Published STT result to Publish_STT_Result")

        except Exception as e:
            self.get_logger().error(f"STT error: {e}")

        finally:
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
