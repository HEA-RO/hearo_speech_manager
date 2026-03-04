#!/usr/bin/env python3

import wave
from datetime import datetime
from pathlib import Path

import pyaudio
import rclpy
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from std_msgs.msg import Bool


class AudioInputTestNode(Node):
    """
    테스트용 오디오 입력 노드

    마이크에서 오디오를 받아서 Publish2Server_Chunk 토픽으로 발행합니다.
    외부 패키지 없이 VAD/STT 테스트를 위해 사용됩니다.

    디버깅용 녹음 기능:
    - 발화 세션별로 입력 오디오를 WAV 파일로 자동 저장
    - VAD가 침묵을 감지하여 세션이 종료되면 파일 저장 완료
    - 저장 경로: test/recordings/session_YYYY-MM-DD_HH-MM-SS.wav
    """

    # 오디오 설정 (VAD와 동일하게 설정)
    SAMPLE_RATE = 48000  # 48kHz
    CHANNELS = 1  # 모노
    CHUNK_DURATION_MS = 30  # 30ms
    FORMAT = pyaudio.paInt16  # 16-bit

    def __init__(self):
        super().__init__("audio_input_test_node")

        # 오디오 설정
        self.chunk_size = int(self.SAMPLE_RATE * self.CHUNK_DURATION_MS / 1000.0)

        # 녹음 관련 변수
        self.recordings_dir = Path(__file__).parent / "recordings"
        self.recordings_dir.mkdir(exist_ok=True)

        self.wav_file = None
        self.recording_path = None
        self.is_recording = False
        self.session_count = 0

        # Publisher 생성
        self.audio_publisher = self.create_publisher(
            AudioData, "Publish2Server_Chunk", 10
        )

        # VAD 신호 Subscriber 생성 (세션 시작/종료 감지)
        self.vad_signal_subscriber = self.create_subscription(
            Bool, "Publish2Agent_STT", self.vad_signal_callback, 10
        )

        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()

        # 사용 가능한 오디오 장치 찾기
        self._find_audio_device()

        # 오디오 스트림 열기
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
            )

            self.get_logger().info(
                "\n"
                "=" * 70 + "\n"
                "  오디오 입력 테스트 노드 시작 (세션별 녹음)\n"
                "=" * 70 + "\n"
                f"  마이크 장치: {self.device_name}\n"
                f"  샘플링 레이트: {self.SAMPLE_RATE} Hz\n"
                f"  채널: {self.CHANNELS}\n"
                f"  청크 크기: {self.chunk_size} 샘플 ({self.CHUNK_DURATION_MS}ms)\n"
                f"  발행 토픽: Publish2Server_Chunk\n"
                f"  녹음 디렉터리: {self.recordings_dir}\n"
                "  * 발화 세션 시작 시 자동으로 녹음이 시작됩니다\n"
                "  * 침묵 4초 후 세션 종료 시 파일이 저장됩니다\n"
                "=" * 70
            )

            self.stream.start_stream()

        except Exception as e:
            self.get_logger().error(f"오디오 스트림 열기 실패: {e}")
            raise

    def _find_audio_device(self):
        """사용 가능한 오디오 입력 장치 찾기"""
        self.device_index = None
        self.device_name = "기본 마이크"

        # 기본 입력 장치 사용
        try:
            default_device = self.audio.get_default_input_device_info()
            self.device_index = default_device["index"]
            self.device_name = default_device["name"]
            self.get_logger().info(
                f"기본 입력 장치 사용: {self.device_name} (index: {self.device_index})"
            )
        except Exception as e:
            self.get_logger().warning(f"기본 입력 장치를 찾을 수 없음: {e}")

            # 사용 가능한 첫 번째 입력 장치 찾기
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    self.device_index = i
                    self.device_name = device_info["name"]
                    self.get_logger().info(
                        f"사용 가능한 입력 장치 찾음: {self.device_name} (index: {i})"
                    )
                    break

        if self.device_index is None:
            raise RuntimeError("사용 가능한 오디오 입력 장치를 찾을 수 없습니다")

    def vad_signal_callback(self, msg: Bool):
        """
        VAD 신호 콜백 - 세션 시작/종료 처리

        Args:
            msg (Bool): True=발화 중, False=발화 종료
        """
        if msg.data:  # True: 발화 시작/진행 중
            if not self.is_recording:
                # 새로운 세션 시작 - 녹음 파일 생성
                self.session_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.recording_path = (
                    self.recordings_dir
                    / f"session_{self.session_count:03d}_{timestamp}.wav"
                )

                self.wav_file = wave.open(str(self.recording_path), "wb")
                self.wav_file.setnchannels(self.CHANNELS)
                self.wav_file.setsampwidth(pyaudio.get_sample_size(self.FORMAT))
                self.wav_file.setframerate(self.SAMPLE_RATE)

                self.is_recording = True
                self.get_logger().info(f"🎙️  녹음 시작: {self.recording_path.name}")

        else:  # False: 발화 종료 (침묵 감지)
            if self.is_recording:
                # 세션 종료 - 녹음 파일 저장
                if self.wav_file:
                    self.wav_file.close()
                    self.wav_file = None
                    self.get_logger().info(f"✅ 녹음 완료: {self.recording_path.name}")

                self.is_recording = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio 콜백 함수

        Args:
            in_data: 오디오 데이터 (bytes)
            frame_count: 프레임 수
            time_info: 타임스탬프 정보
            status: 스트림 상태
        """
        if status:
            self.get_logger().warning(f"오디오 스트림 상태: {status}")

        # 녹음 중이면 WAV 파일에 오디오 데이터 저장
        if self.is_recording and self.wav_file:
            self.wav_file.writeframes(in_data)

        # AudioData 메시지 생성 및 발행 (기존 기능 유지)
        audio_msg = AudioData()
        audio_msg.data = list(in_data)
        self.audio_publisher.publish(audio_msg)

        return (in_data, pyaudio.paContinue)

    def destroy_node(self):
        """노드 종료 시 정리"""
        # 현재 녹음 중인 파일이 있으면 닫기
        if hasattr(self, "wav_file") and self.wav_file:
            self.wav_file.close()
            if self.recording_path:
                self.get_logger().info(f"녹음 완료 (종료): {self.recording_path.name}")

        # 오디오 스트림 정리
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, "audio"):
            self.audio.terminate()

        self.get_logger().info(f"총 {self.session_count}개 세션 녹음 완료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        audio_node = AudioInputTestNode()
        rclpy.spin(audio_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"오디오 입력 노드 오류: {e}")
    finally:
        if rclpy.ok():
            audio_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
