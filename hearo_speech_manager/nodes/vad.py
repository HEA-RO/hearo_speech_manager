#!/usr/bin/env python3

import json
import time

import rclpy
import webrtcvad
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from six.moves import queue
from std_msgs.msg import Bool, Int8MultiArray, String


class VAD:
    """
    AudioData 메시지를 입력받아 VAD를 수행
    각 프레임별로 음성 감지 결과를 boolean 리스트로 반환

    Returns:
        - True: 해당 프레임에서 음성이 감지됨
        - False: 해당 프레임에서 음성이 감지되지 않음
    """

    VAD_MODE = 2  # 0-3, 높을수록 aggressive (더 엄격한 음성 감지)
    SAMPLE_RATE = 48000  # 샘플링 레이트 (48kHz)
    FRAME_DURATION = 30  # 프레임 길이 (ms)

    def __init__(self):
        # 프레임당 데이터 크기 계산 (샘플링레이트 * 프레임길이(ms) * 채널수(2) * 바이트(2))
        self.n_data = int(self.SAMPLE_RATE * (self.FRAME_DURATION / 1000.0) * 2)
        self.duration = (float(self.n_data) / self.SAMPLE_RATE) / 2.0

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.VAD_MODE)

        self.threshold = 0.1
        self.prev_data = 0

        print("INIT VAD (Boolean List Mode)!")

    def filter(self, cur_data: bool):
        """
        지수 이동 평균 필터링 (0.8 * 이전값 + 0.2 * 현재값)
        급격한 변화를 부드럽게 만들어 노이즈에 강건함

        Args:
            cur_data (bool): VAD 결과 (True/False)

        Returns:
            tuple: (필터링된 결과 (0 or 1), 필터링 값)
        """
        cur_data = 0.8 * self.prev_data + 0.2 * int(cur_data)
        self.prev_data = cur_data

        if cur_data > self.threshold:
            return 1, cur_data
        return 0, cur_data

    def detect(self, audio_msg):
        """
        AudioData 메시지에서 VAD를 수행하여 프레임별 음성 감지 결과 반환

        Args:
            audio_msg (AudioData): ROS2 AudioData 메시지

        Returns:
            list[bool]: 각 프레임별 음성 감지 결과 (True: 음성 있음, False: 음성 없음)
        """

        audio_data = bytes(audio_msg.data)
        vad_results = []
        offset = 0

        # 오디오 데이터를 프레임 단위로 분할하여 VAD 수행
        while offset + self.n_data <= len(audio_data):
            frame_data = audio_data[offset : offset + self.n_data]
            is_speech = self.vad.is_speech(frame_data, self.SAMPLE_RATE)

            # 지수 이동 평균 필터링 적용
            filtered_result, _ = self.filter(is_speech)
            vad_results.append(bool(filtered_result))

            offset += self.n_data

        return vad_results


class VADSignalEngine:
    """
    VAD 결과(boolean list)를 받아 발화 구간을 판정하고 partial/final 이벤트 생성
    """

    def __init__(
        self,
        partial_interval_ms: int = 500,
        vad_silence_ms: int = 4000,
    ):
        self.partial_interval = partial_interval_ms / 1000.0
        self.default_vad_silence = vad_silence_ms / 1000.0  # 기본값 저장
        self.vad_silence = self.default_vad_silence

        self.speaking = False
        self.session_started = False  # 세션 시작 여부
        self.session_start_ts = 0.0  # 세션 시작 시간
        self.last_voice_ts = 0.0
        self.last_partial_ts = 0.0

        # Static Waiting 모드 관련
        self.static_waiting_mode = False  # Static Waiting 모드 플래그
        self.first_voice_ts = 0.0  # 첫 음성 감지 시점

        print(
            f"VADSignalEngine initialized: partial={partial_interval_ms}ms, silence={vad_silence_ms}ms"
        )

    def set_vad_silence(self, vad_silence_ms: int, static_waiting: bool = False):
        """
        VAD 침묵 timeout 동적 변경

        Args:
            vad_silence_ms: 새로운 침묵 timeout (밀리초)
            static_waiting: Static Waiting 모드 여부 (첫 발화 시점 기준 timeout)
        """
        self.vad_silence = vad_silence_ms / 1000.0
        self.static_waiting_mode = static_waiting
        mode_str = " (Static Waiting mode)" if static_waiting else ""
        print(f"VADSignalEngine: vad_silence updated to {vad_silence_ms}ms{mode_str}")

    def reset_vad_silence(self):
        """VAD 침묵 timeout을 기본값으로 복원"""
        self.vad_silence = self.default_vad_silence
        self.static_waiting_mode = False
        print(
            f"VADSignalEngine: vad_silence reset to default {self.default_vad_silence * 1000}ms"
        )

    def reset(self):
        """발화 상태 초기화"""
        self.speaking = False
        self.session_started = False
        self.session_start_ts = 0.0
        self.last_voice_ts = 0.0
        self.last_partial_ts = 0.0
        self.first_voice_ts = 0.0

    def process(self, vad_results):
        """
        VAD 결과(boolean list)를 받아 발화 상태 반환

        Args:
            vad_results (list[bool]): VAD 결과 리스트

        Returns:
            bool or None: True (말하는 중), False (말 끝남), None (이벤트 없음)
        """
        now = time.time()
        signal = None

        # 세션이 시작되지 않았으면 지금 시작
        # if not self.session_started:
        #     self.session_started = True
        #     self.session_start_ts = now
        #     self.last_voice_ts = now
        #     self.last_partial_ts = now
        #     self.speaking = True
        #     # 세션 시작과 함께 즉시 True 신호 발행
        #     return True

        # VAD 결과에서 음성이 감지되었는지 확인
        vad_has_voice = any(vad_results) if vad_results else False

        if vad_has_voice:
            # 음성이 감지된 경우
            if not self.speaking:
                # 발화 시작
                self.speaking = True
                self.last_partial_ts = now
                # 첫 음성 감지 시점 저장 (Static Waiting 모드용)
                if self.first_voice_ts == 0.0:
                    self.first_voice_ts = now
            self.last_voice_ts = now
        else:
            # 음성이 감지되지 않은 경우
            if self.speaking:
                # 발화 중이었는데 침묵이 감지됨
                # 일반 모드: 마지막 음성부터의 침묵 시간 체크 (Static Waiting 모드는 아래에서 별도 처리)
                if not self.static_waiting_mode:
                    if (now - self.last_voice_ts) >= self.vad_silence:
                        # 침묵이 vad_silence 초 이상 지속되면 False (말 끝남)
                        signal = False
                        self.reset()
                        return signal

        # Static Waiting 모드에서도 첫 발화 시작 후 timeout 체크
        if self.static_waiting_mode and self.first_voice_ts > 0.0:
            if (now - self.first_voice_ts) >= self.vad_silence:
                # 첫 발화 시점부터 vad_silence 초 경과 시 무조건 종료
                signal = False
                self.reset()
                return signal

        # 발화 중일 때 주기적으로 True (말하는 중) 반환
        if self.speaking and (now - self.last_partial_ts) >= self.partial_interval:
            signal = True
            self.last_partial_ts = now

        return signal


class VADNode(Node):
    """
    ROS2 VAD 노드
    - Subscriber: Publish2Server_Chunk (AudioData)
    - Publisher:
        - Publish2Agent_Vad (Int8MultiArray): 프레임별 음성 감지 결과
        - Publish2Agent_STT (Bool): 발화 상태 (True: 말하는 중, False: 말 끝남)
    """

    def __init__(self):
        super().__init__("vad_node")

        self.vad = VAD()
        self.signal_engine = VADSignalEngine(
            partial_interval_ms=500, vad_silence_ms=4000
        )

        # VAD 활성화 상태 플래그
        self.vad_enabled = True  # 기본적으로 활성화 상태

        # Subscriber 생성
        self.audio_subscriber = self.create_subscription(
            AudioData, "Publish2Server_Chunk", self.audio_callback, 10
        )

        # VAD 리셋 Subscriber 추가 (로봇 발화 시작 전 리셋 신호 수신)
        self.reset_subscriber = self.create_subscription(
            Bool, "VAD_Reset", self.reset_callback, 10
        )

        # VAD Enable/Disable Subscriber 추가
        self.enable_subscriber = self.create_subscription(
            Bool, "VAD_Enable", self.enable_callback, 10
        )

        # 현재 Plan Subscriber 추가 (Dynamic timeout 조정)
        self.plan_subscriber = self.create_subscription(
            String, "Publish_Current_Plan", self.plan_callback, 10
        )

        # Publisher 생성
        self.vad_publisher = self.create_publisher(
            Int8MultiArray, "Publish2Agent_Vad", 10
        )

        # 발화 상태 Publisher 추가 (True: 말하는 중, False: 말 끝남)
        self.signal_publisher = self.create_publisher(Bool, "Publish2Agent_STT", 10)

        self.get_logger().info(
            "VAD Node initialized - listening on Publish2Server_Chunk"
        )
        self.get_logger().info(
            "Signal parameters: partial=500ms, silence=4000ms (default)"
        )

    def reset_callback(self, msg: Bool):
        """
        VAD 리셋 신호 수신 콜백
        로봇 발화 시작 전에 VAD 상태를 초기화하여 이전 세션의 영향을 제거

        Args:
            msg (Bool): 리셋 신호 (True이면 리셋)
        """
        if msg.data:
            self.signal_engine.reset()
            self.get_logger().info("VAD reset - session cleared before robot speech")

    def enable_callback(self, msg: Bool):
        """
        VAD 활성화/비활성화 제어 콜백
        로봇 발화 중에는 VAD를 비활성화하여 로봇 음성이나 준비 소음을 감지하지 않음

        Args:
            msg (Bool): True=VAD 활성화, False=VAD 비활성화
        """
        self.vad_enabled = msg.data
        status = "활성화" if self.vad_enabled else "비활성화"
        self.get_logger().info(f"VAD {status} (enabled={self.vad_enabled})")

    def plan_callback(self, msg: String):
        """
        현재 Plan 정보 수신 콜백
        Static Waiting plan인 경우 VAD timeout을 동적으로 조정

        Args:
            msg (String): Plan 정보 JSON
                {
                    "Topic": "Static Waiting" | "Wait for Response" | "None",
                    "Contents": {"Timeout": "30", ...}
                }
        """
        try:
            plan_data = json.loads(msg.data)
            topic = plan_data.get("Topic", "")
            contents = plan_data.get("Contents", {})

            if topic == "Static Waiting" or topic == "StaticWaiting":
                # Static Waiting: Timeout 시간으로 VAD silence 설정 (첫 발화 시점 기준)
                timeout_seconds = int(contents.get("Timeout", "30"))
                timeout_ms = timeout_seconds * 1000
                self.signal_engine.set_vad_silence(timeout_ms, static_waiting=True)
                self.get_logger().info(
                    f"VAD timeout updated for Static Waiting: {timeout_seconds}s (first voice trigger)"
                )
            elif topic == "Wait for Response" or topic == "WaitForResponse":
                # 기본 Wait for Response: 기본값(4초) 유지 (마지막 음성 기준)
                self.signal_engine.reset_vad_silence()
                self.get_logger().info(
                    "VAD timeout reset to default for Wait for Response: 4s"
                )
            elif topic == "None":
                # Plan 종료: 기본값으로 복원
                self.signal_engine.reset_vad_silence()
                self.get_logger().info("VAD timeout reset to default (plan ended)")

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Plan JSON decode error: {e}")
        except Exception as e:
            self.get_logger().error(f"Plan callback error: {e}")

    def audio_callback(self, msg):
        """
        AudioData 메시지를 받아 VAD를 수행하고 결과를 publish

        Args:
            msg (AudioData): 입력 오디오 데이터
        """
        try:
            # VAD가 비활성화되어 있으면 아무것도 하지 않음
            if not self.vad_enabled:
                return

            # VAD 수행
            vad_results = self.vad.detect(msg)

            # Int8MultiArray 메시지 생성
            vad_msg = Int8MultiArray()
            vad_msg.data = [int(result) for result in vad_results]  # boolean to int

            self.vad_publisher.publish(vad_msg)

            # VADSignalEngine에 결과 전달하여 발화 상태 확인
            signal = self.signal_engine.process(vad_results)

            # 이벤트가 있는 경우에만 발행
            if signal is not None:
                signal_msg = Bool()
                signal_msg.data = signal
                self.signal_publisher.publish(signal_msg)

                if signal:
                    self.get_logger().info("VAD Signal: True (발화 진행중)")
                else:
                    self.get_logger().info("VAD Signal: False (발화 종료)")

            if vad_results:
                # voice_frames = sum(vad_results)
                # total_frames = len(vad_results)
                # self.get_logger().info(
                #     f"VAD: {voice_frames}/{total_frames} frames have voice"
                # )
                colored_results = [
                    f"\033[91m{result}\033[0m" if result else str(result)
                    for result in vad_results
                ]
                self.get_logger().info(f"VAD: {', '.join(colored_results)}")

        except Exception as e:
            self.get_logger().error(f"VAD processing error: {e}")


def main(args=None):
    rclpy.init(args=args)

    vad_node = VADNode()

    try:
        rclpy.spin(vad_node)
    except KeyboardInterrupt:
        pass
    finally:
        vad_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
