#!/usr/bin/env python3

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int8MultiArray, String


class VADSTTTestNode(Node):
    """
    VAD 및 STT 테스트 노드

    구독하는 토픽:
        - Publish2Agent_Vad (Int8MultiArray): VAD 프레임별 결과 (0/1)
        - Publish2Agent_STT (Bool): VAD 신호 (True: 말하는 중, False: 말 끝남)
        - Publish_STT_Result (String): STT 최종 결과 (텍스트)

    사용법:
        1. 오디오 입력 노드 실행 (마이크)
        2. VAD 노드 실행
        3. STT 노드 실행
        4. 이 테스트 노드 실행
        5. 마이크에 말하기

    실행 예시:
        ros2 run hearo_interaction_manager vad_stt_test_node
    """

    def __init__(self):
        super().__init__("vad_stt_test_node")

        # VAD 상태
        self.is_speaking = False
        self.speech_start_time = None
        self.frame_count = 0
        self.voice_frame_count = 0

        # VAD 프레임별 결과 구독 (0/1 배열)
        self.vad_frame_sub = self.create_subscription(
            Int8MultiArray, "Publish2Agent_Vad", self.vad_frame_callback, 10
        )

        # VAD 신호 구독 (True: 말하는 중, False: 말 끝남)
        self.vad_signal_sub = self.create_subscription(
            Bool, "Publish2Agent_STT", self.vad_signal_callback, 10
        )

        # STT 결과 구독 (STT 노드에서 직접 발행)
        self.stt_result_sub = self.create_subscription(
            String, "Publish_STT_Result", self.stt_result_callback, 10
        )

        self.get_logger().info(
            "\n"
            "=" * 70 + "\n"
            "  VAD & STT 테스트 노드 시작\n"
            "=" * 70 + "\n"
            "  구독 토픽 (기존 노드 출력을 직접 구독):\n"
            "    - Publish2Agent_Vad (Int8MultiArray): VAD 프레임별 결과\n"
            "    - Publish2Agent_STT (Bool): VAD 신호 (발화 시작/종료)\n"
            "    - Publish_STT_Result (String): STT 최종 결과\n"
            "\n"
            "  테스트 방법:\n"
            "    1. 마이크에 대고 말하세요\n"
            "    2. VAD 프레임별 결과 (0/1)가 실시간으로 표시됩니다\n"
            "    3. 발화 시작/종료 신호가 표시됩니다\n"
            "    4. 최종 STT 텍스트 결과가 표시됩니다\n"
            "\n"
            "  기존 vad.py와 stt_node.py를 그대로 사용합니다\n"
            "=" * 70
        )

    def vad_frame_callback(self, msg: Int8MultiArray):
        """
        VAD 프레임별 결과 수신 (실시간 모니터링)

        Args:
            msg (Int8MultiArray): 프레임별 음성 감지 결과 (0: 음성 없음, 1: 음성 있음)
        """
        vad_frames = list(msg.data)

        if not vad_frames:
            return

        # 프레임 통계 업데이트
        self.frame_count += len(vad_frames)
        self.voice_frame_count += sum(vad_frames)

        # 음성 감지된 프레임이 있을 때만 출력
        if sum(vad_frames) > 0:
            # 프레임 결과를 문자열로 변환 (가독성을 위해 10개씩 그룹화)
            frame_str = ""
            for i in range(0, len(vad_frames), 10):
                chunk = vad_frames[i : i + 10]
                frame_str += "".join(map(str, chunk)) + " "

            self.get_logger().info(
                f"[VAD 프레임] {frame_str.strip()} "
                f"(음성: {sum(vad_frames)}/{len(vad_frames)} 프레임)"
            )

    def vad_signal_callback(self, msg: Bool):
        """
        VAD 신호 수신 (발화 시작/종료 이벤트)

        Args:
            msg (Bool): True (말하는 중), False (말 끝남)
        """
        signal = msg.data

        if signal and not self.is_speaking:
            # 발화 시작
            self.is_speaking = True
            self.speech_start_time = time.time()
            self.frame_count = 0
            self.voice_frame_count = 0

            self.get_logger().info(
                "\n"
                "┏" + "━" * 68 + "┓\n"
                "┃  발화 시작 감지                                                    ┃\n"
                "┗" + "━" * 68 + "┛"
            )

        elif not signal and self.is_speaking:
            # 발화 종료
            self.is_speaking = False
            duration = (
                time.time() - self.speech_start_time if self.speech_start_time else 0
            )

            self.get_logger().info(
                "\n"
                "┏" + "━" * 68 + "┓\n"
                "┃  발화 종료 감지                                                    ┃\n"
                f"┃    - 발화 시간: {duration:.2f}초                                           ┃\n"
                f"┃    - 총 프레임: {self.frame_count}개                                        ┃\n"
                f"┃    - 음성 프레임: {self.voice_frame_count}개 "
                f"({self.voice_frame_count/max(self.frame_count,1)*100:.1f}%)                          ┃\n"
                "┃    - STT 결과 대기 중...                                           ┃\n"
                "┗" + "━" * 68 + "┛"
            )

    def stt_result_callback(self, msg: String):
        """
        STT 최종 결과 수신 (stt_node.py에서 직접 발행)

        Args:
            msg (String): STT 결과 JSON
                형식: {"ResponseUrl": "...", "resultSTT": "텍스트"}
        """
        try:
            response_data = json.loads(msg.data)

            # stt_node.py가 발행하는 형식
            text = response_data.get("resultSTT", "")

            if not text or not text.strip():
                self.get_logger().warning(
                    "\n"
                    "┏" + "━" * 68 + "┓\n"
                    "┃  STT 결과 없음 (빈 문자열 또는 인식 실패)                         ┃\n"
                    "┗" + "━" * 68 + "┛\n"
                )
                return

            # 텍스트 길이에 따라 표시
            text_display = text[:60] + "..." if len(text) > 60 else text

            self.get_logger().info(
                "\n"
                "┏" + "━" * 68 + "┓\n"
                "┃  STT 결과                                                          ┃\n"
                "┣" + "━" * 68 + "┫\n"
                f"┃  {text_display}                                                   ┃\n"
                "┗" + "━" * 68 + "┛\n"
            )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"[오류] STT JSON 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"[오류] STT 결과 처리 실패: {e}")


def main(args=None):
    rclpy.init(args=args)

    test_node = VADSTTTestNode()

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("\n[VAD-STT Test] 테스트 노드 종료")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
