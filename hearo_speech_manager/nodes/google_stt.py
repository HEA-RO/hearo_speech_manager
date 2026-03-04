#!/usr/bin/env python3
"""
Google STT Node - Google Cloud Speech API 기반 실시간 음성 인식 노드

Legacy ROS1 코드를 ROS2로 마이그레이션하고, VAD 기반 타이밍 로직을 추가하여
Whisper STT의 동작을 제어합니다.

Subscriber:
    - Publish2Server_Chunk (audio_common_msgs/AudioData): 오디오 청크 수신
    - Publish2Agent_Vad (std_msgs/Int8MultiArray): VAD 프레임별 결과 수신

Publisher:
    - Google_STT_Control (std_msgs/Bool): Whisper 제어 신호
        - True: 발화 감지됨, Whisper transcribe 진행
        - False: 발화 없음, Whisper buffer clear 및 skip
    - Google_STT_Transcript (std_msgs/String): Google STT 실시간 결과 (디버깅용)
"""

import json
import threading
import time
from typing import Optional

import rclpy
from audio_common_msgs.msg import AudioData
from google.cloud import speech
from rclpy.node import Node
from six.moves import queue
from std_msgs.msg import Bool, Int8MultiArray, String


class AudioStreamManager:
    """
    ROS2 오디오 데이터 수신 및 형변환 클래스
    오디오 청크를 Queue에 버퍼링하고 Generator 패턴으로 Google Speech API에 스트리밍

    게이트 기능:
        - 게이트가 열려있을 때만 오디오가 버퍼에 추가됨
        - 게이트가 닫혀있으면 오디오는 무시됨 (Google STT에 전달되지 않음)
        - 스트림 자체는 항상 열린 상태로 유지 (Google STT 연결 유지)
    """

    def __init__(self, logger):
        self.logger = logger
        self._buff = queue.Queue()
        self.open = False
        self._gate_open = False  # 게이트: True일 때만 오디오가 버퍼에 추가됨

        self.logger.info("AudioStreamManager initialized")

    def open_gate(self):
        """게이트 열기 - 오디오가 버퍼에 추가되기 시작"""
        self._gate_open = True
        self.logger.info("AudioStreamManager gate OPENED")

    def close_gate(self):
        """게이트 닫기 - 오디오가 버퍼에 추가되지 않음"""
        self._gate_open = False
        self.logger.info("AudioStreamManager gate CLOSED")

    def add_audio_chunk(self, msg: AudioData):
        """
        오디오 청크를 버퍼에 추가

        Google STT 연결 유지를 위해 항상 버퍼에 추가합니다.
        게이트 상태는 STT 결과 처리 여부를 결정하는 데 사용됩니다.

        Args:
            msg (AudioData): ROS2 AudioData 메시지
        """
        self._buff.put(msg)

    def close(self):
        """스트림 종료 (Google STT 연결 종료 시에만 호출)"""
        self.open = False
        self._gate_open = False
        self._buff.put(None)
        self.logger.info("AudioStreamManager closed")

    def clear_buffer(self):
        """버퍼 초기화"""
        with self._buff.mutex:
            self._buff.queue.clear()
        self.logger.info("AudioStreamManager buffer cleared")

    def generator(self):
        """
        오디오 청크를 Generator로 반환 (Google Speech API 스트리밍용)

        Yields:
            bytes: 오디오 데이터 (bytes)
        """
        while self.open:
            try:
                # 타임아웃 설정으로 블로킹 방지 (0.5초)
                chunk = self._buff.get(timeout=0.5)
            except queue.Empty:
                # 타임아웃 발생 시 계속 루프 (self.open 체크)
                continue

            if chunk is None:
                return

            # AudioData를 bytes로 변환
            data = [bytes(chunk.data)]

            # 버퍼에 추가 데이터가 있으면 함께 처리
            while self.open:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(bytes(chunk.data))
                except queue.Empty:
                    break

            yield b"".join(data)


class TurnTakingController:
    """
    Turn-Taking 로직 컨트롤러
    Legacy 코드의 turn_taking() 메서드를 클래스화하여 타이밍 기반 발화 감지 구현

    타이밍 설정:
        - FIRST_STT_WAIT_TIME: 4.0초 (초기 STT 대기 시간)
        - VAD_CHECK_DURATION: 4.0초 (first-stt 이후 VAD 체크 기간)
        - SECOND_STT_WAIT_TIME: 0.5초 (VAD 감지 후 추가 STT 대기 시간)

    플로우:
        1. first-stt: FIRST_STT_WAIT_TIME 동안 발화 없으면 -> VAD 체크 진입
        2. vad-check: VAD_CHECK_DURATION 동안 VAD true 감지 여부 확인
           - VAD true 없으면 -> 종료
           - VAD true 있으면 -> second-stt 진입
        3. second-stt: SECOND_STT_WAIT_TIME 동안 발화 없으면 -> 종료 (무조건 True)
    """

    FIRST_STT_WAIT_TIME = 4.0  # first-stt 대기 시간 (초)
    VAD_CHECK_DURATION = 2.0  # VAD 체크 기간 (first 이후 추가 시간, 초)
    SECOND_STT_WAIT_TIME = 3.0  # second-stt 대기 시간 (초)

    def __init__(self, logger):
        self.logger = logger
        self.stt_cnt = 0

        # 타이머 변수
        self._init_stt_time = 0.0
        self._last_stt_time = 0.0

        # 상태 변수
        self._second_stt_check = False  # Second STT 단계 진입 여부
        self._stt_checked = False  # STT 응답을 한 번이라도 받았는지
        self._vad_state = False  # 현재 VAD 상태 (음성 감지 여부)

        # 결과 변수
        self._speech_detected = None  # None: 진행중, True: 발화 있음, False: 발화 없음

        self.logger.info(
            f"TurnTakingController initialized - "
            f"First: {self.FIRST_STT_WAIT_TIME}s, "
            f"VAD: {self.VAD_CHECK_DURATION}s, "
            f"Second: {self.SECOND_STT_WAIT_TIME}s"
        )

    def start_session(self):
        """세션 시작, 타이머 초기화"""
        current_time = time.perf_counter()
        self._init_stt_time = current_time
        self._last_stt_time = current_time

        self._second_stt_check = False
        self._stt_checked = False
        self._vad_state = False
        self._speech_detected = None

        self.logger.info("Turn-taking session started")

    def process_stt_result(self, transcript: str, is_final: bool):
        """
        STT 결과 처리, 타이머 업데이트

        Args:
            transcript (str): STT 결과 텍스트
            is_final (bool): 최종 결과 여부
        """
        if transcript:
            self._last_stt_time = time.perf_counter()
            self._stt_checked = True
            # self.logger.info(f"STT result: '{transcript}' (final={is_final})")

    def process_vad_signal(self, vad_results: list):
        """
        VAD 신호 처리

        Args:
            vad_results (list): VAD 프레임별 결과 (Int8MultiArray.data)
        """
        # VAD 결과에서 음성이 감지되었는지 확인
        self._vad_state = any(vad_results) if vad_results else False

    def check_timeout(self) -> Optional[bool]:
        """
        타임아웃 체크 및 발화 감지 결과 반환

        Returns:
            Optional[bool]:
                - None: 계속 진행 (타임아웃 발생 안 함)
                - True: 발화 있음 (Whisper transcribe 진행)
                - False: 발화 없음 (Whisper skip)
        """
        self.stt_cnt += 1

        if self._speech_detected is not None:
            # 이미 결과가 결정됨
            return self._speech_detected

        current_time = time.perf_counter()
        silent_dur = current_time - self._last_stt_time

        # 디버그 로그 (매 10회마다 = 약 1초)
        if self.stt_cnt % 10 == 0:
            # True 값은 빨간색으로 표시
            stt_checked_str = (
                f"\033[91m{self._stt_checked}\033[0m"
                if self._stt_checked
                else str(self._stt_checked)
            )
            vad_state_str = (
                f"\033[91m{self._vad_state}\033[0m"
                if self._vad_state
                else str(self._vad_state)
            )
            second_check_str = (
                f"\033[91m{self._second_stt_check}\033[0m"
                if self._second_stt_check
                else str(self._second_stt_check)
            )
            self.logger.info(
                f"[Timeout Check] silent_dur: {silent_dur:.1f}s, "
                f"stt_checked: {stt_checked_str}, vad_state: {vad_state_str}, "
                f"second_check: {second_check_str}"
            )

        # Second STT 단계
        if self._second_stt_check:
            if silent_dur >= self.SECOND_STT_WAIT_TIME:
                # Second STT 타임아웃 - first 또는 second에서 발화가 있었는지 확인
                if self._stt_checked:
                    # First 또는 Second에서 발화 감지됨
                    self.logger.info(
                        f"second-stt 타임아웃! 발화 감지됨 -> Whisper로 전달 (silent: {silent_dur:.1f}s)"
                    )
                    self._speech_detected = True
                else:
                    # First와 Second 모두 발화 없음
                    self.logger.info(
                        f"second-stt 타임아웃! 발화 없음 -> Whisper skip (silent: {silent_dur:.1f}s)"
                    )
                    self._speech_detected = False

        # First STT 단계
        else:
            if silent_dur >= self.FIRST_STT_WAIT_TIME:
                # First STT 타임아웃 - VAD 체크 시작
                # VAD 체크 종료 시점 = FIRST_STT_WAIT_TIME + VAD_CHECK_DURATION
                vad_check_end = self.FIRST_STT_WAIT_TIME + self.VAD_CHECK_DURATION

                if silent_dur < vad_check_end:
                    # VAD 체크 구간 (first-stt 이후 ~ vad_check_end 까지)
                    if self._vad_state:
                        self.logger.info(
                            "\033[93m"
                            + "VAD signal detected! -> second-stt 진입"
                            + "\033[0m"
                        )
                        self._last_stt_time = time.perf_counter()
                        self._second_stt_check = True
                else:
                    # VAD 체크 타임아웃 (VAD true 없이 vad_check_end 경과)
                    if self._stt_checked:
                        # first-stt 동안 발화가 있었음
                        self.logger.info(
                            f"사용자 발화 대기 종료! 성공! (VAD timeout: {silent_dur:.1f}s)"
                        )
                        self._speech_detected = True
                    else:
                        # first-stt 동안 발화가 없었음
                        self.logger.info(
                            f"사용자 최초 발화 없음!! (VAD timeout: {silent_dur:.1f}s)"
                        )
                        self._speech_detected = False

        return self._speech_detected


class GoogleSTTClient:
    """
    Google Cloud Speech API 클라이언트
    실시간 스트리밍 인식 수행 및 Partial/Final 결과 처리
    """

    SAMPLE_RATE = 48000
    LANGUAGE = "ko-KR"

    def __init__(self, logger, audio_stream: AudioStreamManager):
        self.logger = logger
        self.audio_stream = audio_stream

        # 스트리밍 준비 상태 플래그
        self.streaming_ready = False

        # Google Speech Client 초기화
        try:
            self.client = speech.SpeechClient()

            # Recognition Config 설정
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLE_RATE,
                language_code=self.LANGUAGE,
            )

            self.streaming_config = speech.StreamingRecognitionConfig(
                config=config, interim_results=True
            )

            self.logger.info(
                f"Google STT Client initialized - "
                f"Rate: {self.SAMPLE_RATE}Hz, Lang: {self.LANGUAGE}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Google STT Client: {e}")
            raise

    def start_streaming(self, stt_callback):
        """
        스트리밍 인식 시작 (별도 스레드에서 실행)

        Args:
            stt_callback: STT 결과를 받을 콜백 함수 (transcript, is_final)
        """

        def _stream_loop():
            while True:
                try:
                    # 오디오 스트림이 열릴 때까지 대기
                    while not self.audio_stream.open:
                        time.sleep(0.1)

                    self.logger.info(
                        "\033[32m" + "Google STT streaming connecting..." + "\033[0m"
                    )

                    # Generator 생성
                    audio_generator = self.audio_stream.generator()
                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator
                    )

                    # 스트리밍 인식 수행
                    responses = self.client.streaming_recognize(
                        self.streaming_config, requests
                    )

                    # 스트리밍 연결 완료 - gRPC 연결되면 ready 설정
                    # 참고: 실제 응답은 음성 데이터가 있을 때만 옴
                    self.streaming_ready = True
                    self.logger.info(
                        "\033[32m" + "Google STT streaming READY!" + "\033[0m"
                    )

                    # 결과 수신 및 콜백 호출
                    self._listen_loop(responses, stt_callback)

                except Exception as e:
                    self.logger.error(f"Google STT streaming error: {e}")
                    self.streaming_ready = False
                    time.sleep(1)  # 에러 후 재시도 대기

        # 별도 스레드에서 실행
        thread = threading.Thread(target=_stream_loop, daemon=True)
        thread.start()
        self.logger.info("Google STT streaming thread started")

    def _listen_loop(self, responses, stt_callback):
        """
        응답 수신 루프 (Legacy 코드의 listen_loop 메서드)

        Args:
            responses: Google Speech API 응답 이터레이터
            stt_callback: STT 결과를 받을 콜백 함수
        """
        try:
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                is_final = result.is_final

                # 콜백 호출
                stt_callback(transcript, is_final)

        except Exception as e:
            # Audio Timeout Error는 로봇 발화 중 정상적으로 발생하므로 무시
            error_msg = str(e)
            if "Audio Timeout Error" in error_msg or "audio" in error_msg.lower():
                self.logger.debug(
                    f"Google STT audio timeout (expected during robot speech)"
                )
            else:
                self.logger.error(f"Google STT listen loop error: {e}")


class GoogleSTTNode(Node):
    """
    Google STT ROS2 노드
    실시간 음성 인식 및 Whisper STT 제어 신호 발행
    """

    # Wait for Response 모드에서 VAD true 없이 timeout 되는 시간 (초)
    # 발화뿐 아니라 소음도 없는 경우를 의미
    NO_SOUND_TIMEOUT = 6.0

    def __init__(self):
        super().__init__("google_stt_node")

        # 컴포넌트 초기화
        self.audio_stream = AudioStreamManager(self.get_logger())
        self.turn_taking = TurnTakingController(self.get_logger())
        self.stt_client = GoogleSTTClient(self.get_logger(), self.audio_stream)

        # 세션 상태
        self._session_active = False
        self._control_published = False  # 제어 신호 발행 여부

        # 쿨다운 및 VAD 기반 세션 시작 제어
        self._session_cooldown_end = 0.0  # 쿨다운 종료 시간
        self._vad_voice_detected = False  # VAD 음성 감지 플래그
        self.SESSION_COOLDOWN = 1.0  # 쿨다운 시간 (초)

        # Static Waiting 모드 (Turn-taking 비활성화, Timeout 후 종료)
        self._static_waiting_mode = False
        self._static_waiting_timeout = 0.0  # Timeout 종료 시간
        self._static_waiting_start_time = 0.0  # Static Waiting 시작 시간

        # Wait for Response 모드 - No Sound Timeout
        # VAD true가 한 번도 없이 NO_SOUND_TIMEOUT초 경과 시 빈 문자열 발행
        self._wait_for_response_mode = False
        self._wait_for_response_start_time = 0.0  # 세션 시작 시간
        self._any_vad_true_detected = False  # 세션 중 VAD true를 한 번이라도 받았는지

        # Subscriber 생성
        self.audio_subscriber = self.create_subscription(
            AudioData, "Publish2Server_Chunk", self._audio_callback, 10
        )

        self.vad_subscriber = self.create_subscription(
            Int8MultiArray, "Publish2Agent_Vad", self._vad_callback, 10
        )

        # Publish_Current_Plan 구독 (Static Waiting timeout 동기화)
        self.plan_subscriber = self.create_subscription(
            String, "Publish_Current_Plan", self._plan_callback, 10
        )

        # Publisher 생성
        self.control_publisher = self.create_publisher(Bool, "Google_STT_Control", 10)

        self.transcript_publisher = self.create_publisher(
            String, "Google_STT_Transcript", 10
        )

        # 타이머 생성 (주기적으로 타임아웃 체크)
        self.timer = self.create_timer(0.1, self._check_timeout_callback)

        # 오디오 스트림 상시 열기 (Google STT 연결 유지)
        self.audio_stream.open = True

        # STT 스트리밍 시작
        self.stt_client.start_streaming(self._stt_result_callback)

        self.get_logger().info("Google STT Node initialized")
        self.get_logger().info("Listening on: Publish2Server_Chunk, Publish2Agent_Vad")
        self.get_logger().info(
            "Publishing to: Google_STT_Control, Google_STT_Transcript"
        )

        # 초기화 시에는 세션을 시작하지 않음 (VAD 음성 감지 시 시작)
        self.get_logger().info("Waiting for VAD voice detection...")

        # Google STT 스트림 warm-up (첫 세션 cold start 방지)
        self._warmup_stream()

    def _warmup_stream(self):
        """
        Google STT 스트림 warm-up

        무음 데이터를 전송하여 gRPC 미리 연결
        노드 초기화 시 호출되어 스트리밍 스레드가 연결

        주의:
        - 무음 데이터에 대해서는 Google STT가 transcript를 반환하지 않음
        - 따라서 streaming_ready는 streaming_recognize() 호출 시점에 설정됨
        - 이 메서드는 단순히 gRPC 연결 수립을 트리거하는 역할만 함
        - blocking 없이 즉시 반환됨
        """
        # 48kHz, 16-bit PCM 무음 데이터 (약 0.5초)
        # gRPC 연결 수립을 트리거하기 위한 최소 데이터
        silence_duration = 0.5  # 초
        sample_rate = 48000
        num_samples = int(sample_rate * silence_duration)
        silence_data = bytes(num_samples * 2)  # 16-bit = 2 bytes per sample

        # AudioData 형태로 버퍼에 추가
        warmup_msg = AudioData()
        warmup_msg.data = list(silence_data)

        self.audio_stream.add_audio_chunk(warmup_msg)
        self.get_logger().info(
            "\033[33m"
            + f"Warm-up: {silence_duration}s silence sent to trigger gRPC connection"
            + "\033[0m"
        )

    def _start_session(self):
        """
        새로운 세션 시작

        Google STT 연결은 상시 유지
        세션 활성화로 STT 결과 처리
        """
        # 세션 활성화
        self._session_active = True
        self._control_published = False
        self.audio_stream.open_gate()  # 로깅용
        self.turn_taking.start_session()

        # Google STT ready 상태 로깅
        if self.stt_client.streaming_ready:
            self.get_logger().info("=== Session started (STT ready) ===")
        else:
            self.get_logger().warn("=== Session started (STT connecting...) ===")

    def _end_session(self, speech_detected: bool):
        """
        세션 종료 및 제어 신호 발행

        Google STT 연결은 유지하고, 게이트만 닫아서 오디오 수집을 중단

        Args:
            speech_detected (bool): 발화 감지 여부
        """
        if self._control_published:
            return

        self._session_active = False
        self._control_published = True

        # 쿨다운 설정 및 VAD 플래그 리셋
        self._session_cooldown_end = time.perf_counter() + self.SESSION_COOLDOWN
        self._vad_voice_detected = False

        # Static Waiting 모드 리셋
        self._static_waiting_mode = False

        # Wait for Response 모드 리셋
        self._wait_for_response_mode = False
        self._any_vad_true_detected = False

        # 게이트 닫기 및 버퍼 초기화 (스트림은 유지)
        self.audio_stream.close_gate()
        self.audio_stream.clear_buffer()

        # 주의: streaming_ready는 리셋하지 않음 (연결 유지)

        # Whisper 제어 신호 발행
        control_msg = Bool()
        control_msg.data = speech_detected
        self.control_publisher.publish(control_msg)

        if speech_detected:
            self.get_logger().info(
                "\033[92m"
                + f"=== Session ended: Speech DETECTED -> Whisper will transcribe ==="
                + "\033[0m"
            )
        else:
            self.get_logger().info(
                "\033[91m"
                + f"=== Session ended: NO speech -> Whisper will skip ==="
                + "\033[0m"
            )

        # 다음 VAD 음성 감지 시 새 세션 시작
        self.get_logger().info("Waiting for VAD voice detection...")

    def _audio_callback(self, msg: AudioData):
        """
        오디오 청크 수신 콜백

        게이트가 열려있으면 버퍼에 추가, 닫혀있으면 무시
        세션 시작/종료와 관계없이 항상 호출

        Args:
            msg (AudioData): 입력 오디오 데이터
        """
        # 항상 add_audio_chunk 호출 (게이트가 필터링)
        self.audio_stream.add_audio_chunk(msg)

    def _vad_callback(self, msg: Int8MultiArray):
        """
        VAD 신호 수신 콜백

        세션 비활성 상태에서 음성 감지 시 즉시 세션 시작

        Args:
            msg (Int8MultiArray): VAD 프레임별 결과
        """
        vad_results = list(msg.data)
        has_voice = any(vad_results)

        if self._session_active:
            # 세션 활성 상태: Turn-taking 처리
            self.turn_taking.process_vad_signal(vad_results)

            # Wait for Response 모드: VAD true 감지 시 플래그 업데이트
            if self._wait_for_response_mode and has_voice:
                if not self._any_vad_true_detected:
                    self.get_logger().info(
                        "\033[92m[Wait for Response] First VAD true detected!\033[0m"
                    )
                self._any_vad_true_detected = True
        else:
            # 세션 비활성 상태: 음성 감지 + 쿨다운 종료 시 세션 시작
            if has_voice:
                now = time.perf_counter()
                if now >= self._session_cooldown_end:
                    self._start_session()

    def _stt_result_callback(self, transcript: str, is_final: bool):
        """
        Google STT 결과 수신 콜백

        Args:
            transcript (str): STT 결과 텍스트
            is_final (bool): 최종 결과 여부
        """
        if self._session_active:
            # Turn-taking 컨트롤러에 전달
            self.turn_taking.process_stt_result(transcript, is_final)

            # 디버깅용 발행
            if transcript:
                transcript_msg = String()
                transcript_msg.data = f"{transcript} (final={is_final})"
                self.transcript_publisher.publish(transcript_msg)

    def _check_timeout_callback(self):
        """타이머 콜백: 주기적으로 타임아웃 체크"""
        if not self._session_active:
            return

        # Static Waiting 모드: _static_waiting_timeout까지 대기 후 발화 여부에 따라 True/False 발행
        # 주의: TurnTakingController의 자체 타임아웃(FIRST_STT_WAIT_TIME 등)은 적용하지 않음
        if self._static_waiting_mode:
            now = time.perf_counter()
            elapsed = now - self._static_waiting_start_time

            # 10회마다 로그 출력 (약 1초마다)
            if int(elapsed * 10) % 10 == 0 and int(elapsed) > 0:
                remaining = self._static_waiting_timeout - now
                if remaining > 0:
                    stt_checked_str = (
                        f"\033[91m{self.turn_taking._stt_checked}\033[0m"
                        if self.turn_taking._stt_checked
                        else str(self.turn_taking._stt_checked)
                    )
                    self.get_logger().info(
                        f"[Static Waiting] {elapsed:.1f}s elapsed, {remaining:.1f}s remaining "
                        f"(stt_checked: {stt_checked_str})"
                    )

            if now >= self._static_waiting_timeout:
                # Timeout 도달 - 발화 여부에 따라 True/False 발행
                speech_detected = self.turn_taking._stt_checked
                if speech_detected:
                    self.get_logger().info(
                        f"\033[92m[Static Waiting] Timeout reached ({elapsed:.1f}s), "
                        f"speech DETECTED -> Whisper transcribe\033[0m"
                    )
                else:
                    self.get_logger().info(
                        f"\033[91m[Static Waiting] Timeout reached ({elapsed:.1f}s), "
                        f"NO speech -> Sending empty string\033[0m"
                    )
                self._end_session(speech_detected)
            return

        # Wait for Response 모드: VAD true 없이 NO_SOUND_TIMEOUT초 경과 시 빈 문자열 발행
        if self._wait_for_response_mode and not self._any_vad_true_detected:
            now = time.perf_counter()
            elapsed = now - self._wait_for_response_start_time

            # 1초마다 로그 출력
            if int(elapsed) > 0 and int(elapsed * 10) % 10 == 0:
                remaining = self.NO_SOUND_TIMEOUT - elapsed
                if remaining > 0:
                    self.get_logger().info(
                        f"[Wait for Response] No VAD true yet... "
                        f"{elapsed:.1f}s elapsed, {remaining:.1f}s until timeout"
                    )

            if elapsed >= self.NO_SOUND_TIMEOUT:
                # VAD true 없이 NO_SOUND_TIMEOUT초 경과 - 빈 문자열 발행
                self.get_logger().info(
                    "\033[91m[Wait for Response] No sound timeout! "
                    f"({elapsed:.1f}s without VAD true) -> Sending empty string\033[0m"
                )
                self._end_session(False)
                return

        # 일반 모드: Turn-taking 로직 사용
        speech_detected = self.turn_taking.check_timeout()
        if speech_detected is not None:
            # 세션 종료 및 제어 신호 발행
            self._end_session(speech_detected)

    def _plan_callback(self, msg: String):
        """
        현재 Plan 정보 수신 콜백 (Publish_Current_Plan)

        Static Waiting일 경우 Turn-taking 로직을 비활성화하고
        Timeout 후에만 종료 시그널을 발행합니다.

        Wait for Response일 경우 VAD true 없이 NO_SPEECH_TIMEOUT초 경과 시
        빈 문자열을 발행하여 timeout 처리합니다.

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
                # Static Waiting 모드 활성화
                timeout_seconds = int(contents.get("Timeout", "30"))
                self._static_waiting_mode = True
                self._static_waiting_start_time = time.perf_counter()
                # self._static_waiting_timeout = (
                #     self._static_waiting_start_time + timeout_seconds + 18
                # )  # 로봇 발화 18초
                self._static_waiting_timeout = (
                    self._static_waiting_start_time + timeout_seconds
                )

                # Wait for Response 모드 비활성화
                self._wait_for_response_mode = False

                self.get_logger().info(
                    f"[Static Waiting] Mode ACTIVATED: {timeout_seconds}s timeout"
                )

                # 세션이 비활성이면 시작
                if not self._session_active:
                    self._start_session()

            elif topic == "Wait for Response" or topic == "WaitForResponse":
                # Wait for Response 모드 활성화
                # VAD true가 없이 NO_SPEECH_TIMEOUT초 경과 시 빈 문자열 발행
                self._wait_for_response_mode = True
                self._wait_for_response_start_time = time.perf_counter()
                self._any_vad_true_detected = False

                # Static Waiting 모드 비활성화
                self._static_waiting_mode = False

                self.get_logger().info(
                    f"[Wait for Response] Mode ACTIVATED: "
                    f"{self.NO_SOUND_TIMEOUT}s no-sound timeout"
                )

                # 세션이 비활성이면 시작 (VAD 없이도 세션 시작)
                if not self._session_active:
                    self._start_session()

            elif topic == "None":
                # Plan 종료 - 모든 모드 비활성화
                if self._static_waiting_mode:
                    self.get_logger().info("[Static Waiting] Mode DEACTIVATED")
                if self._wait_for_response_mode:
                    self.get_logger().info("[Wait for Response] Mode DEACTIVATED")
                self._static_waiting_mode = False
                self._wait_for_response_mode = False

            else:
                # 기타 모드 - 모든 특수 모드 비활성화
                if self._static_waiting_mode:
                    self.get_logger().info(
                        f"[Static Waiting] Mode DEACTIVATED (switching to {topic})"
                    )
                if self._wait_for_response_mode:
                    self.get_logger().info(
                        f"[Wait for Response] Mode DEACTIVATED (switching to {topic})"
                    )
                self._static_waiting_mode = False
                self._wait_for_response_mode = False

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Plan JSON decode error: {e}")
        except Exception as e:
            self.get_logger().error(f"Plan callback error: {e}")


def main(args=None):
    rclpy.init(args=args)

    google_stt_node = GoogleSTTNode()

    try:
        rclpy.spin(google_stt_node)
    except KeyboardInterrupt:
        pass
    finally:
        google_stt_node.audio_stream.close()
        google_stt_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
