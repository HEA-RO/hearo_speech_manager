#!/usr/bin/env python3
"""
HeaRo Speech Manager Utility Functions
HeaRo 음성 관리 유틸리티 함수

hearo_knowledge_manager/config/utility.py에서 TTS 노드가 필요로 하는 항목만
독립 복사한 모듈입니다. 외부 패키지 의존 없이 자체 완결적으로 동작합니다.
"""

import os
import gc
import inspect
from typing import Optional, Any, Dict
from enum import Enum


# ============================================================================
# ANSI Color Utilities
# ============================================================================


class AnsiColor(Enum):
    """터미널 출력용 ANSI 색상 코드."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"


def print_color(
    message: str,
    color: str = "RESET",
    mode: str = "info",
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    background: Optional[str] = None,
    logger: Any = None,
) -> None:
    """ANSI 컬러 메시지 출력. logger가 있으면 ROS2 로거 사용, 없으면 print."""
    ansi_codes = []
    if bold:
        ansi_codes.append(AnsiColor.BOLD.value)
    if italic:
        ansi_codes.append(AnsiColor.ITALIC.value)
    if underline:
        ansi_codes.append(AnsiColor.UNDERLINE.value)

    try:
        ansi_codes.append(AnsiColor[color.upper()].value)
    except KeyError:
        pass

    if background:
        try:
            ansi_codes.append(AnsiColor[background.upper()].value)
        except KeyError:
            pass

    prefix = "".join(ansi_codes)
    suffix = AnsiColor.RESET.value
    styled_message = f"{prefix}{message}{suffix}"

    if logger:
        log_functions = {
            "info": logger.info,
            "warn": logger.warn,
            "warning": logger.warn,
            "error": logger.error,
            "debug": logger.debug,
        }
        log_func = log_functions.get(mode.lower(), logger.info)
        log_func(styled_message)
    else:
        print(styled_message, flush=True)


def format_colored(
    message: str,
    color: str = "RESET",
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    background: Optional[str] = None,
) -> str:
    """ANSI 컬러 포맷 문자열 반환 (출력 없음)."""
    ansi_codes = []
    if bold:
        ansi_codes.append(AnsiColor.BOLD.value)
    if italic:
        ansi_codes.append(AnsiColor.ITALIC.value)
    if underline:
        ansi_codes.append(AnsiColor.UNDERLINE.value)

    try:
        ansi_codes.append(AnsiColor[color.upper()].value)
    except KeyError:
        pass

    if background:
        try:
            ansi_codes.append(AnsiColor[background.upper()].value)
        except KeyError:
            pass

    prefix = "".join(ansi_codes)
    suffix = AnsiColor.RESET.value
    return f"{prefix}{message}{suffix}"


# ============================================================================
# Error Handling Utilities (TTS 전용 에러 코드 인라인)
# ============================================================================

_TTS_ERROR_CODES: Dict[str, Dict[str, str]] = {
    "TTS001": {
        "message": "TTS initialization failed",
        "category": "TTS",
        "description": "TTS 핸들러 초기화 실패",
        "cause": "모델 로드 실패 또는 GPU 메모리 부족",
        "solution": "GPU 메모리 확인 및 모델 경로 점검",
    },
    "TTS002": {
        "message": "TTS generation failed",
        "category": "TTS",
        "description": "TTS 음성 생성 실패",
        "cause": "백엔드 오류 또는 입력 텍스트 문제",
        "solution": "에러 로그 확인 및 텍스트 유효성 점검",
    },
    "TTS003": {
        "message": "TTS backend not available",
        "category": "TTS",
        "description": "요청된 TTS 백엔드를 사용할 수 없음",
        "cause": "백엔드 미설치 또는 비활성화",
        "solution": "백엔드 설치 상태 및 설정 확인",
    },
    "TTS004": {
        "message": "Audio cache operation failed",
        "category": "TTS",
        "description": "오디오 캐시 DB 작업 실패",
        "cause": "DB 연결 실패 또는 쿼리 오류",
        "solution": "MySQL 서버 상태 및 tb_audioCache 스키마 확인",
    },
    "TTS005": {
        "message": "GPU session management error",
        "category": "TTS",
        "description": "GPU 세션 관리 오류",
        "cause": "CUDA 장치 오류 또는 메모리 부족",
        "solution": "GPU 상태 확인 및 불필요한 프로세스 종료",
    },
    "AUD001": {
        "message": "Invalid AudioID",
        "category": "Audio",
        "description": "유효하지 않은 AudioID 값",
        "cause": "AudioID가 0 이하이거나 존재하지 않음",
        "solution": "올바른 AudioID 전달 확인",
    },
    "AUD002": {
        "message": "Audio not found",
        "category": "Audio",
        "description": "요청한 AudioID에 대한 오디오 데이터 없음",
        "cause": "tb_audioCache에 해당 레코드 미존재",
        "solution": "AudioID 유효성 확인 및 캐시 재생성",
    },
    "AUD003": {
        "message": "Audio data is empty",
        "category": "Audio",
        "description": "AudioID에 대한 오디오 데이터가 비어있음",
        "cause": "TTS 생성 실패 후 빈 레코드가 저장됨",
        "solution": "해당 캐시 엔트리 삭제 후 재생성",
    },
    "GEN001": {
        "message": "Unknown error occurred",
        "category": "General",
        "description": "분류되지 않은 예외 발생",
        "cause": "시스템 오류 또는 미처리 예외",
        "solution": "시스템 로그 확인",
    },
}


def get_error_info(error_code: str) -> Dict[str, str]:
    """에러 코드로 에러 정보 딕셔너리 반환."""
    return _TTS_ERROR_CODES.get(
        error_code,
        {
            "message": "Unknown error code",
            "category": "General",
            "description": "Error code not found",
            "cause": "Unknown cause",
            "solution": "Contact system administrator",
        },
    )


def get_detailed_error_message(error_code: str, additional_info: str = "") -> str:
    """상세 에러 메시지 반환 (설명 + 추가 정보)."""
    error_info = get_error_info(error_code)
    detailed_message = f"{error_info['message']}"
    if error_info["description"]:
        detailed_message += f" - {error_info['description']}"
    if additional_info:
        detailed_message += f" | Details: {additional_info}"
    return detailed_message


def log_error_with_details(error_code: str, additional_info: str = "", logger=None):
    """파일 위치를 포함한 상세 에러 정보 로깅."""
    error_info = get_error_info(error_code)
    log_func = logger.error if logger else print

    try:
        stack = inspect.stack()
        target_frame = None
        for i in range(1, len(stack)):
            frame = stack[i]
            if frame.function not in [
                "_log_error_with_details", "log_error_with_details",
                "_get_error_info", "get_error_info",
                "_get_detailed_error_message", "get_detailed_error_message",
            ]:
                target_frame = frame
                break

        if target_frame:
            filename = os.path.basename(target_frame.filename)
            location = f"{filename}:{target_frame.lineno} in {target_frame.function}()"
        else:
            location = "Unknown location"
    except Exception as e:
        location = f"Unable to determine location: {str(e)}"

    log_func(f"Error {error_code}: {error_info['message']}")
    log_func(f"Location: {location}")
    if error_info["description"]:
        log_func(f"Description: {error_info['description']}")
    if error_info["cause"]:
        log_func(f"Cause: {error_info['cause']}")
    if error_info["solution"]:
        log_func(f"Solution: {error_info['solution']}")
    if additional_info:
        log_func(f"Additional Info: {additional_info}")


# ============================================================================
# GracefulShutdownMixin
# ============================================================================


class GracefulShutdownMixin:
    """
    ROS2 노드에 Publish_HeaRo_Process_Kill 토픽 기반 정상 종료 기능을 부여하는 Mixin.

    사용법:
        class MyNode(GracefulShutdownMixin, Node):
            def __init__(self):
                super().__init__('my_node')
                self._init_graceful_shutdown()

            def _graceful_cleanup(self):
                pass  # 노드별 리소스 정리
    """

    _shutdown_requested: bool = False

    def _init_graceful_shutdown(self):
        """정상 종료 구독 초기화. super().__init__() 이후 호출."""
        from std_msgs.msg import Empty

        self._shutdown_requested = False
        self._kill_subscription = self.create_subscription(
            Empty, "Publish_HeaRo_Process_Kill", self._on_process_kill_received, 10
        )
        self.get_logger().debug("[GracefulShutdown] Subscribed to Publish_HeaRo_Process_Kill")

    def _on_process_kill_received(self, msg):
        """Kill 토픽 수신 → 3단계 종료: cleanup → GPU 해제 → SIGINT."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        node_name = self.get_name() if hasattr(self, "get_name") else "unknown"
        self.get_logger().warn(
            f"[GracefulShutdown] Kill signal received on [{node_name}]. "
            f"Phase 1: Cleaning up node-specific resources..."
        )

        try:
            self._graceful_cleanup()
            self.get_logger().info(
                f"[GracefulShutdown] [{node_name}] Phase 1 complete: resources cleaned up."
            )
        except Exception as e:
            self.get_logger().error(f"[GracefulShutdown] [{node_name}] Phase 1 error: {e}")

        self.get_logger().info(f"[GracefulShutdown] [{node_name}] Phase 2: Releasing GPU memory...")
        self._force_release_gpu_memory()

        self.get_logger().warn(
            f"[GracefulShutdown] [{node_name}] Phase 3: Scheduling process termination..."
        )
        self._shutdown_timer = self.create_timer(0.1, self._deferred_sigint)

    def _graceful_cleanup(self):
        """서브클래스에서 오버라이드: 노드별 리소스 정리."""
        pass

    def _force_release_gpu_memory(self):
        """torch.cuda.empty_cache + gc.collect으로 GPU 메모리 강제 해제."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                self.get_logger().info(
                    "[GracefulShutdown] GPU memory released (torch.cuda.empty_cache)"
                )
        except ImportError:
            pass
        except Exception as e:
            self.get_logger().warn(f"[GracefulShutdown] GPU release warning: {e}")

        gc.collect()

    def _deferred_sigint(self):
        """콜백 반환 후 SIGINT 전송 → spin() 정상 탈출."""
        import signal

        if hasattr(self, "_shutdown_timer") and self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
            self._shutdown_timer = None

        os.kill(os.getpid(), signal.SIGINT)


__all__ = [
    "AnsiColor",
    "print_color",
    "format_colored",
    "get_error_info",
    "get_detailed_error_message",
    "log_error_with_details",
    "GracefulShutdownMixin",
]
