#!/usr/bin/env python3
"""
HeaRo Speech Manager Launch File

VAD, Google STT, Whisper STT, TTS Handler 노드를 실행합니다.
각 노드를 개별 활성화/비활성화할 수 있으며, TTS Handler는
PYTORCH_CUDA_ALLOC_CONF 환경변수와 DB 접속 환경변수를 설정합니다.

사용법:
    # 전체 노드 실행 (OPENAI_API_KEY 전달)
    ros2 launch hearo_speech_manager hearo_speech_manager.launch.py \\
        openai_api_key:="sk-proj-..."

    # TTS Handler만 실행
    ros2 launch hearo_speech_manager hearo_speech_manager.launch.py \\
        openai_api_key:="sk-proj-..." \\
        use_vad:=false use_google_stt:=false use_stt_node:=false

    # VAD + STT만 실행 (TTS 비활성화)
    ros2 launch hearo_speech_manager hearo_speech_manager.launch.py use_tts_handler:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, EnvironmentVariable


def generate_launch_description():

    # ========== Launch Arguments ==========

    use_vad_arg = DeclareLaunchArgument(
        "use_vad", default_value="true", description="Enable VAD node"
    )
    use_google_stt_arg = DeclareLaunchArgument(
        "use_google_stt", default_value="true", description="Enable Google STT node"
    )
    use_stt_node_arg = DeclareLaunchArgument(
        "use_stt_node", default_value="true", description="Enable Whisper STT node"
    )
    use_tts_handler_arg = DeclareLaunchArgument(
        "use_tts_handler", default_value="true", description="Enable TTS Handler node"
    )

    # OpenAI API Key (required for TTS)
    openai_api_key_arg = DeclareLaunchArgument(
        "openai_api_key",
        default_value=EnvironmentVariable("OPENAI_API_KEY", default_value=""),
        description="OpenAI API Key for TTS generation",
    )

    # DB environment variable arguments
    db_host_arg = DeclareLaunchArgument(
        "db_host", default_value="127.0.0.1", description="Database host"
    )
    db_port_arg = DeclareLaunchArgument(
        "db_port", default_value="3306", description="Database port"
    )
    db_user_arg = DeclareLaunchArgument(
        "db_user", default_value="HeaRo", description="Database user"
    )
    db_password_arg = DeclareLaunchArgument(
        "db_password", default_value="yslim_005064", description="Database password"
    )
    db_name_arg = DeclareLaunchArgument(
        "db_name", default_value="HeaRo", description="Database name"
    )

    # ========== Launch Configurations ==========

    use_vad = LaunchConfiguration("use_vad")
    use_google_stt = LaunchConfiguration("use_google_stt")
    use_stt_node = LaunchConfiguration("use_stt_node")
    use_tts_handler = LaunchConfiguration("use_tts_handler")
    openai_api_key = LaunchConfiguration("openai_api_key")

    db_host = LaunchConfiguration("db_host")
    db_port = LaunchConfiguration("db_port")
    db_user = LaunchConfiguration("db_user")
    db_password = LaunchConfiguration("db_password")
    db_name = LaunchConfiguration("db_name")

    # ========== Shared environment for all nodes ==========

    shared_env = {
        "OPENAI_API_KEY": openai_api_key,
    }

    # ========== Banner ==========

    startup_banner = LogInfo(
        msg=[
            "\n",
            "=" * 80, "\n",
            "  HeaRo Speech Manager\n",
            "  VAD / Google STT / Whisper STT / TTS Handler\n",
            "=" * 80, "\n",
        ]
    )

    # ========== Nodes ==========

    vad_node = ExecuteProcess(
        cmd=["python3", "-m", "hearo_speech_manager.nodes.vad"],
        name="HeaRo_VAD",
        output="screen",
        additional_env=shared_env,
        condition=IfCondition(use_vad),
    )

    google_stt_node = ExecuteProcess(
        cmd=["python3", "-m", "hearo_speech_manager.nodes.google_stt"],
        name="HeaRo_Google_STT",
        output="screen",
        additional_env=shared_env,
        condition=IfCondition(use_google_stt),
    )

    stt_node = ExecuteProcess(
        cmd=["python3", "-m", "hearo_speech_manager.nodes.stt_node"],
        name="HeaRo_Whisper_STT",
        output="screen",
        additional_env=shared_env,
        condition=IfCondition(use_stt_node),
    )

    tts_handler_node = ExecuteProcess(
        cmd=["python3", "-m", "hearo_speech_manager.nodes.HeaRo_TTS_Handler_Node"],
        name="HeaRo_TTS_Handler",
        output="screen",
        additional_env={
            **shared_env,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HEARO_DB_HOST": db_host,
            "HEARO_DB_PORT": db_port,
            "HEARO_DB_USER": db_user,
            "HEARO_DB_PASSWORD": db_password,
            "HEARO_DB_NAME": db_name,
        },
        condition=IfCondition(use_tts_handler),
    )

    return LaunchDescription(
        [
            # Arguments
            use_vad_arg,
            use_google_stt_arg,
            use_stt_node_arg,
            use_tts_handler_arg,
            openai_api_key_arg,
            db_host_arg,
            db_port_arg,
            db_user_arg,
            db_password_arg,
            db_name_arg,
            # Startup
            startup_banner,
            # Nodes
            vad_node,
            google_stt_node,
            stt_node,
            tts_handler_node,
        ]
    )
