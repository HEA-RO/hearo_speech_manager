#!/usr/bin/env python3
"""
AudioCacheManager — tb_audioCache에 대한 pymysql 직접 접근을 캡슐화.

hearo_speech_manager의 TTS Handler Node가 캐시 조회/저장을 수행할 때
이 클래스를 통해 단일 수정 지점을 확보합니다.
"""

import io
import time
from typing import Optional, Tuple

import pymysql
from pymysql.cursors import DictCursor
from pydub import AudioSegment

from hearo_speech_manager.config.speech_config import (
    CHUNK_DURATION_MS,
    get_database_config,
)


class AudioCacheManager:
    """tb_audioCache 테이블에 대한 CRUD 캡슐화."""

    def __init__(self, db_config: Optional[dict] = None, logger=None):
        self._db_config = db_config or get_database_config()
        self._logger = logger

    def _log(self, msg: str, level: str = "info"):
        if self._logger:
            getattr(self._logger, level, self._logger.info)(msg)
        else:
            print(msg, flush=True)

    def _get_connection(self):
        return pymysql.connect(**self._db_config)

    # ------------------------------------------------------------------ #
    # 캐시 조회
    # ------------------------------------------------------------------ #
    def check_cache(self, audio_text: str) -> Optional[int]:
        """
        tb_audioCache에서 strAudioText로 AudioID 조회.
        캐시 히트 시 AudioID 반환, 미스 시 None.
        """
        connection = None
        try:
            connection = self._get_connection()
            with connection.cursor(DictCursor) as cursor:
                cursor.execute(
                    "SELECT AudioID FROM tb_audioCache WHERE strAudioText = %s LIMIT 1",
                    (audio_text,),
                )
                row = cursor.fetchone()
                if row:
                    return row["AudioID"]
                return None
        except Exception as e:
            self._log(f"Cache check error: {e}", "error")
            return None
        finally:
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # 캐시 저장 (INSERT ... ON DUPLICATE KEY UPDATE)
    # ------------------------------------------------------------------ #
    def save_to_cache(
        self,
        audio_text: str,
        audio_bytes: bytes,
        tts_model: str,
    ) -> Tuple[int, int, float]:
        """
        TTS 생성 결과를 tb_audioCache에 저장.

        Returns:
            (audio_id, n_chunks, audio_duration_sec)
        """
        connection = None
        try:
            n_chunks = self._calculate_n_chunks(audio_bytes)
            n_sent_chunks = self._calculate_sentence_chunks(audio_bytes)
            duration_sec = self._get_audio_duration_sec(audio_bytes)

            connection = self._get_connection()
            with connection.cursor(DictCursor) as cursor:
                sql = """
                    INSERT INTO tb_audioCache
                        (strAudioText, aStreaming, nChunk, nSentChunk, ttsModel, regDt, edDt)
                    VALUES
                        (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE
                        aStreaming = VALUES(aStreaming),
                        nChunk = VALUES(nChunk),
                        nSentChunk = VALUES(nSentChunk),
                        ttsModel = VALUES(ttsModel),
                        edDt = NOW()
                """
                cursor.execute(
                    sql,
                    (audio_text, audio_bytes, n_chunks, n_sent_chunks, tts_model),
                )
                connection.commit()

                if cursor.lastrowid and cursor.lastrowid > 0:
                    audio_id = cursor.lastrowid
                else:
                    cursor.execute(
                        "SELECT AudioID FROM tb_audioCache WHERE strAudioText = %s",
                        (audio_text,),
                    )
                    row = cursor.fetchone()
                    audio_id = row["AudioID"] if row else -1

            self._log(f"Audio cached: AudioID={audio_id}, chunks={n_chunks}, "
                       f"duration={duration_sec:.2f}s, model={tts_model}")
            return audio_id, n_chunks, duration_sec

        except Exception as e:
            self._log(f"Cache save error: {e}", "error")
            raise
        finally:
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # 오디오 메타 계산
    # ------------------------------------------------------------------ #
    def _calculate_n_chunks(self, audio_bytes: bytes) -> int:
        """오디오 바이트에서 시간 기반 청크 수 계산."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            duration_ms = len(audio)
            return max(1, (duration_ms + CHUNK_DURATION_MS - 1) // CHUNK_DURATION_MS)
        except Exception:
            return 1

    def _calculate_sentence_chunks(self, audio_bytes: bytes) -> int:
        """오디오 바이트에서 문장 기반 청크 수 계산 (현재는 시간 기반과 동일)."""
        return self._calculate_n_chunks(audio_bytes)

    def _get_audio_duration_sec(self, audio_bytes: bytes) -> float:
        """오디오 바이트에서 총 길이(초) 계산."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return len(audio) / 1000.0
        except Exception:
            return 0.0
