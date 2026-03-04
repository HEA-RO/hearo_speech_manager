#!/usr/bin/env python3
"""
Audio Boundary Detector for Zonos TTS Quality Pipeline
Zonos TTS 품질 파이프라인을 위한 오디오 경계 탐지기

Detects sentence boundaries in generated audio by analysing
frame-level RMS energy to locate pause valleys. Used by the
Continuity Generator (Stage 2) to precisely trim overlapping
prefix audio instead of relying on unreliable character-ratio
estimates.

생성된 오디오에서 프레임별 RMS 에너지를 분석하여 pause valley를
찾아 문장 경계를 탐지합니다. Continuity Generator(2단계)에서
글자 비율 추정 대신 정확한 프리픽스 오디오 트리밍에 사용됩니다.

Pipeline context:
  [ContextAwareSegmenter] → [Continuity Generator + **AudioBoundaryDetector**] → [SpectralAudioAssembler]
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Korean/CJK punctuation weights: estimated audio-duration multiplier
# relative to a single syllable character (weight=1.0).
# 한국어/CJK 구두점 가중치: 단일 음절 대비 오디오 지속 시간 배수
_DEFAULT_PUNCTUATION_WEIGHTS: Dict[str, float] = {
    ".": 3.0,
    "!": 3.0,
    "?": 3.0,
    "。": 3.0,
    "！": 3.0,
    "？": 3.0,
    ",": 1.5,
    "、": 1.5,
    " ": 0.3,
}


def estimate_punctuation_aware_ratio(
    text: str,
    boundary_char_idx: int,
    punctuation_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Estimate audio-duration ratio at a character boundary, weighting
    punctuation marks by their typical pause duration.

    구두점의 실제 pause 길이를 반영하여 문자 경계에서의
    오디오 duration 비율을 추정합니다.

    Unlike a naive ``boundary_char_idx / len(text)`` ratio, this gives
    heavier weight to sentence-ending punctuation (period ~300ms pause)
    than to regular syllables (~100-150ms each), yielding a more
    accurate estimate of where the boundary falls in the audio.

    Args:
        text: Full text (prefix + core) passed to TTS.
        boundary_char_idx: Character index where prefix ends
                           (= ``prefix_char_count``).
        punctuation_weights: Override character-weight mapping.
                             Defaults to ``_DEFAULT_PUNCTUATION_WEIGHTS``.

    Returns:
        Estimated fraction (0.0–1.0) of total audio duration that
        corresponds to ``text[:boundary_char_idx]``.
    """
    weights = punctuation_weights or _DEFAULT_PUNCTUATION_WEIGHTS

    weighted_before = 0.0
    weighted_total = 0.0
    for i, char in enumerate(text):
        w = weights.get(char, 1.0)
        weighted_total += w
        if i < boundary_char_idx:
            weighted_before += w

    if weighted_total < 1e-10:
        return 0.5
    return weighted_before / weighted_total


def _compute_frame_energy(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
) -> np.ndarray:
    """
    Compute per-frame RMS energy of a 1-D audio signal.
    1-D 오디오 신호의 프레임별 RMS 에너지를 계산합니다.

    Args:
        audio: 1-D float numpy array (mono).
        sample_rate: Sample rate in Hz.
        frame_ms: Frame length in milliseconds.

    Returns:
        1-D numpy array of RMS energy values, one per frame.
    """
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame_len)

    energy = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        frame = audio[start:end].astype(np.float64)
        energy[i] = np.sqrt(np.mean(frame**2))

    return energy


def _find_valleys(
    energy: np.ndarray,
    threshold_ratio: float = 0.15,
    min_consecutive: int = 1,
) -> List[Tuple[int, int, float]]:
    """
    Find contiguous low-energy regions (valleys) in the energy curve.
    에너지 곡선에서 연속적인 저에너지 영역(valley)을 찾습니다.

    A valley is a run of consecutive frames whose energy is below
    ``threshold_ratio * median_energy``.

    valley는 에너지가 ``threshold_ratio * median_energy`` 미만인
    연속 프레임의 연속 구간입니다.

    Args:
        energy: 1-D array of frame RMS energy values.
        threshold_ratio: Fraction of median energy below which a frame
                         is considered "silent" (0.15 = 15% of median).
        min_consecutive: Minimum number of consecutive low-energy frames
                         to qualify as a valley.

    Returns:
        List of (start_frame, end_frame, min_energy) tuples.
        NOT sorted — caller decides selection strategy.
    """
    if len(energy) == 0:
        return []

    median_e = float(np.median(energy))
    if median_e < 1e-10:
        return []

    threshold = median_e * threshold_ratio
    below = energy < threshold

    valleys: List[Tuple[int, int, float]] = []
    in_valley = False
    start = 0

    for i, is_low in enumerate(below):
        if is_low and not in_valley:
            in_valley = True
            start = i
        elif not is_low and in_valley:
            in_valley = False
            length = i - start
            if length >= min_consecutive:
                min_e = float(np.min(energy[start:i]))
                valleys.append((start, i, min_e))

    if in_valley:
        length = len(energy) - start
        if length >= min_consecutive:
            min_e = float(np.min(energy[start:]))
            valleys.append((start, len(energy), min_e))

    return valleys


def _score_valley(
    v_start: int,
    v_end: int,
    min_energy: float,
    estimated_center_frame: float,
    window_half_frames: float,
    all_min_energies: List[float],
    depth_weight: float = 0.4,
    proximity_weight: float = 0.6,
) -> float:
    """
    Score a valley by weighted combination of depth and proximity.
    Valley의 깊이와 추정 위치 근접도를 가중 결합하여 점수를 산출합니다.

    proximity(0.6) > depth(0.4): WHERE the valley is matters more
    than HOW deep it is, because the char-ratio estimate already
    narrows down the approximate location.

    Args:
        v_start, v_end: Valley frame range (window-local).
        min_energy: Minimum energy within this valley.
        estimated_center_frame: Expected trim position (window-local).
        window_half_frames: Half-width of the search window in frames.
        all_min_energies: Min-energy values of ALL detected valleys
                          (for normalisation).
        depth_weight: Weight for depth component (default 0.4).
        proximity_weight: Weight for proximity component (default 0.6).

    Returns:
        Combined score (higher = better candidate).
    """
    max_e = max(all_min_energies)
    min_e = min(all_min_energies)
    e_range = max_e - min_e

    depth_norm = 1.0 - (min_energy - min_e) / e_range if e_range > 1e-15 else 1.0

    v_center = (v_start + v_end) / 2.0
    dist = abs(v_center - estimated_center_frame)
    max_dist = max(window_half_frames, 1.0)
    proximity_norm = max(0.0, 1.0 - dist / max_dist)

    return depth_norm * depth_weight + proximity_norm * proximity_weight


def detect_sentence_boundary(
    audio: np.ndarray,
    sample_rate: int,
    estimated_ratio: float,
    search_window_ratio: float = 0.20,
    frame_ms: int = 20,
    min_valley_duration_ms: int = 60,
) -> int:
    """
    Detect the sentence-boundary pause nearest to *estimated_ratio*
    using proximity-scored valley selection.

    *estimated_ratio* 근처에서 proximity-scored valley 선택을 사용하여
    문장 경계 pause를 탐지합니다.

    Algorithm:
      1. Compute per-frame RMS energy over the entire audio.
      2. Define a search window around the estimated trim position
         (``estimated_ratio +/- search_window_ratio``).
      3. Within that window, find energy valleys (consecutive frames
         below 15% of median energy) that are at least
         ``min_valley_duration_ms`` long.
      4. Score each valley: ``0.4 * depth_norm + 0.6 * proximity_norm``
         — proximity to the estimate is weighted higher than depth,
         because comma pauses can be deeper than sentence-end pauses.
      5. Return the sample index at the **centre** of the highest-scored
         valley.
      6. Fallback: if no valley is found, return the estimated
         position + 5% overshoot (slightly aggressive trim is
         preferable to duplication).

    알고리즘:
      1. 전체 오디오의 프레임별 RMS 에너지를 계산합니다.
      2. 추정 위치 주변에 탐색 윈도우를 정의합니다.
      3. 윈도우 내에서 최소 ``min_valley_duration_ms`` 이상인 valley를 탐지합니다.
      4. 각 valley에 ``0.4 * depth + 0.6 * proximity`` 점수를 부여합니다.
         — 쉼표 pause가 문장 종결 pause보다 깊을 수 있으므로
         추정 위치 근접도에 더 높은 가중치를 둡니다.
      5. 최고 점수 valley의 **중앙** 샘플 인덱스를 반환합니다.
      6. Valley가 없으면 추정 위치 + 5% 오버슈트를 반환합니다.

    Args:
        audio: 1-D float numpy array (mono audio).
        sample_rate: Audio sample rate in Hz.
        estimated_ratio: Approximate trim position as a fraction of
                         total audio length (punctuation-aware estimate).
        search_window_ratio: Half-width of the search window around
                             *estimated_ratio* (0.20 = +/-20% of total).
        frame_ms: Frame length in ms for RMS energy computation.
        min_valley_duration_ms: Minimum duration of a valley to be
                                considered a sentence boundary (60ms
                                filters comma pauses while keeping
                                sentence-end pauses of 200-500ms).

    Returns:
        Sample index at which to trim (0-based). The caller should
        discard ``audio[:trim_sample]``.
    """
    total_samples = len(audio)
    if total_samples == 0:
        return 0

    energy = _compute_frame_energy(audio, sample_rate, frame_ms)
    if len(energy) == 0:
        return int(total_samples * estimated_ratio)

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    min_consecutive = max(1, math.ceil(min_valley_duration_ms / frame_ms))

    # Search window in frame indices
    est_frame = int(estimated_ratio * len(energy))
    window_half = max(1, int(search_window_ratio * len(energy)))
    win_start = max(0, est_frame - window_half)
    win_end = min(len(energy), est_frame + window_half)

    # Extract windowed energy and find valleys
    windowed_energy = energy[win_start:win_end]
    valleys = _find_valleys(
        windowed_energy,
        threshold_ratio=0.15,
        min_consecutive=min_consecutive,
    )

    if valleys:
        all_min_energies = [v[2] for v in valleys]
        est_frame_local = est_frame - win_start

        best_score = -1.0
        best_valley = valleys[0]
        for v_start, v_end, min_e in valleys:
            score = _score_valley(
                v_start=v_start,
                v_end=v_end,
                min_energy=min_e,
                estimated_center_frame=est_frame_local,
                window_half_frames=float(window_half),
                all_min_energies=all_min_energies,
            )
            if score > best_score:
                best_score = score
                best_valley = (v_start, v_end, min_e)

        v_start, v_end, _ = best_valley
        global_frame = win_start + (v_start + v_end) // 2
        trim_sample = global_frame * frame_len
        trim_sample = max(0, min(trim_sample, total_samples - 1))
        return trim_sample

    # --- Fallback: no valley found → aggressive char-ratio + 5% overshoot ---
    overshoot_ratio = min(estimated_ratio + 0.05, 0.95)
    fallback_sample = int(total_samples * overshoot_ratio)
    return max(0, min(fallback_sample, total_samples - 1))
