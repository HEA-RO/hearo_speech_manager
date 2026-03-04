#!/usr/bin/env python3
"""
Spectral Audio Assembler for Zonos TTS
Zonos TTS를 위한 스펙트럴 오디오 어셈블러

Assembles multiple audio segments into a seamless output using
spectral-domain crossfading and RMS normalisation.

여러 오디오 세그먼트를 스펙트럴 도메인 크로스페이딩과 RMS 정규화를
사용하여 매끄러운 출력으로 조립합니다.

Stage 3 of the 3-Stage Quality Pipeline:
  [ContextAwareSegmenter] → [Continuity Generator] → [SpectralAudioAssembler]
"""

import io
from typing import List, Optional

import numpy as np

try:
    from scipy.signal import stft, istft

    _SCIPY_SIGNAL_AVAILABLE = True
except ImportError:
    _SCIPY_SIGNAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# RMS Normalisation
# ---------------------------------------------------------------------------


def _rms(signal: np.ndarray) -> float:
    """Root-mean-square energy of a 1-D signal."""
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def rms_normalize(
    segments: List[np.ndarray],
    target_rms: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Normalise RMS energy across *segments* so that volume is consistent.
    *segments*의 RMS 에너지를 정규화하여 볼륨을 일관되게 합니다.

    Args:
        segments: List of 1-D float numpy arrays.
        target_rms: If None, uses the mean RMS of all segments.

    Returns:
        List of normalised segments (same length, new arrays).
    """
    rms_values = [_rms(s) for s in segments]

    if target_rms is None:
        non_zero = [r for r in rms_values if r > 1e-8]
        if not non_zero:
            return segments
        target_rms = float(np.mean(non_zero))

    normalised: List[np.ndarray] = []
    for seg, seg_rms in zip(segments, rms_values):
        if seg_rms < 1e-8:
            normalised.append(seg.copy())
        else:
            scale = target_rms / seg_rms
            # Clamp scale to avoid extreme amplification
            scale = min(scale, 4.0)
            normalised.append((seg * scale).astype(seg.dtype))

    return normalised


# ---------------------------------------------------------------------------
# Spectral Crossfade
# ---------------------------------------------------------------------------


def _spectral_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    overlap_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Blend *left* and *right* in the frequency domain over *overlap_samples*.
    *overlap_samples* 구간에서 *left*와 *right*를 주파수 도메인에서 블렌딩합니다.

    Uses STFT → weighted magnitude/phase averaging → ISTFT.
    Phase mismatch wobble is avoided because the blend window is short
    and the weighting is smooth (raised-cosine).

    Args:
        left: Tail portion of the preceding segment (1-D float array).
        right: Head portion of the following segment (1-D float array).
        overlap_samples: Number of samples in the overlap region.
        sample_rate: Audio sample rate for STFT parameters.

    Returns:
        Blended overlap region as 1-D float numpy array.
    """
    if not _SCIPY_SIGNAL_AVAILABLE:
        return _time_domain_crossfade(left, right, overlap_samples)

    n = min(overlap_samples, len(left), len(right))
    if n < 4:
        return _time_domain_crossfade(left, right, overlap_samples)

    left_tail = left[-n:].astype(np.float64)
    right_head = right[:n].astype(np.float64)

    nperseg = min(256, n)
    if nperseg < 4:
        return _time_domain_crossfade(left, right, overlap_samples)

    noverlap = nperseg // 2

    try:
        _, _, Zl = stft(left_tail, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, _, Zr = stft(right_head, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

        n_frames = min(Zl.shape[1], Zr.shape[1])
        Zl = Zl[:, :n_frames]
        Zr = Zr[:, :n_frames]

        # Raised-cosine weighting across STFT frames
        weight = np.linspace(0.0, 1.0, n_frames)[np.newaxis, :]
        weight_cos = 0.5 * (1.0 - np.cos(np.pi * weight))

        mag_l, phase_l = np.abs(Zl), np.angle(Zl)
        mag_r, phase_r = np.abs(Zr), np.angle(Zr)

        mag_blend = mag_l * (1.0 - weight_cos) + mag_r * weight_cos

        # Phase: use the dominant source's phase at each frame
        phase_blend = np.where(weight_cos < 0.5, phase_l, phase_r)

        Z_blend = mag_blend * np.exp(1j * phase_blend)

        _, blended = istft(Z_blend, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        blended = blended[:n]

        if len(blended) < n:
            pad = np.zeros(n - len(blended), dtype=np.float64)
            blended = np.concatenate([blended, pad])

        return blended.astype(np.float32)

    except Exception:
        return _time_domain_crossfade(left, right, overlap_samples)


def _time_domain_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    overlap_samples: int,
) -> np.ndarray:
    """
    Simple raised-cosine crossfade in the time domain (fallback).
    시간 도메인에서의 간단한 레이즈드-코사인 크로스페이드 (폴백).
    """
    n = min(overlap_samples, len(left), len(right))
    if n == 0:
        return np.array([], dtype=np.float32)

    left_tail = left[-n:].astype(np.float64)
    right_head = right[:n].astype(np.float64)

    t = np.linspace(0.0, 1.0, n)
    fade_out = 0.5 * (1.0 + np.cos(np.pi * t))
    fade_in = 1.0 - fade_out

    blended = left_tail * fade_out + right_head * fade_in
    return blended.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble(
    segments: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
    rms_normalize_enabled: bool = True,
    spectral_crossfade_enabled: bool = True,
) -> np.ndarray:
    """
    Assemble *segments* into a single continuous audio array.
    *segments*를 하나의 연속 오디오 배열로 조립합니다.

    Pipeline:
      1. RMS normalisation across all segments (if enabled).
      2. Spectral crossfade at each junction (if enabled, else time-domain).
      3. Concatenation of non-overlapping + blended regions.

    Args:
        segments: List of 1-D float32 numpy arrays (mono audio).
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: Duration of crossfade overlap in milliseconds.
        rms_normalize_enabled: Whether to apply RMS normalisation.
        spectral_crossfade_enabled: Use spectral crossfade (True) or
                                    time-domain crossfade (False).

    Returns:
        Assembled 1-D float32 numpy array.
    """
    if not segments:
        return np.array([], dtype=np.float32)

    if len(segments) == 1:
        seg = segments[0]
        if rms_normalize_enabled:
            seg = rms_normalize([seg])[0]
        return seg

    if rms_normalize_enabled:
        segments = rms_normalize(segments)

    overlap_samples = int(sample_rate * crossfade_ms / 1000)

    parts: List[np.ndarray] = []

    for i, seg in enumerate(segments):
        if i == 0:
            if overlap_samples > 0 and len(seg) > overlap_samples:
                parts.append(seg[:-overlap_samples])
            else:
                parts.append(seg)
            continue

        prev = segments[i - 1]
        actual_overlap = min(overlap_samples, len(prev), len(seg))

        if actual_overlap > 0:
            if spectral_crossfade_enabled:
                blended = _spectral_crossfade(prev, seg, actual_overlap, sample_rate)
            else:
                blended = _time_domain_crossfade(prev, seg, actual_overlap)
            parts.append(blended)

            if len(seg) > actual_overlap:
                remaining = seg[actual_overlap:]
                if (
                    i < len(segments) - 1
                    and overlap_samples > 0
                    and len(remaining) > overlap_samples
                ):
                    parts.append(remaining[:-overlap_samples])
                else:
                    parts.append(remaining)
        else:
            if i < len(segments) - 1 and overlap_samples > 0 and len(seg) > overlap_samples:
                parts.append(seg[:-overlap_samples])
            else:
                parts.append(seg)

    non_empty = [p for p in parts if len(p) > 0]
    if not non_empty:
        return np.array([], dtype=np.float32)

    return np.concatenate(non_empty).astype(np.float32)
