#!/usr/bin/env python3
"""
Context-Aware Text Segmenter for Zonos TTS
Zonos TTS를 위한 컨텍스트 인식 텍스트 분할기

Splits long text into segments with overlapping context windows
to maintain prosodic continuity across segment boundaries.

긴 텍스트를 오버래핑 컨텍스트 윈도우와 함께 세그먼트로 분할하여
세그먼트 경계에서의 운율 연속성을 유지합니다.

Stage 1 of the 3-Stage Quality Pipeline:
  [ContextAwareSegmenter] → [Continuity Generator] → [SpectralAudioAssembler]
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentPlan:
    """
    Plan for a single TTS segment, including overlap context metadata.
    오버랩 컨텍스트 메타데이터를 포함한 단일 TTS 세그먼트 계획.

    Attributes:
        text: Full text to feed to the TTS model (prefix_context + core_text).
              TTS 모델에 입력할 전체 텍스트 (prefix_context + core_text).
        core_text: The actual new content for this segment.
                   이 세그먼트의 실제 신규 콘텐츠.
        prefix_context: Trailing sentence(s) from the previous segment,
                        prepended for prosodic continuity.
                        운율 연속성을 위해 앞에 추가된 이전 세그먼트의 마지막 문장.
        prefix_char_count: Number of characters belonging to the prefix context.
                           Used by the generator to calculate trim offset.
                           프리픽스 컨텍스트에 해당하는 글자 수.
                           생성기에서 트림 오프셋 계산에 사용.
        segment_index: 0-based index within the full segment list.
                       전체 세그먼트 목록 내 0-기반 인덱스.
        total_segments: Total number of segments for this text.
                        이 텍스트의 전체 세그먼트 수.
    """

    text: str
    core_text: str
    prefix_context: str = ""
    prefix_char_count: int = 0
    segment_index: int = 0
    total_segments: int = 1


# Korean/English/Japanese/Chinese sentence-ending pattern
# 한국어/영어/일본어/중국어 문장 종결 패턴
_SENTENCE_ENDINGS = re.compile(r"(?<=[.!?。！？])\s*")


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation-based boundaries.
    문장 부호 기반 경계를 사용하여 텍스트를 문장으로 분할합니다.

    Handles Korean particles that may follow punctuation (e.g. "다." "요.").
    문장 부호 뒤에 오는 한국어 조사도 처리합니다.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings (whitespace-stripped, non-empty).
    """
    parts = _SENTENCE_ENDINGS.split(text)
    sentences: List[str] = []

    for part in parts:
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)

    if not sentences and text.strip():
        sentences.append(text.strip())

    return sentences


def _rejoin_short_sentences(sentences: List[str], min_length: int) -> List[str]:
    """
    Merge sentences shorter than *min_length* into their neighbours.
    *min_length*보다 짧은 문장을 인접 문장과 병합합니다.

    This prevents tiny fragments that waste a full TTS generation call.
    전체 TTS 생성 호출을 낭비하는 작은 조각을 방지합니다.
    """
    if not sentences:
        return sentences

    merged: List[str] = [sentences[0]]
    for s in sentences[1:]:
        if len(merged[-1]) < min_length:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)

    # Final pass: if last segment is too short, merge it backwards
    if len(merged) > 1 and len(merged[-1]) < min_length:
        merged[-2] = merged[-2] + " " + merged[-1]
        merged.pop()

    return merged


def _group_sentences_into_segments(sentences: List[str], max_length: int) -> List[List[str]]:
    """
    Group sentences into segments that respect *max_length* character limit.
    *max_length* 글자 제한을 준수하면서 문장을 세그먼트로 그룹화합니다.

    Each segment is a list of sentence strings.
    """
    groups: List[List[str]] = []
    current_group: List[str] = []
    current_len = 0

    for sentence in sentences:
        added_len = len(sentence) + (1 if current_group else 0)

        if current_group and current_len + added_len > max_length:
            groups.append(current_group)
            current_group = [sentence]
            current_len = len(sentence)
        else:
            current_group.append(sentence)
            current_len += added_len

    if current_group:
        groups.append(current_group)

    return groups


def segment(
    text: str,
    max_length: int = 120,
    overlap_sentences: int = 1,
    overlap_enabled: bool = True,
    min_segment_length: int = 30,
    split_sentences: bool = True,
) -> List[SegmentPlan]:
    """
    Split *text* into SegmentPlans with overlapping context windows.
    *text*를 오버래핑 컨텍스트 윈도우와 함께 SegmentPlan으로 분할합니다.

    For segment N (N > 0), the last *overlap_sentences* sentences of
    segment N-1 are prepended as prefix context. The generator will
    produce audio for the full text but trim the prefix portion,
    yielding natural prosodic continuity at segment boundaries.

    세그먼트 N (N > 0)에 대해 세그먼트 N-1의 마지막 *overlap_sentences*개
    문장이 프리픽스 컨텍스트로 앞에 추가됩니다. 생성기는 전체 텍스트에 대해
    오디오를 생성하지만 프리픽스 부분을 잘라내어 세그먼트 경계에서
    자연스러운 운율 연속성을 제공합니다.

    Args:
        text: Full text to segment.
        max_length: Maximum characters per core segment (excluding prefix).
        overlap_sentences: Number of trailing sentences from previous segment
                           to include as prefix context (0 = disabled).
        overlap_enabled: Master switch for overlap context.
        min_segment_length: Minimum characters for a standalone segment;
                            shorter fragments are merged with neighbours.
        split_sentences: If True, split on sentence boundaries;
                         if False, split on raw character count.

    Returns:
        List of SegmentPlan dataclass instances.
    """
    text = text.strip()
    if not text:
        return []

    # Short text: single segment, no overlap needed
    if len(text) <= max_length:
        return [
            SegmentPlan(
                text=text,
                core_text=text,
                prefix_context="",
                prefix_char_count=0,
                segment_index=0,
                total_segments=1,
            )
        ]

    if not split_sentences:
        return _fallback_char_split(text, max_length)

    sentences = _split_into_sentences(text)
    sentences = _rejoin_short_sentences(sentences, min_segment_length)

    if not sentences:
        return _fallback_char_split(text, max_length)

    groups = _group_sentences_into_segments(sentences, max_length)

    if not groups:
        return _fallback_char_split(text, max_length)

    effective_overlap = overlap_sentences if overlap_enabled else 0
    total = len(groups)
    plans: List[SegmentPlan] = []

    for idx, group in enumerate(groups):
        core_text = " ".join(group)

        if idx == 0 or effective_overlap == 0:
            plans.append(
                SegmentPlan(
                    text=core_text,
                    core_text=core_text,
                    prefix_context="",
                    prefix_char_count=0,
                    segment_index=idx,
                    total_segments=total,
                )
            )
        else:
            prev_group = groups[idx - 1]
            overlap_sents = prev_group[-effective_overlap:]
            prefix_context = " ".join(overlap_sents)
            full_text = prefix_context + " " + core_text

            plans.append(
                SegmentPlan(
                    text=full_text,
                    core_text=core_text,
                    prefix_context=prefix_context,
                    prefix_char_count=len(prefix_context)
                    + 1,  # +1 for the separator space in full_text
                    segment_index=idx,
                    total_segments=total,
                )
            )

    return plans


def _fallback_char_split(text: str, max_length: int) -> List[SegmentPlan]:
    """
    Fallback: split by raw character count when sentence splitting is disabled.
    문장 분할이 비활성화된 경우 원시 글자 수로 분할하는 폴백.
    """
    plans: List[SegmentPlan] = []
    total = (len(text) + max_length - 1) // max_length

    for i in range(total):
        chunk = text[i * max_length : (i + 1) * max_length]
        plans.append(
            SegmentPlan(
                text=chunk,
                core_text=chunk,
                prefix_context="",
                prefix_char_count=0,
                segment_index=i,
                total_segments=total,
            )
        )

    return plans
