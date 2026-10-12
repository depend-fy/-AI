"""阶段 5 测试：修正生成与应用（纯逻辑，不联网、不花钱）。

要守住的边界：
    · 定位不到原文片段的修正 → 丢弃（不允许在错的位置上改字）；
    · 引用了不属于该主张的证据 → 剔除；
    · 改事实但没有证据 → 整条丢弃（宁可建议"弱化"，也不许编数字）；
    · 应用时区间重叠 / 越界 / 与原文对不上 → 跳过并记录，绝不静默处理。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from verimind.core.enums import ChangeType
from verimind.core.schemas import Claim, Correction, CorrectionBatch, CorrectionDraft, Span
from verimind.correction.applier import apply_corrections
from verimind.correction.generator import _resolve_in_sentence

TEXT = "某研究（DOI: 10.1038/nature12373）发表于 2008 年，显示准确率提高了 40%。"


def make_correction(
    *, correction_id: str = "K001", start: int, end: int, new_text: str, change_type: ChangeType
) -> Correction:
    return Correction(
        correction_id=correction_id,
        claim_id="C001",
        original_span=Span.from_text(TEXT, start, end),
        proposed_text=new_text,
        change_type=change_type,
        rationale="测试用",
        evidence_ids=["E001"] if change_type in (ChangeType.FIX_NUMBER, ChangeType.REPLACE_FACT) else [],
    )


# ==========================================================================
# 区间定位
# ==========================================================================
def test_resolve_in_sentence_finds_the_quote() -> None:
    sentence = Span.from_text(TEXT, 0, len(TEXT))
    span = _resolve_in_sentence(TEXT, "2008 年", sentence)
    assert span is not None and span.text == "2008 年"


def test_resolve_in_sentence_rejects_quote_outside_the_sentence() -> None:
    outside = Span.from_text(TEXT, 0, 6)  # 只覆盖"某研究（DOI"
    assert _resolve_in_sentence(TEXT, "2008 年", outside) is None


def test_resolve_in_sentence_returns_none_for_missing_quote() -> None:
    sentence = Span.from_text(TEXT, 0, len(TEXT))
    assert _resolve_in_sentence(TEXT, "1999 年", sentence) is None


# ==========================================================================
# 应用到文本
# ==========================================================================
def test_apply_single_correction() -> None:
    start = TEXT.index("2008 年")
    correction = make_correction(
        start=start, end=start + len("2008 年"), new_text="2013 年", change_type=ChangeType.FIX_NUMBER
    )
    outcome = apply_corrections(TEXT, [correction])
    assert outcome.corrected_text == TEXT.replace("2008 年", "2013 年")
    assert outcome.applied_ids == ["K001"]
    assert not outcome.skipped_ids


def test_apply_multiple_corrections_keeps_offsets_valid() -> None:
    year_start = TEXT.index("2008 年")
    pct_start = TEXT.index("40%")
    corrections = [
        make_correction(
            correction_id="K001",
            start=year_start,
            end=year_start + 4,
            new_text="2013 年",
            change_type=ChangeType.FIX_NUMBER,
        ),
        make_correction(
            correction_id="K002",
            start=pct_start,
            end=pct_start + 3,
            new_text="12%",
            change_type=ChangeType.FIX_NUMBER,
        ),
    ]
    outcome = apply_corrections(TEXT, corrections)
    assert "2013 年" in outcome.corrected_text
    assert "12%" in outcome.corrected_text
    assert "2008" not in outcome.corrected_text and "40%" not in outcome.corrected_text
    assert set(outcome.applied_ids) == {"K001", "K002"}


def test_apply_skips_overlapping_corrections() -> None:
    """区间重叠时只应用靠后的那条，另一条跳过并说明（不静默处理）。

    这是刻意的取舍：从文本末尾往前替换才能保证偏移量不被打乱，
    因此重叠时"靠后的那条"胜出。结果与选择都会被记录在 notes 里。
    """
    start = TEXT.index("2008 年")
    corrections = [
        make_correction(
            correction_id="K001",
            start=start,
            end=start + 4,
            new_text="2013 年",
            change_type=ChangeType.FIX_NUMBER,
        ),
        make_correction(
            correction_id="K002",
            start=start + 2,
            end=start + 6,
            new_text="乱改",
            change_type=ChangeType.WEAKEN,
        ),
    ]
    outcome = apply_corrections(TEXT, corrections)
    assert outcome.applied_ids == ["K002"]
    assert outcome.skipped_ids == ["K001"]
    assert any("重叠" in note for note in outcome.notes)


def test_apply_skips_when_span_does_not_match_text() -> None:
    """区间长度对但文本对不上 → 必须跳过（防止把改动落在错误的位置）。"""
    correction = make_correction(
        start=0, end=4, new_text="xx", change_type=ChangeType.WEAKEN
    )
    tampered = correction.model_copy(
        update={"original_span": Span(start=0, end=4, text="某研究X")}
    )
    outcome = apply_corrections(TEXT, [tampered])
    assert not outcome.applied_ids
    assert outcome.skipped_ids == ["K001"]
    assert any("不一致" in note for note in outcome.notes)


def test_apply_only_selected_ids() -> None:
    start = TEXT.index("2008 年")
    corrections = [
        make_correction(
            correction_id="K001", start=start, end=start + 4, new_text="2013 年",
            change_type=ChangeType.FIX_NUMBER,
        )
    ]
    outcome = apply_corrections(TEXT, corrections, only=set())
    assert outcome.applied_ids == []
    assert outcome.corrected_text == TEXT


def test_delete_change_type_allows_empty_text() -> None:
    start = TEXT.index("，显示准确率提高了 40%")
    correction = Correction(
        correction_id="K001",
        claim_id="C001",
        original_span=Span.from_text(TEXT, start, start + 13),
        proposed_text="",
        change_type=ChangeType.DELETE,
        rationale="该数值无任何出处，直接删除。",
    )
    outcome = apply_corrections(TEXT, [correction])
    assert "准确率" not in outcome.corrected_text


# ==========================================================================
# 数据契约
# ==========================================================================
def test_fact_changing_correction_requires_evidence_at_schema_level() -> None:
    start = TEXT.index("40%")
    with pytest.raises(ValidationError):
        Correction(
            correction_id="K001",
            claim_id="C001",
            original_span=Span.from_text(TEXT, start, start + 3),
            proposed_text="12%",
            change_type=ChangeType.FIX_NUMBER,
            rationale="改成摘要里的数值。",
            evidence_ids=[],
        )


def test_non_delete_correction_requires_non_empty_text() -> None:
    start = TEXT.index("40%")
    with pytest.raises(ValidationError):
        Correction(
            correction_id="K001",
            claim_id="C001",
            original_span=Span.from_text(TEXT, start, start + 3),
            proposed_text="   ",
            change_type=ChangeType.WEAKEN,
            rationale="弱化表述。",
        )


def test_draft_delete_may_have_empty_text() -> None:
    draft = CorrectionDraft(
        claim_id="C001",
        target_quote="40%",
        proposed_text="",
        change_type=ChangeType.DELETE,
        rationale="删除无出处的数值。",
    )
    assert draft.proposed_text == ""


def test_correction_batch_parses_from_json() -> None:
    payload = {
        "corrections": [
            {
                "claim_id": "C001",
                "target_quote": "40%",
                "proposed_text": "12%",
                "change_type": "更正数字",
                "rationale": "摘要写的是 12%。",
                "evidence_ids": ["E002"],
            }
        ],
        "notes": [],
    }
    batch = CorrectionBatch.model_validate(payload)
    assert batch.corrections[0].change_type is ChangeType.FIX_NUMBER
