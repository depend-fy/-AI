"""阶段 2 测试：原子主张拆解。

重点验证三件事：
    1. 位置精确：每条主张的 `source_span.text` 必须等于原文对应切片（否则报告会高亮错地方）；
    2. 不重复、不注水：一句话不会产出多条同义主张（回归：模糊引用句曾一次产出 3 条）；
    3. 诚实：观点类主张必须标为不可核验，而不是硬当成事实错误。
"""

from __future__ import annotations

from verimind.claims.extractor import RuleBasedExtractor, collect_drafts, drafts_to_claims
from verimind.core.enums import Checkability, ClaimPriority, ClaimType, ExtractionConfidence
from verimind.core.schemas import ClaimDraft, ClaimExtraction

EXTRACTOR = RuleBasedExtractor()

FAKE_DOI_TEXT = (
    "据一项发表于 2023 年的研究（DOI: 10.xxxx/verimind.2023.001）显示，"
    "该团队提出的新方法可将图像分类准确率提高 40%。"
)


def extract(text: str, max_claims: int = 8):
    return EXTRACTOR.extract(text, max_claims=max_claims)


# ==========================================================================
# 位置精确性
# ==========================================================================
def test_every_claim_span_points_to_the_real_text() -> None:
    result = extract(FAKE_DOI_TEXT)
    assert result.claims, "这段文本应当能拆出主张"
    for claim in result.claims:
        assert claim.source_span.is_inside(FAKE_DOI_TEXT), (
            f"{claim.claim_id} 的片段 {claim.source_span.text!r} 与原文位置不一致"
        )
        if claim.parent_sentence_span is not None:
            assert claim.source_span.relative_to(claim.parent_sentence_span) is not None, (
                f"{claim.claim_id} 的片段不在它所属的句子内"
            )


def test_claim_ids_are_sequential_and_unique() -> None:
    result = extract(FAKE_DOI_TEXT)
    ids = [claim.claim_id for claim in result.claims]
    assert ids == [f"C{index:03d}" for index in range(1, len(ids) + 1)]
    assert len(set(ids)) == len(ids)


# ==========================================================================
# 类型与内容
# ==========================================================================
def test_fake_doi_text_yields_expected_claim_types() -> None:
    result = extract(FAKE_DOI_TEXT)
    types = [claim.claim_type for claim in result.claims]
    assert ClaimType.DOI_VALIDITY in types, "含 DOI 时应当产出 DOI 真实性主张"
    assert ClaimType.METADATA_YEAR in types, "含年份时应当产出年份主张"
    assert ClaimType.QUANTITATIVE in types, "含具体数值时应当产出数值主张"


def test_year_claim_is_not_duplicated() -> None:
    """回归用例：'2023 年' 曾经被同时拆成 '2023 年' 和 '2023' 两条主张。"""
    result = extract(FAKE_DOI_TEXT)
    year_claims = [claim for claim in result.claims if claim.claim_type is ClaimType.METADATA_YEAR]
    assert len(year_claims) == 1
    assert year_claims[0].source_span.text.strip() == "2023 年"


def test_claim_priority_puts_doi_and_numbers_first() -> None:
    result = extract(FAKE_DOI_TEXT)
    assert result.claims[0].priority is ClaimPriority.P0
    assert all(
        claim.priority is ClaimPriority.P0
        for claim in result.claims
        if claim.claim_type in (ClaimType.DOI_VALIDITY, ClaimType.QUANTITATIVE, ClaimType.METADATA_YEAR)
    )


def test_doi_claim_carries_actionable_questions() -> None:
    result = extract(FAKE_DOI_TEXT)
    doi_claim = next(
        claim for claim in result.claims if claim.claim_type is ClaimType.DOI_VALIDITY
    )
    assert len(doi_claim.verification_questions) >= 2
    assert any("DOI" in question for question in doi_claim.verification_questions)


def test_invalid_doi_claim_has_high_confidence() -> None:
    result = extract(FAKE_DOI_TEXT)
    doi_claim = next(
        claim for claim in result.claims if claim.claim_type is ClaimType.DOI_VALIDITY
    )
    assert doi_claim.extraction_confidence is ExtractionConfidence.HIGH
    assert "格式" in doi_claim.normalized_claim


# ==========================================================================
# 不注水：模糊引用只出一条
# ==========================================================================
def test_vague_authority_sentence_does_not_explode_into_many_claims() -> None:
    """回归用例：'多项研究表明，这一进程不可逆转' 曾一次产出 3 条同义主张。"""
    result = extract("多项研究表明，这一进程不可逆转。")
    assert len(result.claims) == 1, f"期望 1 条主张，实际得到 {len(result.claims)} 条"
    claim = result.claims[0]
    assert claim.claim_type is ClaimType.CAUSAL, "同时含因果措辞时按因果类处理"
    assert "多项研究" in claim.normalized_claim


def test_vague_authority_without_causal_word_is_general_fact() -> None:
    result = extract("据统计，该行业的规模已经翻倍。")
    assert len(result.claims) == 1
    assert result.claims[0].claim_type is ClaimType.GENERAL_FACT


# ==========================================================================
# 观点类
# ==========================================================================
def test_opinion_claim_is_marked_not_checkable() -> None:
    result = extract("业内专家普遍认为，到 2030 年通用人工智能将全面超越人类水平。")
    opinions = [claim for claim in result.claims if claim.claim_type is ClaimType.OPINION]
    assert opinions, "预测性表述应当被识别为观点类主张"
    assert all(claim.checkability is Checkability.NOT_CHECKABLE for claim in opinions)
    assert all(claim.verification_questions == [] for claim in opinions)


# ==========================================================================
# 截断与空文本
# ==========================================================================
def test_max_claims_truncates_and_reports_it() -> None:
    result = extract(FAKE_DOI_TEXT, max_claims=1)
    assert len(result.claims) == 1
    assert result.truncated is True
    assert result.total_detected >= 3


def test_text_without_checkable_claims_returns_empty_with_note() -> None:
    result = extract("你好，今天天气不错。")
    assert result.claims == []
    assert result.notes, "拆不出主张时必须留下说明，而不是静默返回空"

    result_long = extract("嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯，好的好的好的好的好的。")
    assert result_long.claims == []


# ==========================================================================
# 定位失败的草稿会被丢弃（而不是硬凑位置）
# ==========================================================================
def test_draft_with_quote_not_in_text_is_dropped() -> None:
    text = "该研究的结论已经发表。"
    extraction = ClaimExtraction(
        claims=[
            ClaimDraft(
                quote="这句话根本不在原文里",
                normalized_claim="伪造的主张",
                claim_type=ClaimType.GENERAL_FACT,
                verification_questions=["是否真实？"],
            )
        ]
    )
    result = drafts_to_claims(text, extraction, max_claims=8)
    assert result.claims == []
    assert result.dropped_unlocatable == 1
    assert any("无法在原文中精确定位" in note for note in result.notes)


def test_span_hints_keep_duplicate_quotes_distinguishable() -> None:
    """原文里出现两次同样的片段时，规则引擎给出的位置提示必须能区分它们。"""
    text = "第一批：准确率 95%。第二批：准确率 95%。"
    drafts, hints, _ = collect_drafts(text)
    assert drafts and len(drafts) == len(hints)
    starts = {hint.start for hint in hints}
    assert len(starts) == len(hints), "每个候选主张都应当有自己的位置"
