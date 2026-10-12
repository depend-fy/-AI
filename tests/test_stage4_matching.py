"""阶段 4 测试：证据与主张的匹配（确定性部分）。

"哪条证据能判断哪个层面"是本项目最容易出错、也最需要守住的边界：
    · 元数据 → 只能判断"文献是否存在""年份/期刊是否一致"；
    · 摘要/全文 → 才能判断"内容结论是否被支持"。

这里的测试同时也验证了"角色 B 编造引用会被丢弃"这条防幻觉机制。
"""

from __future__ import annotations

import pytest

from verimind.claims.extractor import RuleBasedExtractor
from verimind.claims.preprocess import analyze_text
from verimind.core.enums import (
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    JudgedAspect,
    MatchBasis,
)
from verimind.core.schemas import Claim, Evidence, Span
from verimind.verification.matcher import (
    apply_content_assessment,
    assess_metadata,
    assess_title_match,
    content_pairs,
    snippet_in_text,
)

EXTRACTOR = RuleBasedExtractor()

TEXT = "某研究（DOI: 10.1038/nature12373）发表于 2008 年，显示准确率提高了 40%。"
ABSTRACT = "We report a certified power conversion efficiency of 12% in perovskite solar cells."


def make_claim(claim_type: ClaimType, quote: str) -> Claim:
    start = TEXT.index(quote)
    return Claim(
        claim_id="C001",
        source_span=Span.from_text(TEXT, start, start + len(quote)),
        parent_sentence_span=Span.from_text(TEXT, 0, len(TEXT)),
        normalized_claim="（测试用）",
        claim_type=claim_type,
        verification_questions=["是否可核验？"],
    )


def make_evidence(**overrides) -> Evidence:
    data = {
        "evidence_id": "E001",
        "claim_ids": ["C001"],
        "source": EvidenceSource.CROSSREF,
        "level": EvidenceLevel.METADATA_ONLY,
        "url": "https://api.crossref.org/works/10.1038/nature12373",
        "title": "Certified perovskite solar cells",
        "doi": "10.1038/nature12373",
        "year": 2013,
        "venue": "Nature",
        "query": "10.1038/nature12373",
    }
    data.update(overrides)
    return Evidence(**data)  # type: ignore[arg-type]


# ==========================================================================
# 文献是否存在
# ==========================================================================
def test_existence_claim_gets_support_from_metadata() -> None:
    analysis = analyze_text(TEXT)
    claim = make_claim(ClaimType.DOI_VALIDITY, "DOI: 10.1038/nature12373")
    revised = assess_metadata(claim, make_evidence(), analysis)
    assert revised.relation is EvidenceRelation.SUPPORTS
    assert revised.judged_aspect is JudgedAspect.EXISTENCE
    assert revised.match_basis is MatchBasis.METADATA
    assert "文献真实存在" in revised.relation_rationale
    assert "不代表其内容支持" in revised.relation_rationale, "必须提醒：存在≠内容被支持"


# ==========================================================================
# 年份
# ==========================================================================
def test_year_mismatch_is_refuted_by_metadata() -> None:
    analysis = analyze_text(TEXT)
    claim = make_claim(ClaimType.METADATA_YEAR, "2008 年")
    revised = assess_metadata(claim, make_evidence(year=2013), analysis)
    assert revised.relation is EvidenceRelation.REFUTES
    assert revised.judged_aspect is JudgedAspect.BIBLIOGRAPHIC
    assert "2013" in revised.relation_rationale and "2008" in revised.relation_rationale


def test_year_match_is_supported() -> None:
    analysis = analyze_text(TEXT)
    claim = make_claim(ClaimType.METADATA_YEAR, "2008 年")
    revised = assess_metadata(claim, make_evidence(year=2008), analysis)
    assert revised.relation is EvidenceRelation.SUPPORTS
    assert revised.judged_aspect is JudgedAspect.BIBLIOGRAPHIC


def test_year_is_left_unjudged_when_evidence_has_no_year() -> None:
    analysis = analyze_text(TEXT)
    claim = make_claim(ClaimType.METADATA_YEAR, "2008 年")
    revised = assess_metadata(claim, make_evidence(year=None), analysis)
    assert revised.relation is EvidenceRelation.UNKNOWN
    assert revised.judged_aspect is None


# ==========================================================================
# 内容类主张：元数据一概不作判断
# ==========================================================================
def test_metadata_evidence_never_judges_content_claims() -> None:
    """数值类主张拿到的是一篇真实存在的论文的元数据 —— 不能据此说"论文支持了 40%"。"""
    analysis = analyze_text(TEXT)
    claim = make_claim(ClaimType.QUANTITATIVE, "40%")
    revised = assess_metadata(claim, make_evidence(), analysis)
    assert revised.relation is EvidenceRelation.UNKNOWN, (
        "只有元数据时绝不能对内容层面下任何结论"
    )
    assert revised.judged_aspect is None


def test_content_pairs_only_accepts_text_bearing_evidence() -> None:
    metadata_only = make_evidence()
    abstract_evidence = make_evidence(
        evidence_id="E002",
        level=EvidenceLevel.ABSTRACT,
        snippet=ABSTRACT,
    )
    assert content_pairs([metadata_only]) == []
    assert [item.evidence_id for item in content_pairs([metadata_only, abstract_evidence])] == [
        "E002"
    ]


# ==========================================================================
# 标题匹配（防止张冠李戴）
# ==========================================================================
def test_title_match_rejects_unrelated_search_results() -> None:
    evidence = make_evidence(doi=None, title="A totally different study about batteries")
    assert assess_title_match(evidence, "Certified perovskite solar cells") is None


def test_title_match_accepts_similar_titles() -> None:
    evidence = make_evidence(doi=None, title="Certified Perovskite Solar Cells")
    assert assess_title_match(evidence, "Certified perovskite solar cells") is not None


def test_title_match_skipped_when_doi_present() -> None:
    evidence = make_evidence(title="完全不同的标题")
    assert assess_title_match(evidence, "另一个查询") is not None, "有 DOI 时不必做标题匹配"


# ==========================================================================
# 角色 B 的判断落地与防幻觉
# ==========================================================================
def test_content_assessment_is_applied_when_quote_matches() -> None:
    evidence = make_evidence(level=EvidenceLevel.ABSTRACT, snippet=ABSTRACT)
    revised = apply_content_assessment(
        evidence,
        relation=EvidenceRelation.REFUTES,
        rationale="摘要给出的效率是 12%，与原文的 40% 不符。",
        quoted_snippet="a certified power conversion efficiency of 12%",
    )
    assert revised is not None
    assert revised.relation is EvidenceRelation.REFUTES
    assert revised.judged_aspect is JudgedAspect.CONTENT
    assert revised.match_basis is MatchBasis.ABSTRACT


def test_content_assessment_without_quote_is_rejected() -> None:
    evidence = make_evidence(level=EvidenceLevel.ABSTRACT, snippet=ABSTRACT)
    assert (
        apply_content_assessment(
            evidence,
            relation=EvidenceRelation.REFUTES,
            rationale="摘要与原文不符。",
            quoted_snippet=None,
        )
        is None
    ), "没有引用原句的反驳不可采纳"


def test_content_assessment_on_metadata_only_is_rejected() -> None:
    evidence = make_evidence()  # 只有元数据
    assert (
        apply_content_assessment(
            evidence,
            relation=EvidenceRelation.SUPPORTS,
            rationale="看起来支持。",
            quoted_snippet="some quote",
        )
        is None
    ), "没有摘要/全文时不允许承载内容层面的判断"


def test_snippet_in_text_detects_fabricated_quotes() -> None:
    assert snippet_in_text("certified power conversion efficiency of 12%", ABSTRACT)
    assert snippet_in_text("CERTIFIED   POWER conversion efficiency of 12%", ABSTRACT), (
        "大小写与多余空格不应影响判断"
    )
    assert not snippet_in_text("We achieved 99% efficiency", ABSTRACT), (
        "摘要里没有的句子必须被识别为编造"
    )
    assert not snippet_in_text("", ABSTRACT)
    assert not snippet_in_text("abc", "")


@pytest.mark.parametrize(
    "relation", [EvidenceRelation.SUPPORTS, EvidenceRelation.PARTIALLY_SUPPORTS, EvidenceRelation.REFUTES]
)
def test_every_judged_relation_requires_an_aspect(relation: EvidenceRelation) -> None:
    with pytest.raises(Exception):
        make_evidence(level=EvidenceLevel.ABSTRACT, relation=relation, relation_rationale="x")
