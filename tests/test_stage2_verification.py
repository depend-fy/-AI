"""阶段 2 测试：文内一致性检查 + 规则裁决。

这是本项目"可解释性"的核心，所以测得更细一些。要钉死的铁律：
    · 阶段 2 没有外部检索，因此**绝不允许**出现「已核实」或「存在矛盾」；
    · 说「证据不足」时必须同时给出"查过什么/没查"和"还缺什么"；
    · 规则命中的每条证据 ID 都必须在报告里真实存在（不能是占位符）。
"""

from __future__ import annotations

import pytest

from verimind.claims.extractor import RuleBasedExtractor
from verimind.claims.preprocess import analyze_text
from verimind.core.enums import (
    ProblemType,
    SearchStatus,
    Severity,
    VerificationStatus,
)
from verimind.core.schemas import Claim, Evidence, VerificationResult
from verimind.verification.consistency import (
    conflicts_by_claim,
    find_internal_conflicts,
)
from verimind.verification.rules import (
    AdjudicationContext,
    evaluate_claim,
    internal_evidence_for_conflicts,
)

EXTRACTOR = RuleBasedExtractor()

CONFLICT_TEXT = (
    "该研究于 2023 年发表在《自然》杂志（DOI: 10.1038/nature12373），"
    "实验显示模型准确率提高了 40%。"
    "同一篇 2019 年的论文（DOI: 10.1038/nature12373）进一步指出，准确率提升了 12%。"
)


def build(text: str, max_claims: int = 8) -> tuple[list[Claim], list[Evidence], AdjudicationContext]:
    """按流水线的顺序装配：拆解 → 文内检查 → 证据 → 裁决上下文。"""
    claims = EXTRACTOR.extract(text, max_claims=max_claims).claims
    analysis = analyze_text(text)
    conflicts = find_internal_conflicts(text, claims, analysis=analysis)
    evidence, ids_by_claim = internal_evidence_for_conflicts(conflicts)
    context = AdjudicationContext(
        text=text,
        analysis=analysis,
        conflicts_by_claim=conflicts_by_claim(conflicts),
        evidence_ids_by_claim=ids_by_claim,
    )
    return claims, evidence, context


def evaluate(text: str, max_claims: int = 8) -> tuple[list[Claim], list[Evidence], list[VerificationResult]]:
    claims, evidence, context = build(text, max_claims)
    return claims, evidence, [evaluate_claim(claim, context) for claim in claims]


def result_of(results: list[VerificationResult], claim: Claim) -> VerificationResult:
    return next(item for item in results if item.claim_id == claim.claim_id)


# ==========================================================================
# 文内一致性检查
# ==========================================================================
def test_same_doi_with_two_years_is_a_conflict() -> None:
    claims, evidence, context = build(CONFLICT_TEXT)
    conflicts = [c for group in context.conflicts_by_claim.values() for c in group]
    kinds = {conflict.kind for conflict in conflicts}
    assert "doi_year" in kinds
    assert "metric_value" in kinds


def test_conflict_fragments_are_the_two_conflicting_years() -> None:
    _, evidence, _ = build(CONFLICT_TEXT)
    snippets = {item.snippet for item in evidence if item.source.value == "原文内部"}
    assert {"2023 年", "2019 年"} <= snippets

    # 数值冲突应当是 40% 与 12%
    assert {"40%", "12%"} <= snippets


def test_same_doi_same_year_is_not_a_conflict() -> None:
    text = "该研究于 2023 年发表（DOI: 10.1038/nature12373）。另一处也提到这篇 2023 年的文献。"
    claims, _, context = build(text)
    assert context.conflicts_by_claim == {}


def test_different_metrics_are_not_a_conflict() -> None:
    text = "该研究显示准确率提高到 95%，同时误差降低到 3%。"
    claims, _, context = build(text)
    assert context.conflicts_by_claim == {}


def test_internal_evidence_needs_no_url_but_needs_a_span() -> None:
    _, evidence, _ = build(CONFLICT_TEXT)
    assert evidence, "文内矛盾应当产出证据"
    for item in evidence:
        assert item.is_internal
        assert item.url is None, "原文内部证据不应该有（也不允许伪造）外部链接"
        assert item.source_span is not None
        assert item.snippet


# ==========================================================================
# 铁律：阶段 2 不可能给出"已核实 / 存在矛盾"
# ==========================================================================
@pytest.mark.parametrize(
    "text",
    [
        "据一项发表于 2023 年的研究（DOI: 10.xxxx/verimind.2023.001）显示，准确率提高 40%。",
        "《自然》杂志在 2008 年发表的一项研究（DOI: 10.1038/nature12373）证实，转换效率突破 50%。",
        "业内专家普遍认为，到 2030 年通用人工智能将全面超越人类水平。多项研究表明，这一进程不可逆转。",
        CONFLICT_TEXT,
    ],
)
def test_stage2_never_claims_verified_or_contradicted(text: str) -> None:
    _, _, results = evaluate(text)
    forbidden = {
        VerificationStatus.VERIFIED,
        VerificationStatus.CONTRADICTED,
        VerificationStatus.PARTIALLY_SUPPORTED,
    }
    assert all(result.status not in forbidden for result in results), (
        "阶段 2 未接入外部检索，不可能得出这些结论——出现即意味着在编造依据"
    )


# ==========================================================================
# 逐条规则
# ==========================================================================
def test_plain_unverified_claim_gets_no_problem_type() -> None:
    """只是"还没查"，不该被扣上任何问题类型。"""
    text = "该研究于 2023 年发表（DOI: 10.1038/nature12373）。"
    claims, _, results = evaluate(text)
    year_claim = next(c for c in claims if c.claim_type.value == "年份类")
    result = result_of(results, year_claim)
    assert result.status is VerificationStatus.INSUFFICIENT
    assert result.problem_types == []
    assert result.severity is Severity.NONE
    assert result.adjudication_rule == "N1"


def test_invalid_doi_rule_n5() -> None:
    text = "相关研究（DOI: 10.xxxx/example.001）表明结论成立。"
    claims, _, results = evaluate(text)
    doi_claim = next(c for c in claims if c.claim_type.value == "DOI 是否真实")
    result = result_of(results, doi_claim)
    assert "N5" in result.adjudication_rule
    assert ProblemType.CITATION_MISSING in result.problem_types
    assert result.severity is Severity.MEDIUM
    assert result.missing_evidence, "指出疑点的同时必须说明还需要什么证据来证实"


def test_number_without_source_rule_n2() -> None:
    text = "该方法的准确率提高了 40%。"
    claims, _, results = evaluate(text)
    number_claim = next(c for c in claims if c.claim_type.value == "数值/统计类")
    result = result_of(results, number_claim)
    assert "N2" in result.adjudication_rule
    assert ProblemType.MISSING_EVIDENCE in result.problem_types


def test_number_with_concrete_source_does_not_trigger_n2() -> None:
    text = "该研究（DOI: 10.1038/nature12373）显示准确率提高了 40%。"
    claims, _, results = evaluate(text)
    number_claim = next(c for c in claims if c.claim_type.value == "数值/统计类")
    result = result_of(results, number_claim)
    assert result.problem_types == [], "有明确出处时不应判为缺少证据"
    assert result.adjudication_rule == "N1"


def test_causal_overreach_rule_n3() -> None:
    text = "该团队的工作证实了这种方法可以彻底解决该问题。"
    claims, _, results = evaluate(text)
    causal_claim = next(c for c in claims if c.claim_type.value == "因果/机制类")
    result = result_of(results, causal_claim)
    assert "N3" in result.adjudication_rule
    assert ProblemType.OVERREACH in result.problem_types


def test_vague_authority_rule_n4() -> None:
    text = "多项研究表明，这一进程不可逆转。"
    claims, _, results = evaluate(text)
    result = results[0]
    assert "N4" in result.adjudication_rule
    assert ProblemType.MISSING_EVIDENCE in result.problem_types


def test_opinion_rule_u1() -> None:
    text = "预计到 2035 年，该技术将全面取代人工。"
    claims, _, results = evaluate(text)
    opinion = next(c for c in claims if c.claim_type.value == "观点/预测类")
    result = result_of(results, opinion)
    assert result.status is VerificationStatus.UNVERIFIABLE
    assert result.adjudication_rule == "U1"
    assert result.problem_types == [], "观点不该被判成事实错误"
    assert result.severity is Severity.NONE


# ==========================================================================
# 证据不足的时候必须交代清楚
# ==========================================================================
def test_insufficient_results_always_explain_what_is_missing() -> None:
    text = "该研究于 2023 年发表（DOI: 10.1038/nature12373），准确率提高 40%。"
    _, _, results = evaluate(text)
    for result in results:
        if result.status is VerificationStatus.INSUFFICIENT:
            assert result.missing_evidence, f"{result.claim_id} 说了证据不足，却没说什么缺"
            assert result.search_trace.records, f"{result.claim_id} 没有检索痕迹"


def test_stage2_search_trace_is_honestly_marked_as_skipped() -> None:
    """没有检索就必须写成「未检索」，且不能声称查过某个来源。"""
    text = "该研究于 2023 年发表（DOI: 10.1038/nature12373）。"
    _, _, results = evaluate(text)
    for result in results:
        for record in result.search_trace.records:
            assert record.status is SearchStatus.SKIPPED
            assert record.source is None, "没查过就不能写检索来源"
            assert "未接入外部文献检索" in record.detail


# ==========================================================================
# 冲突结论必须挂着真实存在的证据
# ==========================================================================
def test_source_conflict_references_real_evidence_ids() -> None:
    claims, evidence, results = evaluate(CONFLICT_TEXT)
    known_ids = {item.evidence_id for item in evidence}
    conflicts = [result for result in results if result.status is VerificationStatus.SOURCE_CONFLICT]
    assert conflicts, "文内矛盾应当产生来源冲突结论"
    for result in conflicts:
        assert result.refuting_evidence_ids, "来源冲突必须挂证据"
        assert set(result.refuting_evidence_ids) <= known_ids, "引用的证据必须真实存在"
    assert {result.adjudication_rule for result in conflicts} <= {"I1", "I2", "I1、I2"}
    assert all(result.severity is Severity.HIGH for result in conflicts)
