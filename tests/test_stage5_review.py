"""阶段 5 测试：独立复核（确定性检查 + 与复核员结论的合并）。

这里最重要的一条：**复核没通过（或者根本没复核）的修正，绝不允许写进正文**。
"改错比不改更糟"——这条规则靠代码保证，不靠模型自觉。
"""

from __future__ import annotations

import pytest

from verimind.core.enums import (
    ChangeType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    JudgedAspect,
    ReviewCheckType,
    ReviewOverall,
    RoleStance,
)
from verimind.core.schemas import Correction, Evidence, ReviewDraft, Span
from verimind.review.independent_review import (
    accepted_for_application,
    combine_review,
    deterministic_checks,
    extract_facts,
)

TEXT = "某研究（DOI: 10.1038/nature12373）发表于 2008 年，显示准确率提高了 40%。"


def make_evidence(
    *, evidence_id: str = "E001", rationale: str, snippet: str | None = None, year: int | None = 2013
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        claim_ids=["C001"],
        source=EvidenceSource.CROSSREF,
        level=EvidenceLevel.METADATA_ONLY,
        url="https://api.crossref.org/works/10.1038/nature12373",
        title="Certified perovskite solar cells",
        year=year,
        relation=EvidenceRelation.REFUTES,
        judged_aspect=JudgedAspect.BIBLIOGRAPHIC,
        relation_rationale=rationale,
        snippet=snippet,
    )


def make_correction(
    *, proposed_text: str, change_type: ChangeType = ChangeType.FIX_NUMBER, evidence_ids=None
) -> Correction:
    start = TEXT.index("2008 年")
    return Correction(
        correction_id="K001",
        claim_id="C001",
        original_span=Span.from_text(TEXT, start, start + 4),
        proposed_text=proposed_text,
        change_type=change_type,
        rationale="测试用",
        evidence_ids=evidence_ids if evidence_ids is not None else ["E001"],
    )


# ==========================================================================
# 事实抽取
# ==========================================================================
def test_extract_facts_picks_years_numbers_and_dois() -> None:
    facts = extract_facts("该研究（DOI: 10.1038/nature12373）于 2013 年把效率提高到 12%，提升 3 倍。")
    assert "2013" in facts
    assert "12%" in facts
    assert "3倍" in facts
    assert "10.1038/nature12373" in facts


def test_extract_facts_ignores_plain_words() -> None:
    assert extract_facts("这是一句没有任何数字的话。") == set()


# ==========================================================================
# 确定性检查
# ==========================================================================
def test_year_fix_backed_by_evidence_passes_all_checks() -> None:
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年，而原文写的是 2008 年。")
    correction = make_correction(proposed_text="2013 年")
    checks = {item.check: item for item in deterministic_checks(correction, [evidence])}
    assert all(item.passed for item in checks.values())
    assert checks[ReviewCheckType.EVIDENCE_MATCHES_TEXT].passed


def test_invented_number_fails_evidence_match() -> None:
    """修正里出现证据中没有的数字 → 必须被拦住。"""
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="25%")
    checks = {item.check: item for item in deterministic_checks(correction, [evidence])}
    assert checks[ReviewCheckType.EVIDENCE_MATCHES_TEXT].passed is False
    assert checks[ReviewCheckType.NO_NEW_FACT_ERROR].passed is False
    assert any("25%" in item.detail for item in checks.values())


def test_number_taken_from_a_different_paper_is_caught() -> None:
    """数字确实存在于"某条证据"里，但不是被引用的那条 → 与所引证据不匹配。"""
    cited = make_evidence(evidence_id="E001", rationale="Crossref 记录显示该文献发表于 2013 年。")
    other = make_evidence(
        evidence_id="E002",
        rationale="另一篇文献报告了 25% 的提升。",
        snippet="efficiency improved by 25%",
    )
    correction = make_correction(proposed_text="25%", evidence_ids=["E001"])
    checks = {item.check: item for item in deterministic_checks(correction, [cited, other])}
    assert checks[ReviewCheckType.EVIDENCE_MATCHES_TEXT].passed is False, (
        "所引证据里没有 25%，必须判不通过"
    )
    assert checks[ReviewCheckType.NO_NEW_FACT_ERROR].passed is True, (
        "但它确实来自另一条证据，所以不算凭空编造——两条检查的语义不同"
    )


def test_fact_change_without_evidence_fails_backed_check() -> None:
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = Correction(
        correction_id="K001",
        claim_id="C001",
        original_span=Span.from_text(TEXT, TEXT.index("2008 年"), TEXT.index("2008 年") + 4),
        proposed_text="2013 年",
        change_type=ChangeType.WEAKEN,
        rationale="测试用",
        evidence_ids=[],
    )
    checks = {item.check: item for item in deterministic_checks(correction, [evidence])}
    # 弱化类修改不写入新事实，因此"是否有证据"这一项直接通过
    assert checks[ReviewCheckType.CORRECTION_EVIDENCE_BACKED].passed is True
    assert "无需外部证据" in checks[ReviewCheckType.CORRECTION_EVIDENCE_BACKED].detail


# ==========================================================================
# 合并复核结论
# ==========================================================================
def test_combine_review_passes_when_everything_is_fine() -> None:
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="2013 年")
    draft = ReviewDraft(
        target_correction_id="K001",
        issue_resolved=True,
        issue_resolved_detail="年份已改为证据中的 2013 年。",
        no_new_overreach=True,
        no_new_overreach_detail="没有新增论断。",
    )
    review = combine_review(
        claim=None, correction=correction, evidence_items=[evidence],
        review_id="V001", draft=draft,
    )
    assert review.overall is ReviewOverall.PASSED
    assert len(review.checks) == 5
    assert accepted_for_application(review) is True


def test_combine_review_marks_risks_when_unresolved_is_not_empty() -> None:
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="2013 年")
    draft = ReviewDraft(
        target_correction_id="K001",
        issue_resolved=True,
        issue_resolved_detail="已改正。",
        no_new_overreach=True,
        no_new_overreach_detail="无新增论断。",
        unresolved=["建议人工确认原文引用的确实是同一篇文献。"],
    )
    review = combine_review(
        claim=None, correction=correction, evidence_items=[evidence],
        review_id="V001", draft=draft,
    )
    assert review.overall is ReviewOverall.PASSED_WITH_RISKS
    assert accepted_for_application(review) is True


def test_combine_review_fails_when_deterministic_check_fails() -> None:
    """模型说"通过"，但代码发现编造的数字 → 必须以代码为准。"""
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="25%")
    draft = ReviewDraft(
        target_correction_id="K001",
        issue_resolved=True,
        issue_resolved_detail="模型认为已解决。",
        no_new_overreach=True,
        no_new_overreach_detail="模型认为没有过度推断。",
    )
    review = combine_review(
        claim=None, correction=correction, evidence_items=[evidence],
        review_id="V001", draft=draft,
    )
    assert review.overall is ReviewOverall.FAILED, "确定性检查是否决项，模型不能推翻"
    assert not accepted_for_application(review)
    assert review.unresolved, "未通过时必须列出未解决的问题"


def test_combine_review_fails_when_review_did_not_run() -> None:
    """根本没有复核结论 → 一律不通过（未复核的改动不许进正文）。"""
    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="2013 年")
    review = combine_review(
        claim=None,
        correction=correction,
        evidence_items=[evidence],
        review_id="V001",
        draft=None,
        failure_reason="独立复核未能完成（模型调用失败）。",
    )
    assert review.overall is ReviewOverall.FAILED
    assert not accepted_for_application(review)
    assert any("未完成" in item or "无法确认" in item for item in review.unresolved)
    # 没有模型复核时，复核身份如实记为"规则校验"，不冒用某个模型的名义
    assert review.reviewer is not None
    assert review.reviewer.model == "deterministic"
    assert review.reviewer.stance is RoleStance.NOT_APPLICABLE


def test_combine_review_records_the_reviewer_identity() -> None:
    from verimind.core.enums import RoleName
    from verimind.core.schemas import RolePosition

    evidence = make_evidence(rationale="Crossref 记录显示该文献发表于 2013 年。")
    correction = make_correction(proposed_text="2013 年")
    draft = ReviewDraft(
        target_correction_id="K001",
        issue_resolved=True,
        issue_resolved_detail="已改正。",
        no_new_overreach=True,
        no_new_overreach_detail="无新增论断。",
    )
    reviewer = RolePosition(
        role=RoleName.INDEPENDENT_REVIEWER,
        provider="deepseek",
        model="deepseek-chat",
        stance=RoleStance.SUPPORT,
        conclusion="通过。",
    )
    review = combine_review(
        claim=None, correction=correction, evidence_items=[evidence],
        review_id="V001", draft=draft, reviewer=reviewer,
    )
    assert review.reviewer is not None and review.reviewer.provider == "deepseek"
