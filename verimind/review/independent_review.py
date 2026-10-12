"""修正后的独立复核：五项检查 + "没通过就不许应用"的硬约束。

为什么要把五项检查拆成"确定性"与"语义"两半？
    · 「是否引入新事实错误」「证据与修正后表述是否匹配」这类问题，
      代码判定**比模型更可靠也更便宜**：把修正后出现的新年份/新数值/新 DOI 抽出来，
      逐个回到所引证据里核对即可。模型一旦"通顺地改错"，这里就会立刻抓住。
    · 「原问题是否解决」「是否引入新的过度推断」需要读懂语义，交给独立复核员（角色 D）。

最后一条硬约束：**复核没通过的修正一律不应用到报告里**。
没有复核结论（例如模型调用失败）时，同样视为不通过——未复核的改动不许进正文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from verimind.core.enums import (
    FACT_CHANGING_TYPES,
    ReviewCheckType,
    ReviewOverall,
    RoleName,
    RoleStance,
)
from verimind.core.schemas import (
    Claim,
    Correction,
    Evidence,
    ReviewCheck,
    ReviewDraft,
    ReviewResult,
    RolePosition,
)

#: 会被当作"事实"来核对的 token：DOI、年份、百分比、倍数/规模
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"10\.\d{4,9}/[^\s，。；、）)】」》]+"),
    re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)"),
    re.compile(r"\d+(?:\.\d+)?\s*[%％]"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:倍|万|亿|个百分点)"),
)


def _normalize_fact(token: str) -> str:
    return re.sub(r"\s+", "", token).replace("％", "%")


def extract_facts(text: str) -> set[str]:
    """抽取文本里"可核对的事实 token"（年份、数值、DOI）。"""
    facts: set[str] = set()
    for pattern in _FACT_PATTERNS:
        for match in pattern.finditer(text or ""):
            facts.add(_normalize_fact(match.group(0)))
    return facts


def _evidence_text(items: list[Evidence]) -> str:
    """把证据里所有"可核对的内容"拼起来：年份、DOI、标题、片段、判断理由。"""
    parts: list[str] = []
    for item in items:
        parts.extend(
            [
                str(item.year or ""),
                item.doi or "",
                item.title or "",
                item.snippet or "",
                item.relation_rationale or "",
            ]
        )
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class DeterministicCheck:
    check: ReviewCheckType
    passed: bool
    detail: str


def deterministic_checks(
    correction: Correction, evidence_items: list[Evidence]
) -> list[DeterministicCheck]:
    """三项可以用代码判定的检查。"""
    cited = [item for item in evidence_items if item.evidence_id in correction.evidence_ids]
    cited_facts = extract_facts(_evidence_text(cited))
    all_facts = extract_facts(_evidence_text(evidence_items))

    original_facts = extract_facts(correction.original_span.text)
    proposed_facts = extract_facts(correction.proposed_text)
    new_facts = proposed_facts - original_facts

    # ① 是否有证据支撑（改事实类才需要）
    if correction.change_type in FACT_CHANGING_TYPES:
        backed = bool(correction.evidence_ids) and bool(cited)
        detail = (
            f"该修改会写入新的事实内容，已引用 {len(cited)} 条证据："
            f"{'、'.join(item.evidence_id for item in cited)}。"
            if backed
            else "该修改会写入新的事实内容，但没有可用的证据支撑。"
        )
    else:
        backed = True
        detail = f"该修改属于「{correction.change_type.value}」，不写入新事实，无需外部证据。"
    check_backed = DeterministicCheck(ReviewCheckType.CORRECTION_EVIDENCE_BACKED, backed, detail)

    # ② 修正后出现的新事实，必须能在**所引证据**里找到
    unmatched = sorted(fact for fact in new_facts if fact not in cited_facts)
    check_match = DeterministicCheck(
        ReviewCheckType.EVIDENCE_MATCHES_TEXT,
        not unmatched,
        (
            "修正后出现的新事实（年份 / 数值 / DOI）都能在所引证据中找到对应内容。"
            if not unmatched
            else "修正后出现了这些新事实：" + "、".join(unmatched) + "，但在所引证据里找不到对应内容。"
        ),
    )

    # ③ 不许凭空编造：新事实至少要能在"全部证据"里找到依据
    invented = sorted(fact for fact in new_facts if fact not in all_facts)
    check_no_new = DeterministicCheck(
        ReviewCheckType.NO_NEW_FACT_ERROR,
        not invented,
        (
            "没有引入原文与全部证据之外的新事实。"
            if not invented
            else "修正引入了原文和所有证据中都不存在的内容："
            + "、".join(invented)
            + "——这属于凭空编造，必须拒绝。"
        ),
    )

    return [check_backed, check_match, check_no_new]


def combine_review(
    *,
    claim: Claim | None,
    correction: Correction,
    evidence_items: list[Evidence],
    review_id: str,
    draft: ReviewDraft | None,
    reviewer: RolePosition | None = None,
    failure_reason: str | None = None,
) -> ReviewResult:
    """把确定性检查与复核员的语义结论合并成完整的复核结果。"""
    checks = [
        ReviewCheck(check=item.check, passed=item.passed, detail=item.detail)
        for item in deterministic_checks(correction, evidence_items)
    ]

    if draft is None:
        # 没有复核结论（模型调用失败等）→ 两个语义检查直接判不通过。
        # 这是刻意的：**未复核的修正不许进正文**。
        reason = failure_reason or "独立复核未能完成。"
        checks.append(
            ReviewCheck(
                check=ReviewCheckType.ORIGINAL_ISSUE_RESOLVED,
                passed=False,
                detail=f"{reason}无法确认原始问题是否已解决，因此该修正不予采用。",
            )
        )
        checks.append(
            ReviewCheck(
                check=ReviewCheckType.NO_NEW_OVERREACH,
                passed=False,
                detail=f"{reason}无法确认是否引入新的无依据结论，因此该修正不予采用。",
            )
        )
        unresolved = ["独立复核未完成，无法确认该修正是否安全。"]
    else:
        checks.append(
            ReviewCheck(
                check=ReviewCheckType.ORIGINAL_ISSUE_RESOLVED,
                passed=draft.issue_resolved,
                detail=draft.issue_resolved_detail,
            )
        )
        checks.append(
            ReviewCheck(
                check=ReviewCheckType.NO_NEW_OVERREACH,
                passed=draft.no_new_overreach,
                detail=draft.no_new_overreach_detail,
            )
        )
        unresolved = list(draft.unresolved)

    failed = [check for check in checks if not check.passed]
    if failed:
        overall = ReviewOverall.FAILED
        if not unresolved:
            unresolved = [f"{check.check.value}：{check.detail}" for check in failed]
    elif unresolved:
        overall = ReviewOverall.PASSED_WITH_RISKS
    else:
        overall = ReviewOverall.PASSED

    if reviewer is None:
        reviewer = RolePosition(
            role=RoleName.INDEPENDENT_REVIEWER,
            provider="规则校验",
            model="deterministic",
            stance=RoleStance.NOT_APPLICABLE,
            conclusion=f"确定性检查{'通过' if not failed else '未通过'}（共 {len(checks)} 项）。",
            reasoning="由代码完成事实核对：新出现的年份 / 数值 / DOI 必须能在所引证据中找到依据。",
        )

    return ReviewResult(
        review_id=review_id,
        target_correction_id=correction.correction_id,
        checks=checks,
        overall=overall,
        unresolved=unresolved,
        reviewer=reviewer,
    )


def accepted_for_application(review: ReviewResult) -> bool:
    """只有"通过"或"通过但有风险"的修正才允许写进修正后全文。

    刻意把 PASSED_WITH_RISKS 也算通过：它表示"证据支持、但仍有需要人工确认的点"，
    报告里会把这些点列出来，而不是因为存在风险就干脆不改。
    """
    return review.overall in (ReviewOverall.PASSED, ReviewOverall.PASSED_WITH_RISKS)


__all__ = [
    "DeterministicCheck",
    "accepted_for_application",
    "combine_review",
    "deterministic_checks",
    "extract_facts",
]
