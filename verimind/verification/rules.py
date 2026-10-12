"""规则裁决层：把"文本信号 + 证据"变成明确的核验状态与问题类型。

这是本项目最重要的设计之一：**最终结论不由大模型说了算**。
大模型（或规则引擎）只负责产出可观察的信号；
"这条主张到底算不算有问题"由下表的规则决定，因此：
    · 可解释：报告里能写出"命中了 N2 规则"；
    · 可测试：这些规则是纯函数，可以被穷举单元测试；
    · 可评测：改规则立刻能跑回归。

阶段 2 的诚实边界
----------------
阶段 2 **没有接入任何外部检索**，因此：
    · 永远不可能给出「已核实」，也永远不可能因为外部证据给出「存在矛盾」
      ——因为那需要真实检索到的来源；
    · 需要外部证据的主张一律为「证据不足」，并且必须带上检索痕迹（标记为"未检索"）
      和"还缺什么证据"的说明；
    · 唯一能给出「来源冲突」的情况是**原文自己前后矛盾**（证据来自原文内部）。
"""

from __future__ import annotations

from dataclasses import dataclass

from verimind.claims.preprocess import (
    DocumentAnalysis,
    SentenceSignals,
    is_valid_doi_prefix,
    looks_like_doi,
)
from verimind.core.enums import (
    Checkability,
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    MatchBasis,
    ProblemType,
    SearchStatus,
    Severity,
    VerificationStatus,
)
from verimind.core.schemas import (
    Claim,
    Evidence,
    SearchRecord,
    SearchTrace,
    VerificationResult,
)
from verimind.verification.consistency import InternalConflict

# ==========================================================================
# 规则说明书（报告里会原文展示命中的规则，让结论可追问）
# ==========================================================================
RULE_LIBRARY: dict[str, str] = {
    "I1": "同一篇文献（同一 DOI）在原文中被写成两个不同的年份 → 来源冲突（证据来自原文内部）",
    "I2": "同一指标在原文中出现了两个不同的数值 → 来源冲突（证据来自原文内部）",
    "U1": "观点 / 预测类内容不属于可用外部证据核验的事实主张 → 暂无法核验",
    "N1": "该主张需要外部证据才能判断，而本次未启用检索 → 证据不足（不判定为错误）",
    "N2": "原文给出了具体数值，但整句中没有任何可核验的出处 → 证据不足 + 缺少必要证据",
    "N3": "使用了「证明 / 证实 / 充分说明」等强因果表述，却没有给出具体来源 → 证据不足 + 逻辑推断过度",
    "N4": "用「大量研究表明」这类模糊表述支撑结论，但没有具体来源 → 证据不足 + 缺少必要证据",
    "N5": "DOI 格式不合法（前缀应为 4~9 位数字）→ 证据不足 + 引用不存在或元数据不匹配（疑似，未证实）",
}

#: 不同类型的主张"还缺什么证据才能判断"——这是对用户最有用的信息之一
_MISSING_EVIDENCE_BY_TYPE: dict[ClaimType, str] = {
    ClaimType.DOI_VALIDITY: "需要通过 Crossref 查询该 DOI 的注册信息，并核对其标题、作者、年份是否与原文一致（阶段 4 实现）。",
    ClaimType.WORK_EXISTENCE: "需要通过标题 / 作者 / 期刊在 Crossref 或 OpenAlex 中检索该文献是否存在（阶段 4 实现）。",
    ClaimType.METADATA_YEAR: "需要核对文献的实际发表年份（Crossref / OpenAlex 元数据，阶段 4 实现）。",
    ClaimType.METADATA_AUTHOR: "需要核对文献的作者列表（Crossref / OpenAlex 元数据，阶段 4 实现）。",
    ClaimType.METADATA_VENUE: "需要核对文献发表的期刊或会议名称（Crossref / OpenAlex 元数据，阶段 4 实现）。",
    ClaimType.QUANTITATIVE: "需要找到该数值的原始出处，并核对数值本身与研究结论（阶段 4/5 实现）。",
    ClaimType.CAUSAL: "需要阅读该来源的摘要或全文，确认其是否真的支持这一结论（阶段 4/5 实现）。",
    ClaimType.ATTRIBUTION: "需要核实该机构是否确实发布过相关内容（阶段 4 实现）。",
    ClaimType.GENERAL_FACT: "需要检索权威公开来源进行核对（阶段 4 实现）。",
    ClaimType.OPINION: "观点与预测无法用外部证据核验，因此不作判断。",
}


# ==========================================================================
# 裁决主体
# ==========================================================================
@dataclass(frozen=True)
class AdjudicationContext:
    """裁决所需的全部上下文。"""

    text: str
    analysis: DocumentAnalysis
    conflicts_by_claim: dict[str, list[InternalConflict]]
    evidence_ids_by_claim: dict[str, list[str]]
    retrieval_enabled: bool = False
    pipeline_stage: str = "阶段 2"


def _skipped_trace(claim: Claim, stage: str) -> SearchTrace:
    """未启用检索时的检索痕迹：如实记录"没有检索"，而不是假装查过了。"""
    return SearchTrace(
        records=[
            SearchRecord(
                source=None,
                query=claim.normalized_claim[:120],
                status=SearchStatus.SKIPPED,
                hit_count=0,
                detail=f"{stage}：未接入外部文献检索，本条主张尚未进行任何外部核查。",            )
        ]
    )


def internal_evidence_for_conflicts(
    conflicts: list[InternalConflict],
) -> tuple[list[Evidence], dict[str, list[str]]]:
    """把文内矛盾转换成证据对象（来源 = 原文内部，不需要也不允许伪造 URL）。

    返回：
        - 证据列表（evidence_id 由调用方统一编号后回填）；
        - 主张 ID → 证据 ID 列表 的映射，供裁决时引用。
    """
    evidence: list[Evidence] = []
    ids_by_claim: dict[str, list[str]] = {}
    counter = 0
    for conflict in conflicts:
        conflict_ids: list[str] = []
        for fragment in conflict.fragments:
            counter += 1
            evidence_id = f"E{counter:03d}"
            conflict_ids.append(evidence_id)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    claim_ids=list(conflict.claim_ids),
                    source=EvidenceSource.INTERNAL,
                    level=EvidenceLevel.USER_TEXT,
                    source_span=fragment,
                    snippet=fragment.text,
                    query=conflict.label,
                    relation=EvidenceRelation.REFUTES,
                    relation_rationale=conflict.detail,
                    match_basis=MatchBasis.INTERNAL,
                )
            )
        for claim_id in conflict.claim_ids:
            ids_by_claim.setdefault(claim_id, []).extend(conflict_ids)
    return evidence, ids_by_claim


def evaluate_claim(claim: Claim, context: AdjudicationContext) -> VerificationResult:
    """对一条主张作出裁决。规则命中是"加法"：可以同时命中多条。"""
    signals: SentenceSignals = context.analysis.for_span(
        claim.parent_sentence_span or claim.source_span
    )
    conflicts = context.conflicts_by_claim.get(claim.claim_id, [])

    problem_types: list[ProblemType] = []
    matched_rules: list[str] = []
    reasons: list[str] = []
    missing: list[str] = []
    severity = Severity.NONE

    # ---------------- ① 文内矛盾：唯一可以给出"来源冲突"的情形 ----------------
    if conflicts:
        problem_types.append(ProblemType.SOURCE_CONFLICT)
        matched_rules.extend(c.rule_id for c in conflicts)
        reasons.extend(c.detail for c in conflicts)
        severity = Severity.HIGH
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.SOURCE_CONFLICT,
            problem_types=problem_types,
            severity=severity,
            rationale=" ".join(reasons) + "（该矛盾来自原文自身，无需外部证据即可确认。）",
            refuting_evidence_ids=list(context.evidence_ids_by_claim.get(claim.claim_id, [])),
            search_trace=_skipped_trace(claim, context.pipeline_stage),
            adjudication_rule="、".join(dict.fromkeys(matched_rules)),
        )

    # ---------------- ② 观点 / 预测：明确不判定对错 ----------------
    if claim.checkability is Checkability.NOT_CHECKABLE:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.UNVERIFIABLE,
            problem_types=[],
            severity=Severity.NONE,
            rationale=(
                "该内容属于观点或预测，不属于可以用外部证据核验的事实主张，"
                "因此系统不对其判定对错——这也意味着它不应被当成事实错误。"
            ),
            search_trace=_skipped_trace(claim, context.pipeline_stage),
            missing_evidence=["观点/预测类内容需要的是论证与前提，而不是事实证据。"],
            adjudication_rule="U1",
        )

    # ---------------- ③ 逐个检查"文本本身已暴露的问题" ----------------
    # N5：DOI 格式不合法
    span_text = claim.source_span.text
    if (
        claim.claim_type is ClaimType.DOI_VALIDITY
        and looks_like_doi(span_text)
        and not is_valid_doi_prefix(span_text)
    ):
        problem_types.append(ProblemType.CITATION_MISSING)
        matched_rules.append("N5")
        reasons.append(
            f"原文引用的 DOI「{span_text}」不符合 DOI 语法（前缀应为 4~9 位数字），"
            "这类写法常见于占位符或凭空生成的引用。"
        )
        missing.append("需要通过 DOI 注册机构（Crossref）确认该 DOI 是否存在——目前格式已不合规。")
        severity = Severity.MEDIUM

    # N2：给出具体数值，但整句没有任何可核验出处
    if claim.claim_type is ClaimType.QUANTITATIVE and not signals.has_concrete_source:
        problem_types.append(ProblemType.MISSING_EVIDENCE)
        matched_rules.append("N2")
        reasons.append(
            f"原文给出了具体数值（{span_text}），但这句话里没有任何可核验的出处"
            "（没有 DOI、文献标题、期刊名或链接）。"
        )
        missing.append("需要找到该数值的原始出处，才能判断数字是否被夸大或写错。")
        severity = Severity.MEDIUM if severity is Severity.NONE else severity

    # N3：强因果表述却没有具体来源
    if claim.claim_type is ClaimType.CAUSAL and not signals.has_concrete_source:
        problem_types.append(ProblemType.OVERREACH)
        matched_rules.append("N3")
        reasons.append(
            "原文使用了「证明 / 证实 / 充分说明」这类强因果表述，但没有给出具体的来源，"
            "推断强度超过了可提供的证据。"
        )
        missing.append("需要具体来源（DOI、标题或期刊）以及该来源是否支持该因果结论。")
        severity = Severity.MEDIUM if severity is Severity.NONE else severity

    # N4：模糊权威
    if signals.vague_words and not signals.has_concrete_source:
        problem_types.append(ProblemType.MISSING_EVIDENCE)
        matched_rules.append("N4")
        reasons.append(
            f"原文使用了「{signals.vague_words[0].text}」这类模糊表述来支撑结论，"
            "但没有指出具体是哪一项研究。"
        )
        missing.append("需要把模糊表述替换为可核验的具体来源（DOI、标题或期刊）。")
        severity = Severity.MEDIUM if severity is Severity.NONE else severity

    # ---------------- ④ 兜底：需要外部证据但本次没有检索 ----------------
    if not matched_rules:
        matched_rules.append("N1")
        reasons.append(
            (
                f"本次检测运行在「{context.pipeline_stage}」，尚未接入外部文献检索，"
                "本条主张无法核实，因此标记为「证据不足」——这不代表内容有误。"
            )
            if not context.retrieval_enabled
            else "本次检索没有获得足以判断该主张的证据。"
        )

    missing.append(_MISSING_EVIDENCE_BY_TYPE.get(claim.claim_type, "需要可核验的公开来源。"))

    return VerificationResult(
        claim_id=claim.claim_id,
        status=VerificationStatus.INSUFFICIENT,
        problem_types=list(dict.fromkeys(problem_types)),
        severity=severity,
        rationale=" ".join(dict.fromkeys(reasons)),
        search_trace=_skipped_trace(claim, context.pipeline_stage),
        missing_evidence=list(dict.fromkeys(missing)),
        adjudication_rule="、".join(dict.fromkeys(matched_rules)),
    )


def evaluate_all(
    claims: list[Claim], context: AdjudicationContext
) -> list[VerificationResult]:
    """按主张顺序逐条裁决（顺序稳定，报告可复现）。"""
    return [evaluate_claim(claim, context) for claim in claims]
