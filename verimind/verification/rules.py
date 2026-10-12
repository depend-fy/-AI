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

from dataclasses import dataclass, field

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
    JudgedAspect,
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
    "I3": "两个外部来源对「该文献是否存在」给出了不一致的答案 → 来源冲突",
    "I4": "两个外部来源对书目信息给出了一致的答案、但与原文不一致 → 来源冲突",
    "U1": "观点 / 预测类内容不属于可用外部证据核验的事实主张 → 暂无法核验",
    "E1": "注册机构（Crossref / OpenAlex）明确回答「未收录该 DOI」→ 存在矛盾 + 引用不存在",
    "E2": "检索到的权威元数据与原文所述年份 / 期刊不一致 → 存在矛盾 + 时间、人物或机构信息错误",
    "E3": "文献摘要明确反驳了原文的结论 → 存在矛盾 + 引用与结论不匹配（数值类主张为数据错误）",
    "E4": "文献摘要只支持原文结论的一部分（例如方向一致但数值或口径不同）→ 部分支持",
    "E5": "注册机构中确实存在该文献 → 已核实（仅限「文献是否存在」这一层面）",
    "E6": "权威元数据与原文所述书目信息一致 → 已核实",
    "E7": "文献摘要明确支持原文主张（且引用了原句）→ 已核实",
    "N0": "对外部来源的检索**全部失败**（网络或服务端故障）→ 暂无法核验（这与「没查到」不是一回事）",
    "N1": "需要外部证据才能判断，而本次检索没有获得证据 → 证据不足（不判定为错误）",
    "N2": "原文在这一处给出了具体数值，却没有任何可核验的出处 → 证据不足 + 缺少必要证据",
    "N3": "用了强因果 / 结论性表述（证明、证实、充分说明等），却没有具体来源 → 证据不足 + 逻辑推断过度",
    "N4": "用「大量研究表明」这类模糊表述支撑结论，但没有具体来源 → 证据不足 + 缺少必要证据",
    "N5": "DOI 格式不合法（前缀应为 4~9 位数字）→ 证据不足 + 引用不存在或元数据不匹配（疑似，未证实）",
    "N6": "找到了对应文献，但拿不到摘要或全文，无法判断内容是否被支持 → 证据不足 + 缺少必要证据",
}

#: 每种主张"必须被哪个层面支持"才算已核实。
#: 这条映射是防止越界判断的关键：内容类主张必须有摘要/全文层面的证据。
_CORE_ASPECT: dict[ClaimType, JudgedAspect] = {
    ClaimType.DOI_VALIDITY: JudgedAspect.EXISTENCE,
    ClaimType.WORK_EXISTENCE: JudgedAspect.EXISTENCE,
    ClaimType.METADATA_YEAR: JudgedAspect.BIBLIOGRAPHIC,
    ClaimType.METADATA_AUTHOR: JudgedAspect.BIBLIOGRAPHIC,
    ClaimType.METADATA_VENUE: JudgedAspect.BIBLIOGRAPHIC,
    ClaimType.QUANTITATIVE: JudgedAspect.CONTENT,
    ClaimType.CAUSAL: JudgedAspect.CONTENT,
    ClaimType.ATTRIBUTION: JudgedAspect.CONTENT,
    ClaimType.GENERAL_FACT: JudgedAspect.CONTENT,
}

#: 不同类型的主张"还缺什么证据才能判断"——这是对用户最有用的信息之一
_MISSING_EVIDENCE_BY_TYPE: dict[ClaimType, str] = {
    ClaimType.DOI_VALIDITY: "需要通过 Crossref 查询该 DOI 的注册信息，并核对其标题、作者、年份是否与原文一致。",
    ClaimType.WORK_EXISTENCE: "需要通过标题 / 作者 / 期刊在 Crossref 或 OpenAlex 中检索该文献是否存在。",
    ClaimType.METADATA_YEAR: "需要核对文献的实际发表年份（Crossref / OpenAlex 元数据）。",
    ClaimType.METADATA_AUTHOR: "需要核对文献的作者列表（Crossref / OpenAlex 元数据）。",
    ClaimType.METADATA_VENUE: "需要核对文献发表的期刊或会议名称（Crossref / OpenAlex 元数据）。",
    ClaimType.QUANTITATIVE: "需要找到该数值的原始出处（摘要或全文），并核对数值本身。",
    ClaimType.CAUSAL: "需要阅读该来源的摘要或全文，确认其是否真的支持这一结论。",
    ClaimType.ATTRIBUTION: "需要核实该机构是否确实发布过相关内容（目前没有覆盖机构网站的公开检索源）。",
    ClaimType.GENERAL_FACT: "需要权威公开来源才能核对（目前只接入了学术文献源，非学术事实无法核验）。",
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
    #: 检索到的**外部**证据（按主张分组）
    external_evidence_by_claim: dict[str, list[Evidence]] = field(default_factory=dict)
    #: 每条主张的真实检索痕迹（来自检索层）
    traces: dict[str, SearchTrace] = field(default_factory=dict)
    retrieval_enabled: bool = False
    pipeline_stage: str = "阶段 3"


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
    conflicts: list[InternalConflict], *, start_index: int = 0
) -> tuple[list[Evidence], dict[str, list[str]]]:
    """把文内矛盾转换成证据对象（来源 = 原文内部，不需要也不允许伪造 URL）。

    `start_index` 用于让编号接在已有证据之后（外部检索证据先编号）。
    """
    evidence: list[Evidence] = []
    ids_by_claim: dict[str, list[str]] = {}
    counter = start_index
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
                    judged_aspect=JudgedAspect.INTERNAL_CONSISTENCY,
                )
            )
        for claim_id in conflict.claim_ids:
            ids_by_claim.setdefault(claim_id, []).extend(conflict_ids)
    return evidence, ids_by_claim


def _overlaps(first: Span, second: Span) -> bool:
    return first.start < second.end and second.start < first.end


def _overlapping(matches: list, span: Span) -> list:
    """取出与给定片段有重叠的匹配项。

    为什么要按"重叠"而不是"在不在句子里"判断？
    真实大模型给出的原文引用往往比规则引擎更宽（例如把 `DOI: 10.xxxx/...` 整段都引上），
    如果只判断"主张片段的文字本身像不像 DOI / 数字"，这类判断就会漏掉。
    """
    return [match for match in matches if _overlaps(match.span, span)]


def _pick(
    evidence_list: list[Evidence], aspect: JudgedAspect, relation: EvidenceRelation
) -> list[Evidence]:
    return [
        item
        for item in evidence_list
        if item.judged_aspect is aspect and item.relation is relation
    ]


def _adjudicate_with_evidence(
    claim: Claim,
    evidence_list: list[Evidence],
    trace: SearchTrace,
) -> VerificationResult | None:
    """依据**外部证据**作出裁决。返回 None 表示证据不足以支撑任何结论，交回文本层面的规则。

    判断顺序刻意设计为"先看矛盾、再看支持"，因为：
        · 一条"存在矛盾"的结论必须有证据支撑，而这里每个分支都挂着真实的 evidence_id；
        · 只要还有互相矛盾的证据，就绝不能给出"已核实"。
    """
    existence_ok = _pick(evidence_list, JudgedAspect.EXISTENCE, EvidenceRelation.SUPPORTS)
    existence_bad = _pick(evidence_list, JudgedAspect.EXISTENCE, EvidenceRelation.REFUTES)
    biblio_ok = _pick(evidence_list, JudgedAspect.BIBLIOGRAPHIC, EvidenceRelation.SUPPORTS)
    biblio_bad = _pick(evidence_list, JudgedAspect.BIBLIOGRAPHIC, EvidenceRelation.REFUTES)
    content_ok = _pick(evidence_list, JudgedAspect.CONTENT, EvidenceRelation.SUPPORTS)
    content_partial = _pick(
        evidence_list, JudgedAspect.CONTENT, EvidenceRelation.PARTIALLY_SUPPORTS
    )
    content_bad = _pick(evidence_list, JudgedAspect.CONTENT, EvidenceRelation.REFUTES)

    def ids(items: list[Evidence]) -> list[str]:
        return [item.evidence_id for item in items]

    def reasons_of(items: list[Evidence]) -> list[str]:
        return [item.relation_rationale for item in items if item.relation_rationale]

    # ---------------- 外部来源互相打架 ----------------
    if existence_ok and existence_bad:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.SOURCE_CONFLICT,
            problem_types=[ProblemType.SOURCE_CONFLICT],
            severity=Severity.HIGH,
            rationale=(
                "不同外部来源对「该文献是否存在」给出了不一致的答案："
                + " ".join(reasons_of([*existence_ok, *existence_bad]))
                + " 在两个可信来源统一之前，不能给出确定结论。"
            ),
            supporting_evidence_ids=ids(existence_ok),
            refuting_evidence_ids=ids(existence_bad),
            search_trace=trace,
            adjudication_rule="I3",
        )

    # 同一个书目问题（例如年份）被不同来源给出了不同答案 → 同样属于来源冲突。
    # 这类情况少见但真实存在（Crossref 与 OpenAlex 的年份口径有时不同），
    # 不能挑一个信了就算，必须如实报告"来源之间不一致"。
    if biblio_ok and biblio_bad:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.SOURCE_CONFLICT,
            problem_types=[ProblemType.SOURCE_CONFLICT],
            severity=Severity.HIGH,
            rationale=(
                "不同外部来源对同一条书目信息给出了不一致的答案："
                + " ".join(reasons_of([*biblio_ok, *biblio_bad]))
                + " 在来源统一之前，不能据此判定原文有误。"
            ),
            supporting_evidence_ids=ids(biblio_ok),
            refuting_evidence_ids=ids(biblio_bad),
            search_trace=trace,
            adjudication_rule="I4",
        )

    # ---------------- 明确矛盾（证据充分） ----------------
    if existence_bad and not existence_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.CONTRADICTED,
            problem_types=[ProblemType.CITATION_MISSING],
            severity=Severity.HIGH,
            rationale=(
                " ".join(reasons_of(existence_bad))
                + " 因此原文引用的这篇文献在注册机构中不存在，属于「引用不存在」。"
            ),
            refuting_evidence_ids=ids(existence_bad),
            search_trace=trace,
            missing_evidence=["如该文献确实存在，请提供正确的 DOI 或其他可核验的标识。"],
            adjudication_rule="E1",
        )

    if biblio_bad:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.CONTRADICTED,
            problem_types=[ProblemType.METADATA_ERROR],
            severity=Severity.HIGH,
            rationale=" ".join(reasons_of(biblio_bad)),
            supporting_evidence_ids=ids(biblio_ok),
            refuting_evidence_ids=ids(biblio_bad),
            search_trace=trace,
            missing_evidence=["需要确认原文引用的是同一篇文献（标题/作者一致），以排除张冠李戴。"],
            adjudication_rule="E2",
        )

    if content_bad:
        problem = (
            ProblemType.NUMERIC_ERROR
            if claim.claim_type is ClaimType.QUANTITATIVE
            else ProblemType.CITATION_MISMATCH
        )
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.CONTRADICTED,
            problem_types=[problem],
            severity=Severity.HIGH,
            rationale=(
                " ".join(reasons_of(content_bad))
                + "（该判断依据的是文献摘要，摘要不等于全文；但摘要与原文结论直接冲突时，"
                "原文的表述至少是不准确的。）"
            ),
            refuting_evidence_ids=ids(content_bad),
            search_trace=trace,
            missing_evidence=["如有全文，建议进一步核对该结论适用的实验条件与统计口径。"],
            adjudication_rule="E3",
        )

    if content_partial:
        problem = (
            ProblemType.NUMERIC_ERROR
            if claim.claim_type is ClaimType.QUANTITATIVE
            else ProblemType.MISSING_EVIDENCE
        )
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.PARTIALLY_SUPPORTED,
            problem_types=[problem],
            severity=Severity.MEDIUM,
            rationale=" ".join(reasons_of(content_partial)),
            supporting_evidence_ids=ids(content_partial),
            search_trace=trace,
            missing_evidence=["需要全文确认具体数值与口径是否与原文一致。"],
            adjudication_rule="E4",
        )

    # ---------------- 已核实（必须命中该主张对应的核心层面） ----------------
    core_aspect = _CORE_ASPECT.get(claim.claim_type)
    if core_aspect is JudgedAspect.EXISTENCE and existence_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            problem_types=[],
            severity=Severity.NONE,
            rationale=(
                " ".join(reasons_of(existence_ok))
                + " 注意：这只核实了「文献存在」这一层面，不代表其内容支持原文的任何结论。"
            ),
            supporting_evidence_ids=ids(existence_ok),
            search_trace=trace,
            adjudication_rule="E5",
        )

    if core_aspect is JudgedAspect.BIBLIOGRAPHIC and biblio_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            problem_types=[],
            severity=Severity.NONE,
            rationale=" ".join(reasons_of(biblio_ok)),
            supporting_evidence_ids=ids(biblio_ok),
            search_trace=trace,
            adjudication_rule="E6",
        )

    if core_aspect is JudgedAspect.CONTENT and content_ok:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            problem_types=[],
            severity=Severity.NONE,
            rationale=(
                " ".join(reasons_of(content_ok))
                + "（依据文献摘要，摘要中已引用相关原句；这是摘要层面的核实，不等于通读全文。）"
            ),
            supporting_evidence_ids=ids(content_ok),
            search_trace=trace,
            adjudication_rule="E7",
        )

    # ---------------- 找到了文献，但拿不到摘要/全文 ----------------
    # 注意这里**不论证据是否被判为"支持存在性"**：只要该主张是内容类的、
    # 而且我们确实检索到了它的来源文献（无论是 Crossref 还是 OpenAlex 的记录），
    # 却说不出内容层面的话，就必须如实说明"缺少摘要或全文"。
    if core_aspect is JudgedAspect.CONTENT and evidence_list and not (
        content_ok or content_partial or content_bad
    ):
        usable = [
            item
            for item in evidence_list
            if item.relation is EvidenceRelation.UNKNOWN or item.judged_aspect is None
        ]
        if usable:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INSUFFICIENT,
                problem_types=[ProblemType.MISSING_EVIDENCE],
                severity=Severity.MEDIUM,
                rationale=(
                    "检索到了对应文献，但**没有拿到摘要或全文**，因此无法判断该文献是否支持原文的这个结论。"
                    "系统不会因为「查到了论文」就默认它支持原文的说法。"
                ),
                supporting_evidence_ids=ids([*existence_ok, *biblio_ok]),
                related_evidence_ids=ids(usable),
                search_trace=trace,
                missing_evidence=["需要该文献的摘要或全文，才能判断内容是否被支持。"],
                adjudication_rule="N6",
            )

    return None


def evaluate_claim(claim: Claim, context: AdjudicationContext) -> VerificationResult:
    """对一条主张作出裁决。规则命中是"加法"：可以同时命中多条。"""
    signals: SentenceSignals = context.analysis.for_span(
        claim.parent_sentence_span or claim.source_span
    )
    conflicts = context.conflicts_by_claim.get(claim.claim_id, [])
    # 检索痕迹：优先用检索层真实记录；没有就如实写"未检索"
    trace = context.traces.get(claim.claim_id) or _skipped_trace(claim, context.pipeline_stage)
    external_evidence = context.external_evidence_by_claim.get(claim.claim_id, [])

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
            search_trace=trace,
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
            search_trace=trace,
            missing_evidence=["观点/预测类内容需要的是论证与前提，而不是事实证据。"],
            adjudication_rule="U1",
        )

    # ---------------- ③ 外部证据：能据此下结论就先下结论 ----------------
    if external_evidence:
        decided = _adjudicate_with_evidence(claim, external_evidence, trace)
        if decided is not None:
            return decided

    # ---------------- ③之二 检索彻底失败：只能说"暂无法核验" ----------------
    # 「查了但没查到」和「检索服务坏了」是两件完全不同的事：
    # 前者可能是证据不足，后者什么都说明不了，绝不能借用它来暗示内容有问题。
    if context.retrieval_enabled and trace.all_failed:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.UNVERIFIABLE,
            problem_types=[],
            severity=Severity.NONE,
            rationale=(
                "本次对外部来源的检索全部失败（网络或服务端故障），因此无法判断这条主张。"
                "这与「查了但没查到」不同：它说明的是检索不可用，而不是证据不存在。"
            ),
            search_trace=trace,
            missing_evidence=["需要等检索服务恢复后重新检测。"],
            adjudication_rule="N0",
        )

    # ---------------- ④ 逐个检查"文本本身已暴露的问题" ----------------
    # N5：DOI 格式不合法
    # 注意两种触发方式都要有：
    #   ① 主张片段本身就是一个不合规 DOI（规则引擎给的精确片段）；
    #   ② 句子里出现的不合规 DOI 与该主张片段重叠（大模型常把 "DOI: " 前缀一起引用）。
    span_text = claim.source_span.text
    invalid_hits = _overlapping(signals.invalid_dois, claim.source_span)
    span_is_invalid_doi = looks_like_doi(span_text) and not is_valid_doi_prefix(span_text)
    if claim.claim_type is ClaimType.DOI_VALIDITY and (invalid_hits or span_is_invalid_doi):
        shown = invalid_hits[0].text if invalid_hits else span_text
        problem_types.append(ProblemType.CITATION_MISSING)
        matched_rules.append("N5")
        reasons.append(
            f"原文引用的 DOI「{shown}」不符合 DOI 语法（前缀应为 4~9 位数字），"
            "这类写法常见于占位符或凭空生成的引用。"
        )
        missing.append("需要通过 DOI 注册机构（Crossref）确认该 DOI 是否存在——目前格式已不合规。")
        severity = Severity.MEDIUM

    # N2：给出具体数值，但整句没有任何可核验出处
    # 双重条件：① 主张类型是数值类；② 主张片段里**确实**含有数值。
    # 第 ② 条是为了防止"模型把不含数字的论断标成数值类"时产生误报。
    number_hits = _overlapping(
        [*signals.percents, *signals.strong_quantities, *signals.weak_quantities],
        claim.source_span,
    )
    if (
        claim.claim_type is ClaimType.QUANTITATIVE
        and not signals.has_concrete_source
        and number_hits
    ):
        invalid_note = ""
        if signals.invalid_dois:
            invalid_note = (
                f"（句中出现的「{signals.invalid_dois[0].text}」格式不合规，不构成可核验来源）"
            )
        problem_types.append(ProblemType.MISSING_EVIDENCE)
        matched_rules.append("N2")
        reasons.append(
            f"原文在这一处给出了具体数值（{span_text[:40]}），"
            f"但整句里没有任何可核验的出处{invalid_note}。"
        )
        missing.append("需要找到该数值的原始出处，才能判断数字是否被夸大或写错。")
        severity = Severity.MEDIUM if severity is Severity.NONE else severity

    # N3：强因果 / 结论性表述却没有具体来源
    if claim.claim_type is ClaimType.CAUSAL and not signals.has_concrete_source:
        problem_types.append(ProblemType.OVERREACH)
        matched_rules.append("N3")
        if signals.causal_words:
            word = signals.causal_words[0].text
            reasons.append(
                f"原文使用了「{word}」这类强因果表述，但没有给出具体的来源，"
                "推断强度超过了可提供的证据。"
            )
        else:
            reasons.append(
                "原文把该说法表述为因果或结论性判断，但整句没有任何可核验的出处，"
                "推断强度可能超过了可提供的证据。"
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
        search_trace=trace,
        missing_evidence=list(dict.fromkeys(missing)),
        adjudication_rule="、".join(dict.fromkeys(matched_rules)),
    )


def evaluate_all(
    claims: list[Claim], context: AdjudicationContext
) -> list[VerificationResult]:
    """按主张顺序逐条裁决（顺序稳定，报告可复现）。"""
    return [evaluate_claim(claim, context) for claim in claims]
