"""证据-主张匹配：把"查到了什么"翻译成"这条主张的哪个层面被支持/被反驳"。

分工非常明确，这是本项目防止越界判断的关键：

    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ 判断层面              │ 用什么方法                                   │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ 文献是否存在          │ 确定性规则（注册机构查到 / 明确返回 404）      │
    │ 书目信息（年份/作者/期刊）│ 确定性规则（与实际元数据逐项比对）            │
    │ 内容结论是否被支持     │ 只能由大模型读摘要判断（角色 B），本项目在此  │
    │                      │ **不做**任何"关键词命中就算支持"的猜测        │
    └──────────────────────┴──────────────────────────────────────────────┘

确定性规则的好处：可以穷举单元测试、零成本、结果可复现；
而且它恰好覆盖了"虚构文献""年份写错"这两类最常见的幻觉。
"""

from __future__ import annotations

from verimind.claims.preprocess import DocumentAnalysis
from verimind.core.enums import (
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    JudgedAspect,
    MatchBasis,
)
from verimind.core.schemas import Claim, Evidence
from verimind.retrieval.parsing import normalize_whitespace, similarity

#: 标题相似度阈值：低于它就不认为"查到的是同一篇"
_TITLE_MATCH_THRESHOLD = 0.86

#: 年份比对的容差（有些文献的"在线发表"与"正式发表"会差一年）
_YEAR_TOLERANCE = 0

#: 期刊/会议名称相似度阈值
_VENUE_MATCH_THRESHOLD = 0.8


def _revise(evidence: Evidence, **updates: object) -> Evidence | None:
    """在证据上附加一个判断（会走一遍完整校验，越界判断会被拒绝并返回 None）。"""
    payload = evidence.model_dump()
    payload.update(updates)
    try:
        return Evidence.model_validate(payload)
    except ValueError:
        # 校验失败说明这个判断"越界"了（例如用元数据判断内容），如实放弃
        return None


def _claimed_years(claim: Claim, analysis: DocumentAnalysis) -> list[int]:
    signals = analysis.for_span(claim.parent_sentence_span or claim.source_span)
    years = [
        int(match.value)
        for match in signals.years
        if isinstance(match.value, int)
        and match.span.start < claim.source_span.end
        and claim.source_span.start < match.span.end
    ]
    if years:
        return years
    return [int(match.value) for match in signals.years if isinstance(match.value, int)]


def assess_metadata(
    claim: Claim, evidence: Evidence, analysis: DocumentAnalysis
) -> Evidence | None:
    """在"存在性 / 书目信息"两个层面上判断证据与主张的关系。

    返回修订后的证据；无法判断时返回原证据（不改动）。
    """
    if not evidence.title and not evidence.doi:
        # 没有可用元数据（例如只有一条 404 记录），交给规则层处理
        return evidence

    # ---------------- ① 文献是否存在 ----------------
    if claim.claim_type in (ClaimType.DOI_VALIDITY, ClaimType.WORK_EXISTENCE):
        revised = _revise(
            evidence,
            relation=EvidenceRelation.SUPPORTS,
            judged_aspect=JudgedAspect.EXISTENCE,
            match_basis=MatchBasis.METADATA,
            relation_rationale=(
                f"在 {evidence.source.value} 中查到了这条记录："
                f"{evidence.title or evidence.doi}（{evidence.year or '年份未知'}），"
                "说明该文献真实存在。注意：这只证明文献存在，不代表其内容支持原文的结论。"
            ),
        )
        return revised or evidence

    # ---------------- ② 书目信息：年份 ----------------
    if claim.claim_type is ClaimType.METADATA_YEAR:
        claimed_years = _claimed_years(claim, analysis)
        if not claimed_years or evidence.year is None:
            return evidence
        claimed = claimed_years[0]
        actual = evidence.year
        if abs(actual - claimed) <= _YEAR_TOLERANCE:
            revised = _revise(
                evidence,
                relation=EvidenceRelation.SUPPORTS,
                judged_aspect=JudgedAspect.BIBLIOGRAPHIC,
                match_basis=MatchBasis.METADATA,
                relation_rationale=(
                    f"{evidence.source.value} 记录显示该文献发表于 {actual} 年，"
                    f"与原文所述一致。"
                ),
            )
        else:
            revised = _revise(
                evidence,
                relation=EvidenceRelation.REFUTES,
                judged_aspect=JudgedAspect.BIBLIOGRAPHIC,
                match_basis=MatchBasis.METADATA,
                relation_rationale=(
                    f"{evidence.source.value} 记录显示该文献发表于 {actual} 年，"
                    f"而原文写的是 {claimed} 年——两者不一致。"
                ),
            )
        return revised or evidence

    # ---------------- ③ 书目信息：期刊/会议 ----------------
    if claim.claim_type is ClaimType.METADATA_VENUE and evidence.venue:
        claimed_venue = normalize_whitespace(claim.source_span.text)
        score = similarity(claimed_venue, evidence.venue)
        if score >= _VENUE_MATCH_THRESHOLD:
            revised = _revise(
                evidence,
                relation=EvidenceRelation.SUPPORTS,
                judged_aspect=JudgedAspect.BIBLIOGRAPHIC,
                match_basis=MatchBasis.METADATA,
                relation_rationale=(
                    f"{evidence.source.value} 记录中的期刊/会议是「{evidence.venue}」，"
                    f"与原文一致（相似度 {score:.2f}）。"
                ),
            )
            return revised or evidence
        # 太模糊就不下结论，避免把中文期刊名与英文刊名的不一致误判成错误
        return evidence

    # ---------------- ④ 其它类型：元数据层面判断不了内容 ----------------
    return evidence


def assess_title_match(evidence: Evidence, query: str) -> Evidence | None:
    """判断"检索到的这篇"是不是"原文说的那篇"。

    只有在按标题检索（没有 DOI）时才需要这一步：如果检索结果标题与查询差异很大，
    就不能把它的元数据当成原文那条引用的元数据来比对，否则会凭空造出一个"年份错误"。
    """
    if evidence.doi or not evidence.title or not query:
        return evidence
    score = similarity(evidence.title, query)
    if score < _TITLE_MATCH_THRESHOLD:
        return None  # 不是一个东西，丢弃
    return evidence


def content_pairs(evidence_list: list[Evidence]) -> list[Evidence]:
    """挑出可以进入"内容层面判断"的证据（必须有摘要或全文片段）。"""
    return [
        item
        for item in evidence_list
        if item.level in (EvidenceLevel.ABSTRACT, EvidenceLevel.FULLTEXT)
        and (item.snippet or "").strip()
    ]


def apply_content_assessment(
    evidence: Evidence,
    *,
    relation: EvidenceRelation,
    rationale: str,
    quoted_snippet: str | None,
) -> Evidence | None:
    """把角色 B 的**内容层面**判断写到证据上（会走完整校验）。

    三条硬约束（校验器会拒绝，这里也会提前返回 None）：
        1. 没有摘要/全文的证据不能承载内容层面的判断；
        2. 判断"支持/部分支持/反驳"时必须**由模型给出**被引用的原句
           （不允许拿整段摘要当引用糊弄过去）；
        3. 引用的原句会被写进 snippet —— 报告里显示的就是"摘要中到底哪句话支持/反驳了你"。
    """
    judged = relation in (
        EvidenceRelation.SUPPORTS,
        EvidenceRelation.PARTIALLY_SUPPORTS,
        EvidenceRelation.REFUTES,
    )
    if judged and not (quoted_snippet or "").strip():
        return None
    return _revise(
        evidence,
        relation=relation,
        judged_aspect=JudgedAspect.CONTENT,
        match_basis=(
            MatchBasis.ABSTRACT if evidence.level is EvidenceLevel.ABSTRACT else MatchBasis.FULLTEXT
        ),
        relation_rationale=rationale,
        snippet=(quoted_snippet or evidence.snippet),
    )


def snippet_in_text(snippet: str | None, text: str | None) -> bool:
    """判断模型引用的原句是否**真的**出现在证据文本里。

    这是防止模型编造引用的关键一步：模型可能"润色"或翻译原句，
    那样的人工引用无法复核，我们选择丢弃该判断（而不是照单全收）。
    """
    if not snippet or not text:
        return False
    needle = normalize_whitespace(snippet).lower()
    haystack = normalize_whitespace(text).lower()
    if not needle:
        return False
    if needle in haystack:
        return True
    # 容忍标点/空格的细微差异：取前 30 个字符再试一次
    if len(needle) >= 30 and needle[:30] in haystack:
        return True
    # 去掉所有空白后再比一次
    compact_needle = "".join(needle.split())
    compact_haystack = "".join(haystack.split())
    if len(compact_needle) >= 20 and compact_needle in compact_haystack:
        return True
    return False


__all__ = [
    "apply_content_assessment",
    "assess_metadata",
    "assess_title_match",
    "content_pairs",
    "snippet_in_text",
]
