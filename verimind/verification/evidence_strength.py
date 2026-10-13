"""证据强度判定（规则化，不是概率）。

为什么用"等级"而不是"概率"：
    概率需要校准；没有校准的概率就是假精度。而"证据覆盖到什么程度"是**可观察、可复现**的：
    有没有摘要片段？有没有形成支持/反驳判断？是不是只有一个来源？这些都能逐条指给你看。

    A 有可直接支持或反驳的来源原文片段（外部摘要/全文，或原文内部的对照片段）
    B 有元数据匹配，或摘要存在但仅间接支持，或多来源相互印证
    C 只发现线索：相关但未形成支持或反驳判断
    未获证据：检索范围内没有找到可用资料
"""

from __future__ import annotations

from dataclasses import dataclass

from verimind.core.enums import EvidenceLevel, EvidenceRelation, EvidenceSource
from verimind.core.schemas import Evidence, VerificationResult

#: 触发了"内容层面判断"的关系（说明模型看过了片段并给出了立场）
_JUDGED_RELATIONS = {
    EvidenceRelation.SUPPORTS,
    EvidenceRelation.PARTIALLY_SUPPORTS,
    EvidenceRelation.REFUTES,
}


@dataclass(frozen=True)
class StrengthVerdict:
    level: str
    label: str
    reason: str
    fg: str
    bg: str
    border: str

    @property
    def is_none(self) -> bool:
        return self.level == "未获证据"


_LEVELS = {
    "A": ("#1a7f37", "#e9f7ee", "#a7e0b8"),
    "B": ("#3a5a8c", "#eef3fb", "#b9cbe8"),
    "C": ("#a15c07", "#fff6e5", "#f2d08a"),
    "未获证据": ("#5f6b7a", "#f4f5f7", "#d5d9df"),
}


def _verdict(level: str, reason: str) -> StrengthVerdict:
    fg, bg, border = _LEVELS[level]
    label = "证据等级 " + level if level in ("A", "B", "C") else level
    return StrengthVerdict(level=level, label=label, reason=reason, fg=fg, bg=bg, border=border)


def assess_evidence_strength(
    result: VerificationResult, evidence_items: list[Evidence]
) -> StrengthVerdict:
    """按规则给出证据强度。只看"证据覆盖到什么程度"，不看模型自信程度。"""
    judged_ids = set(result.supporting_evidence_ids) | set(result.refuting_evidence_ids)
    judged = [item for item in evidence_items if item.evidence_id in judged_ids]
    related = [item for item in evidence_items if item.evidence_id in set(result.related_evidence_ids)]

    if not evidence_items:
        return _verdict("未获证据", "在本次检索范围内没有找到可用资料。")

    # 内容层面：有摘要/全文片段且形成了支持或反驳判断
    content_judged = [
        item
        for item in judged
        if item.judged_aspect is not None
        and item.judged_aspect.requires_text
        and item.level in (EvidenceLevel.ABSTRACT, EvidenceLevel.FULLTEXT)
        and item.relation in _JUDGED_RELATIONS
        and (item.snippet or "")
    ]
    if content_judged:
        return _verdict(
            "A",
            f"取得 {len(content_judged)} 条带原文片段的{content_judged[0].level.value}级证据，"
            "并据此形成了支持或反驳判断。",
        )

    # 原文内部对照（同一文本内的矛盾/重复），依据的是用户原文本身
    internal_judged = [
        item
        for item in judged
        if item.source is EvidenceSource.INTERNAL
        and item.source_span is not None
        and item.relation in _JUDGED_RELATIONS
    ]
    if internal_judged:
        return _verdict(
            "A",
            f"依据原文内部可对照的片段（{len(internal_judged)} 处）直接确认了矛盾或冲突，无需外部来源。",
        )

    metadata_judged = [
        item for item in judged if item.level in (EvidenceLevel.METADATA_ONLY, EvidenceLevel.WEBPAGE)
    ]
    sources = {item.source for item in evidence_items}
    if metadata_judged and len(sources) >= 2:
        return _verdict(
            "B",
            f"{len(sources)} 个独立来源的元数据相互印证，但未取得可引用的内容片段。",
        )
    if metadata_judged:
        return _verdict("B", "取得了权威元数据并完成书目信息比对，但未取得内容片段。")

    if related:
        return _verdict("C", f"检索到 {len(related)} 条相关资料，但尚未形成支持或反驳判断。")

    return _verdict(
        "C",
        "取得了资料，但级别或判断层面不足以支撑结论（例如只有标题、或判断层面不适用）。",
    )


__all__ = ["StrengthVerdict", "assess_evidence_strength"]
