"""修正生成：把大模型给出的修正草稿落成可校验、可定位、可追溯的修正对象。

这一层要挡住三类"看起来像修正、其实是新错误"的东西：
    1. **改错位置**：`target_quote` 必须与原文一字不差，且必须落在该主张所属句子内；
    2. **改没证据的地方**：改事实/数字/引用必须引用证据，引用了不存在的证据会被剔除；
       剔完没有证据了，这条修正直接丢弃（宁可建议"弱化表述"，也不许编一个数字）；
    3. **顺手改写别处**：一条修正只负责一个片段，越界的一律丢弃。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verimind.core.enums import FACT_CHANGING_TYPES
from verimind.core.schemas import Correction, CorrectionBatch, DetectionReport, Span


@dataclass(frozen=True)
class CorrectionDossier:
    """给修正员看的"案件材料"（同时也是提示词的内容）。"""

    claim_id: str
    status: str
    problem_types: list[str]
    claim_quote: str
    sentence: str
    rationale: str
    missing_evidence: list[str]
    evidence: list[dict] = field(default_factory=list)

    def as_prompt_payload(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "状态": self.status,
            "问题类型": self.problem_types,
            "主张原文片段": self.claim_quote,
            "所在整句": self.sentence,
            "判断理由": self.rationale,
            "还缺什么证据": self.missing_evidence,
            "可用证据": self.evidence,
        }


def build_dossiers(report: DetectionReport) -> list[CorrectionDossier]:
    """挑出**确实有问题**的主张，整理成修正材料。

    只有状态为「存在矛盾 / 部分支持 / 来源冲突」的主张才需要修正；
    「证据不足 / 暂无法核验」不生成修正——那会把"还没查清"变成"擅自改写"。
    """
    dossiers: list[CorrectionDossier] = []
    for claim in report.claims:
        result = report.result_by_claim(claim.claim_id)
        if result is None or not result.is_suspicious:
            continue

        evidence_ids = list(
            dict.fromkeys(
                result.refuting_evidence_ids
                + result.supporting_evidence_ids
                + result.related_evidence_ids
            )
        )
        evidence_payload: list[dict] = []
        for evidence_id in evidence_ids:
            item = report.evidence_by_id(evidence_id)
            if item is None:
                continue
            evidence_payload.append(
                {
                    "evidence_id": item.evidence_id,
                    "来源": str(item.source.value),
                    "内容层级": str(item.level.value),
                    "判断层面": item.judged_aspect.value if item.judged_aspect else "未做判断",
                    "关系": str(item.relation.value),
                    "标题": item.title,
                    "年份": item.year,
                    "链接": item.url,
                    "片段": (item.snippet or "")[:600],
                    "判断理由": item.relation_rationale,
                }
            )

        dossiers.append(
            CorrectionDossier(
                claim_id=claim.claim_id,
                status=str(result.status.value),
                problem_types=[str(item.value) for item in result.problem_types],
                claim_quote=claim.source_span.text,
                sentence=(claim.parent_sentence_span or claim.source_span).text,
                rationale=result.rationale,
                missing_evidence=list(result.missing_evidence),
                evidence=evidence_payload,
            )
        )
    return dossiers


def _resolve_in_sentence(text: str, quote: str, sentence: Span | None) -> Span | None:
    """在句子范围内定位模型给出的引用片段（找不到就放弃，绝不硬凑位置）。"""
    if not quote:
        return None
    if sentence is None:
        index = text.find(quote)
        return Span.from_text(text, index, index + len(quote)) if index >= 0 else None
    index = text.find(quote, sentence.start)
    if index < 0 or index + len(quote) > sentence.end:
        return None
    return Span.from_text(text, index, index + len(quote))


def build_corrections(
    *,
    text: str,
    report: DetectionReport,
    batch: CorrectionBatch,
) -> tuple[list[Correction], list[str]]:
    """把修正草稿落成 `Correction` 列表，并返回被丢弃项的说明。"""
    notes: list[str] = []
    corrections: list[Correction] = []
    counter = 0

    available_evidence: dict[str, set[str]] = {}
    for dossier in build_dossiers(report):
        available_evidence[dossier.claim_id] = {
            item["evidence_id"] for item in dossier.evidence
        }

    for draft in batch.corrections:
        claim = report.claim_by_id(draft.claim_id)
        if claim is None:
            notes.append(f"修正草稿引用了不存在的主张 {draft.claim_id}，已丢弃。")
            continue

        sentence = claim.parent_sentence_span or claim.source_span
        span = _resolve_in_sentence(text, draft.target_quote, sentence)
        if span is None:
            notes.append(
                f"{draft.claim_id}：修正草稿给出的原文片段「{draft.target_quote[:24]}…」"
                "在原文该句中找不到，已丢弃（不允许在无法精确定位的位置上做修改）。"
            )
            continue

        allowed = available_evidence.get(draft.claim_id, set())
        evidence_ids = [item for item in draft.evidence_ids if item in allowed]
        dropped_ids = [item for item in draft.evidence_ids if item not in allowed]
        if dropped_ids:
            notes.append(
                f"{draft.claim_id}：修正草稿引用了不属于该主张的证据 "
                f"{'、'.join(dropped_ids)}，已剔除。"
            )

        if draft.change_type in FACT_CHANGING_TYPES and not evidence_ids:
            notes.append(
                f"{draft.claim_id}：这条修正属于「{draft.change_type.value}」，"
                "但没有任何可用的证据支撑，已丢弃。"
                "（拿不到证据时只能弱化表述、标注不确定性或删除——不能凭空改写事实。）"
            )
            continue

        counter += 1
        try:
            corrections.append(
                Correction(
                    correction_id=f"K{counter:03d}",
                    claim_id=draft.claim_id,
                    original_span=span,
                    proposed_text=draft.proposed_text,
                    change_type=draft.change_type,
                    rationale=draft.rationale,
                    evidence_ids=evidence_ids,
                    risk_note=draft.risk_note,
                )
            )
        except ValueError as exc:
            notes.append(f"{draft.claim_id}：修正未通过数据校验（{exc}），已丢弃。")

    notes.extend(batch.notes)
    return corrections, notes


__all__ = [
    "CorrectionDossier",
    "build_corrections",
    "build_dossiers",
]
