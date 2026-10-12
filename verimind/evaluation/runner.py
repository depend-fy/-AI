"""评测运行器：在人工标注集上真实跑一遍，产出可复核的评测报告。

三条纪律：
    1. **跑的就是生产线**：每条用例都调用 `run_detection()`，而不是另写一条"评测专用"的简化路径——
       否则测出来的数不代表产品。
    2. **成本可控**：模型与检索结果都有缓存，重复评测不重复付费；支持按类别/条数限制。
    3. **不粉饰**：未命中的用例、失败案例、样本量、标注者身份、同源偏差风险，全部写进报告。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from verimind.core.config import Settings
from verimind.core.enums import EvidenceSource
from verimind.core.pipeline import ProgressEvent, run_detection
from verimind.core.schemas import DetectionReport, VeriMindModel
from verimind.evaluation.dataset import GoldenCase, load_golden_set, match_claim
from verimind.evaluation.metrics import CaseOutcome, compute_metrics, summarize_failures
from verimind.review.independent_review import extract_facts

#: 评测基线标识。B0（让模型自检）**尚未实现**——我们不做没有真正跑过的对照表。
BASELINE_B2 = "B2 证据增强（本项目完整流程）"
BASELINE_B1 = "B1 离线规则引擎 + 不检索（下界参照）"


class EvaluationRow(VeriMindModel):
    """一条用例的逐条记录（评测页可展开查看）。"""

    case_id: str
    category: str
    text: str
    gold_status: str
    predicted_status: str | None = None
    matched: bool = False
    matched_claim_quote: str | None = None
    gold_problem_types: list[str] = Field(default_factory=list)
    predicted_problem_types: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_urls: list[str] = Field(default_factory=list)
    corrections: int = 0
    corrections_applied: int = 0
    corrected_text: str | None = None
    new_fact_tokens: list[str] = Field(default_factory=list)
    run_id: str = ""
    latency_ms: int = 0
    llm_calls: int = 0
    retrieval_queries: int = 0
    notes: str = ""


class EvaluationReport(VeriMindModel):
    """一次评测的完整结果。"""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    baseline: str
    engine_summary: str
    case_count: int
    rows: list[EvaluationRow] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    def metric(self, name: str) -> Any:
        return self.metrics.get(name)


def _new_fact_tokens(report: DetectionReport) -> list[str]:
    """计算"修正后出现的、原文与全部证据里都没有的事实 token"。

    这是"新增错误率"的判定方式——用确定性规则而不是人工逐条看，
    因此它可复现、可解释：抽出来的每个 token 都能指给你看。
    """
    if not report.corrected_text or report.corrected_text == report.input_text:
        return []
    original = extract_facts(report.input_text)
    evidence_parts: list[str] = []
    for item in report.evidence:
        evidence_parts.extend(
            [
                str(item.year or ""),
                item.doi or "",
                item.title or "",
                item.snippet or "",
                item.relation_rationale or "",
            ]
        )
    known = original | extract_facts(" ".join(evidence_parts))
    produced = extract_facts(report.corrected_text)
    return sorted(produced - known)


def _outcome(case: GoldenCase, report: DetectionReport) -> CaseOutcome:
    claim, result = match_claim(report, case)
    evidence_ids = []
    if result is not None:
        evidence_ids = list(
            dict.fromkeys(
                result.supporting_evidence_ids
                + result.refuting_evidence_ids
                + result.related_evidence_ids
            )
        )
    evidence_items = [report.evidence_by_id(item) for item in evidence_ids]
    traceable = 0
    for item in evidence_items:
        if item is None:
            continue
        if item.source is EvidenceSource.INTERNAL and item.source_span is not None:
            traceable += 1
        elif item.url:
            traceable += 1

    corrected = report.corrected_text or ""
    contains_ok = all(token in corrected for token in case.expect_correction_contains) if case.expect_correction_contains else True
    excludes_ok = not any(token in corrected for token in case.expect_correction_excludes)

    return CaseOutcome(
        case_id=case.case_id,
        category=case.category,
        gold_status=str(case.gold_status.value),
        predicted_status=str(result.status.value) if result else None,
        matched=result is not None,
        gold_problem_types=tuple(str(item.value) for item in case.gold_problem_types),
        predicted_problem_types=tuple(str(item.value) for item in result.problem_types) if result else (),
        evidence_count=len([item for item in evidence_items if item is not None]),
        traceable_evidence_count=traceable,
        corrections_total=len(report.corrections),
        corrections_applied=len(report.applied_correction_ids),
        correction_contains_ok=contains_ok,
        correction_excludes_ok=excludes_ok,
        new_fact_tokens=tuple(_new_fact_tokens(report)),
        latency_ms=report.metrics.latency_ms or 0,
        prompt_tokens=report.metrics.prompt_tokens,
        completion_tokens=report.metrics.completion_tokens,
        llm_calls=report.metrics.llm_calls,
        retrieval_queries=report.metrics.retrieval_queries,
    )


def _row(case: GoldenCase, report: DetectionReport) -> EvaluationRow:
    claim, result = match_claim(report, case)
    evidence_ids: list[str] = []
    if result is not None:
        evidence_ids = list(
            dict.fromkeys(
                result.supporting_evidence_ids
                + result.refuting_evidence_ids
                + result.related_evidence_ids
            )
        )
    urls: list[str] = []
    for evidence_id in evidence_ids:
        item = report.evidence_by_id(evidence_id)
        if item is None:
            continue
        if item.url:
            urls.append(item.url)
        elif item.source_span is not None:
            urls.append(f"原文内部：{item.source_span.text}")
    return EvaluationRow(
        case_id=case.case_id,
        category=case.category,
        text=case.text,
        gold_status=str(case.gold_status.value),
        predicted_status=str(result.status.value) if result else None,
        matched=result is not None,
        matched_claim_quote=claim.source_span.text if claim else None,
        gold_problem_types=[str(item.value) for item in case.gold_problem_types],
        predicted_problem_types=[str(item.value) for item in result.problem_types] if result else [],
        evidence_ids=evidence_ids,
        evidence_urls=urls,
        corrections=len(report.corrections),
        corrections_applied=len(report.applied_correction_ids),
        corrected_text=report.corrected_text,
        new_fact_tokens=_new_fact_tokens(report),
        run_id=report.run_id,
        latency_ms=report.metrics.latency_ms or 0,
        llm_calls=report.metrics.llm_calls,
        retrieval_queries=report.metrics.retrieval_queries,
        notes=case.notes,
    )


def run_evaluation(
    *,
    settings: Settings,
    cases: list[GoldenCase] | None = None,
    golden_path: Path | None = None,
    baseline: str = BASELINE_B2,
    limit: int | None = None,
    only_categories: list[str] | None = None,
    only_case_ids: list[str] | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
    on_case: Callable[[int, int, GoldenCase], None] | None = None,
) -> EvaluationReport:
    """在标注集上跑一遍，返回评测报告。"""
    all_cases = cases if cases is not None else load_golden_set(golden_path)
    selected = [
        case
        for case in all_cases
        if (not only_categories or case.category in only_categories)
        and (not only_case_ids or case.case_id in only_case_ids)
    ]
    if limit is not None:
        selected = selected[:limit]

    outcomes: list[CaseOutcome] = []
    rows: list[EvaluationRow] = []
    for index, case in enumerate(selected, start=1):
        if on_case is not None:
            on_case(index, len(selected), case)
        report = run_detection(case.text, settings=settings, progress=progress)
        outcomes.append(_outcome(case, report))
        rows.append(_row(case, report))

    needs_correction = {case.case_id for case in selected if case.needs_correction}
    metrics = compute_metrics(outcomes, needs_correction)

    ai_labelled = sum(1 for case in selected if case.labeler_is_ai)
    awaiting = sum(1 for case in selected if case.awaiting_confirmation)
    confirmed = len(selected) - awaiting
    caveats: list[str] = [
        (
            f"**样本量只有 {len(selected)} 条**，按类别分组后每类仅 2–3 条；"
            "因此这些指标应视为**初步结果**，不足以支撑统计显著性结论。"
        ),
        (
            "真值锚定方式：涉及文献的用例都用 Crossref **实测**确认过（年份/是否存在），"
            "构造类用例（证据不足、文内矛盾）的真值来自文本自身。"
        ),
        (
            "「新增错误率」由确定性规则计算：抽出修正后出现的年份/数值/DOI，"
            "检查它能否在原文或证据中找到依据；不是人工逐条判定。"
        ),
        (
            "被测系统与标注起草使用的是**同一个模型家族**，存在同源偏差（self-preference bias）风险；"
            "标注也未做双人独立标注，因此**没有计算标注一致性（如 Cohen's kappa）**。"
        ),
        (
            "**最重要的一条局限**：起草标注的人（AI）**知道系统是怎么实现的**，"
            "因此 gold 可能与系统能力过度贴合（自我实现偏差）——"
            "例如「跨语言期刊名不一致」这一项，标注刻意与系统的保守行为保持了一致。"
            "真正可靠的评测应由**不了解系统实现的人独立标注**，再与系统输出比对。"
        ),
    ]
    if awaiting:
        caveats.append(
            f"标注状态：本次涉及的用例中 **{confirmed}/{len(selected)} 条已由人工逐条确认**，"
            f"仍有 {awaiting} 条为 AI 起草、等待人工抽查确认。"
            "在全部确认之前，请不要把这些数字对外表述为「人工标注结果」。"
        )
    else:
        caveats.append(
            f"标注状态：本次涉及的 {len(selected)} 条用例**全部经人工逐条确认**"
            "（确认意见记录在标注集的 human_comment 字段里）。"
        )
    if ai_labelled:
        caveats.append(
            "标注起草方式：AI 起草（与被测系统同一模型家族）；人工确认只判断「标注是否合理」，"
            "并未重新独立标注全文——因此它仍不能替代「由不知情者独立标注」的评测。"
        )

    engine = "离线规则引擎（不检索）" if settings.llm_provider == "mock" else f"{settings.llm_provider}/{settings.llm_model}"
    if settings.reviewer_is_distinct:
        engine += f" + 独立复核 {settings.llm_reviewer_provider}/{settings.llm_reviewer_model}"
    else:
        engine += "（复核由主模型兼任）"

    return EvaluationReport(
        baseline=baseline,
        engine_summary=engine,
        case_count=len(selected),
        rows=rows,
        metrics=metrics,
        failures=summarize_failures(outcomes),
        caveats=caveats,
    )


def offline_settings(settings: Settings) -> Settings:
    """B1 基线用的配置：离线规则引擎 + 不检索（完全免费，可作为下界参照）。"""
    return settings.model_copy(
        update={"llm_provider": "mock", "llm_api_key": "", "retrieve_enabled": False}
    )


__all__ = [
    "BASELINE_B1",
    "BASELINE_B2",
    "EvaluationReport",
    "EvaluationRow",
    "offline_settings",
    "run_evaluation",
]
