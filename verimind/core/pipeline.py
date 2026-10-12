"""流水线编排：把「文本 → 报告」的全过程串起来。

阶段 3 之后的执行步骤
--------------------
    ① 校验输入
    ② 规则引擎拆解（**永远执行**，作为基线与兜底；离线可用）
    ③ 大模型拆解（配置了真实模型时执行）→ 与规则结果合并 + 交叉比对
    ④ 文内一致性检查（唯一无需外部证据就能给出"来源冲突"的来源）
    ⑤ 规则裁决（6 种状态 + 问题类型 + 缺失证据清单）
    ⑥ 事实审查员（角色 A）复核 → 写入各角色结论 + 记录分歧
    ⑦ 生成报告、统计 token/成本、落盘

两个关键设计
------------
1. **模型失败不等于检测失败**：模型调用出错时降级为"仅规则引擎"，
   但必须在报告里**显著声明**降级原因，并把运行模式改回「离线模拟」——
   悄悄降级然后用规则结果冒充模型结果，是这个项目最不能接受的行为。
2. **模型不决定结论**：模型只提供"怀疑方向"，最终状态由规则裁决层给出。
   模型怀疑但规则层未确认的，会作为**分歧**写进报告，而不是直接判成错误。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

from verimind.claims.extractor import ExtractionResult, PRIORITY_RANK
from verimind.claims.preprocess import analyze_text
from verimind.core.config import Settings, get_settings
from verimind.core.enums import RoleName, RunMode
from verimind.core.errors import InputError, InputTooLongError, VeriMindError
from verimind.core.logging import get_logger
from verimind.core.schemas import (
    Claim,
    DetectionReport,
    FactCheckReport,
    LlmCallLog,
    RolePosition,
    RunMetrics,
)
from verimind.models.base import CallBudget, LLMClient, LlmUsage, build_client
from verimind.models.cache import LLMCache
from verimind.models.mock_provider import RuleBasedOfflineClient
from verimind.storage.repository import RunRepository
from verimind.verification.consistency import conflicts_by_claim, find_internal_conflicts
from verimind.verification.rules import (
    AdjudicationContext,
    evaluate_all,
    internal_evidence_for_conflicts,
)

PIPELINE_VERSION = "0.3.0"

#: 阶段 3 仍未实现外部检索。这里显式写死，避免"配置里开了检索"被误当成真的查过了。
#: 阶段 4 实现检索后改成 True。
RETRIEVAL_AVAILABLE = False

logger = get_logger("core.pipeline")


@dataclass(frozen=True)
class ProgressEvent:
    """给界面看的进度事件。"""

    stage: str
    ratio: float
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def _notify(callback: ProgressCallback | None, stage: str, ratio: float, detail: str = "") -> None:
    if callback is not None:
        callback(ProgressEvent(stage=stage, ratio=ratio, detail=detail))


# ==========================================================================
# 主张合并（模型结果 + 规则结果）
# ==========================================================================
def _spans_overlap(first, second) -> bool:
    return first.start < second.end and second.start < first.end


def _renumber(claims: list[Claim]) -> list[Claim]:
    """重新编号，保证 claim_id 连续且唯一（合并两个引擎的结果后必须做）。"""
    return [
        claim.model_copy(update={"claim_id": f"C{index:03d}"})
        for index, claim in enumerate(claims, start=1)
    ]


def _merge_claims(
    primary: list[Claim], secondary: list[Claim], max_claims: int
) -> tuple[ExtractionResult, list[str], list[str]]:
    """合并两个引擎的主张。

    规则：同类型且原文区间重叠 → 视为同一条（保留 primary 的版本）。
    返回（合并后的结果、说明、只有 secondary 发现的主张片段）。
    """
    notes: list[str] = []
    merged: list[Claim] = list(primary)
    added: list[str] = []
    for candidate in secondary:
        duplicate = any(
            candidate.claim_type is existing.claim_type
            and _spans_overlap(candidate.source_span, existing.source_span)
            for existing in merged
        )
        if duplicate:
            continue
        merged.append(candidate)
        added.append(candidate.source_span.text)

    merged.sort(key=lambda item: (PRIORITY_RANK.get(item.priority, 9), item.source_span.start))

    # 去重（同类型 + 同一片段文本）
    unique: list[Claim] = []
    seen: set[tuple[str, str]] = set()
    for claim in merged:
        key = (str(claim.claim_type), claim.source_span.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(claim)

    total = len(unique)
    truncated = total > max_claims
    selected = unique[:max_claims]
    if truncated:
        notes.append(f"两个引擎合计识别出 {total} 条主张，按优先级保留前 {max_claims} 条。")

    kept_texts = {claim.source_span.text for claim in selected}
    secondary_only = [text for text in added if text in kept_texts]
    return (
        ExtractionResult(
            claims=_renumber(selected),
            total_detected=total,
            truncated=truncated,
            dropped_unlocatable=0,
            notes=[],
        ),
        notes,
        secondary_only,
    )


# ==========================================================================
# 报告辅助
# ==========================================================================
def _config_summary(
    *, primary: LLMClient, rules: LLMClient, fact_checker_engine: LLMClient | None
) -> dict[str, dict[str, str]]:
    """报告里记录的"谁在干活"。绝不含密钥。"""
    def describe(client: LLMClient) -> dict[str, str]:
        return {
            "provider": client.provider,
            "model": client.model,
            "simulated": "是（非真实大模型）" if client.is_simulated else "否",
        }

    pending = {"provider": "未启用", "model": "阶段 4/5 实现"}
    return {
        "claim_extractor": describe(primary),
        "rule_baseline": describe(rules),
        "fact_checker": describe(fact_checker_engine) if fact_checker_engine else dict(pending),
        "evidence_checker": dict(pending),
        "corrector": dict(pending),
        "independent_reviewer": dict(pending),
    }


def _build_limitations(
    *,
    primary: LLMClient,
    extraction: ExtractionResult,
    max_claims: int,
    has_conflicts: bool,
    degraded_reason: str | None,
    cache_hits: int,
    fact_check_note: str | None,
) -> list[str]:
    """如实列出本次检测的能力边界。这段文字会原样显示在报告里。"""
    if primary.is_simulated:
        engine_note = (
            f"本次主张拆解由「{primary.engine_id}」完成——这是**规则引擎，不是大模型**："
            "它按正则与关键词找线索，不会理解语义，可能遗漏主张或误判边界。"
        )
    else:
        engine_note = (
            f"本次主张拆解由真实大模型「{primary.engine_id}」完成，输出已通过数据结构校验；"
            "同时用规则引擎做了交叉比对，分歧见「引擎交叉比对」。"
        )
    limitations: list[str] = [
        engine_note,
        (
            "本次**未接入任何外部文献检索**，因此报告不会出现「已核实」，"
            "也不会因外部证据出现「存在矛盾」；需要外部证据的主张一律标记为「证据不足」，"
            "这不代表内容有误（阶段 4 将接入 Crossref / OpenAlex / arXiv）。"
        ),
        "系统未读取任何论文的摘要或全文，因此不对「某文献是否支持该结论」作出判断。",
        "大模型提出的「疑似问题类型」只是怀疑方向，**未经外部证据证实**，不能当作结论。",
        "报告只给出有明确含义的状态与定性严重度，**不包含概率化的「幻觉分数」**（系统未做概率校准）。",
    ]

    if degraded_reason:
        limitations.insert(
            0,
            "⚠️ **本次已降级为离线规则引擎**：大模型调用未能完成，"
            f"原因：{degraded_reason}。报告中的主张与结论均来自规则引擎，不代表模型判断。",
        )
    if cache_hits:
        limitations.append(
            f"本次有 {cache_hits} 次模型调用命中了本地缓存（未产生费用），结果与首次调用一致。"
        )
    if fact_check_note:
        limitations.append(fact_check_note)
    if extraction.truncated:
        limitations.append(
            f"文本中共识别出 {extraction.total_detected} 条候选主张，"
            f"按优先级只保留了前 {max_claims} 条（可调整 VERIMIND_MAX_CLAIMS）。"
        )
    if extraction.dropped_unlocatable:
        limitations.append(
            f"有 {extraction.dropped_unlocatable} 条候选主张因为在原文中无法精确定位而被丢弃"
            "（系统宁可丢一条，也不给一个错误的高亮位置）。"
        )
    if not extraction.claims:
        limitations.append(
            "未能从文本中识别出可核验的事实主张。如果这段文字里确实包含具体事实"
            "（文献、年份、数据、机构结论），那说明拆解环节漏检了。"
        )
    if has_conflicts:
        limitations.append(
            "文内矛盾的检测基于「同一 DOI 不同年份 / 同一指标不同数值」的规则匹配，"
            "属于启发式方法，可能存在漏检或误配。"
        )
    return limitations


def _apply_assessments(
    report: DetectionReport, assessment_report: FactCheckReport, engine: LLMClient
) -> list[str]:
    """把事实审查员的结论写进各主张（只作为"角色意见"，不改状态）。"""
    notes: list[str] = []
    by_id = {item.claim_id: item for item in assessment_report.assessments}

    unknown = [claim_id for claim_id in by_id if report.claim_by_id(claim_id) is None]
    if unknown:
        notes.append(f"审查员返回了不存在的主张编号（已忽略）：{'、'.join(sorted(unknown))}")

    missing: list[str] = []
    for claim in report.claims:
        result = report.result_by_claim(claim.claim_id)
        if result is None:
            continue
        assessment = by_id.get(claim.claim_id)
        if assessment is None:
            missing.append(claim.claim_id)
            continue
        result.role_positions.append(
            RolePosition(
                role=RoleName.FACT_CHECKER,
                provider=engine.provider,
                model=engine.model,
                stance=assessment.stance,
                conclusion=assessment.conclusion,
                reasoning=assessment.reasoning,
            )
        )
        suspected = set(assessment.suspected_problem_types)
        confirmed = set(result.problem_types)
        both = suspected & confirmed
        only_suspected = suspected - confirmed
        if both:
            notes.append(
                f"{claim.claim_id}：模型与规则层都指向「{'、'.join(p.value for p in both)}」。"
            )
        if only_suspected:
            notes.append(
                f"{claim.claim_id}：模型怀疑「{'、'.join(p.value for p in only_suspected)}」，"
                "但规则层未确认——本阶段没有外部证据，规则层不据此判定为错误。"
            )
    if missing:
        notes.append(f"审查员未对以下主张给出结论：{'、'.join(missing)}")

    notes.extend(f"审查员补充说明：{item}" for item in assessment_report.notes)
    return notes


def _make_recorder(repository: RunRepository | None) -> Callable[[LlmCallLog], None] | None:
    if repository is None:
        return None

    def record(call: LlmCallLog) -> None:
        try:
            repository.save_llm_call(call)
        except Exception as exc:  # noqa: BLE001 - 日志写失败不能影响检测
            logger.warning("写入模型调用日志失败：%s", exc)

    return record


# ==========================================================================
# 主流程
# ==========================================================================
def run_detection(
    text: str,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    progress: ProgressCallback | None = None,
    repository: RunRepository | None = None,
) -> DetectionReport:
    """执行一次完整检测，返回可直接落盘/展示的 `DetectionReport`。"""
    started = time.perf_counter()
    settings = settings or get_settings()

    # ---------------- ① 校验输入 ----------------
    raw = (text or "").strip()
    if not raw:
        raise InputError("待核验文本为空，请先粘贴或输入内容。")
    if len(raw) > settings.max_input_chars:
        raise InputTooLongError(len(raw), settings.max_input_chars)
    run_id = uuid.uuid4().hex
    _notify(progress, "① 校验输入", 0.05, f"{len(raw)} 字")

    # ---------------- 准备引擎与存储 ----------------
    repo = repository
    if repo is None and settings.save_runs:
        try:
            repo = RunRepository(settings.sqlite_path)
        except VeriMindError as exc:
            logger.warning("本地数据库不可用，本次不落盘：%s", exc)
            repo = None

    rules_engine = RuleBasedOfflineClient()
    degraded_reason: str | None = None
    primary: LLMClient = rules_engine
    budget = CallBudget(limit=settings.llm_max_calls_per_run)
    cache = LLMCache(settings.data_dir / "cache" / "llm", enabled=settings.llm_cache_enabled)
    usages: list[LlmUsage] = []

    if client is not None:
        primary = client
        if not client.is_simulated:
            budget = CallBudget(limit=settings.llm_max_calls_per_run)
    else:
        try:
            primary = build_client(
                settings,
                cache=cache,
                budget=budget,
                recorder=_make_recorder(repo),
                run_id=run_id,
            )
        except VeriMindError as exc:
            # 配置问题（例如选了 provider 却没填密钥）：不静默降级，直接让界面说清楚
            raise exc

    run_mode = RunMode.MOCK if primary.is_simulated else settings.run_mode

    report = DetectionReport.new(
        run_id=run_id,
        input_text=raw,
        run_mode=run_mode,
        app_version=settings.app_version,
        pipeline_version=PIPELINE_VERSION,
        llm_config_summary={},
    )

    # ---------------- ② 规则引擎拆解（基线，永远执行） ----------------
    _notify(progress, "② 规则引擎拆解（基线）", 0.15)
    # 注意：规则引擎的 usage 不计入 usages —— 它不是大模型调用，不能算进 token 与成本
    rules_result, _rules_usage = rules_engine.extract_claims(
        raw, max_claims=settings.max_claims
    )
    _notify(
        progress, "② 规则引擎拆解（基线）", 0.22, f"规则引擎识别出 {rules_result.count} 条主张"
    )

    merged = rules_result
    cross_check: list[str] = []
    fact_check_note: str | None = None

    # ---------------- ③ 大模型拆解 ----------------
    if not primary.is_simulated:
        _notify(progress, "③ 大模型拆解", 0.28, f"调用 {primary.engine_id}")
        try:
            model_result, usage = primary.extract_claims(raw, max_claims=settings.max_claims)
            usages.append(usage)
        except VeriMindError as exc:
            degraded_reason = str(exc)
            logger.warning("大模型拆解失败，降级为规则引擎：%s", degraded_reason)
            primary = rules_engine
            run_mode = RunMode.MOCK
        else:
            _notify(
                progress, "③ 大模型拆解", 0.38, f"模型识别出 {model_result.count} 条主张"
            )
            merged, merge_notes, secondary_only = _merge_claims(
                model_result.claims, rules_result.claims, settings.max_claims
            )
            cross_check.append(
                f"规则基线识别出 {rules_result.count} 条主张，模型识别出 {model_result.count} 条；"
                f"合并去重后共 {len(merged.claims)} 条。"
            )
            if secondary_only:
                preview = "、".join(f"「{text}」" for text in secondary_only[:5])
                cross_check.append(
                    f"以下主张**只有规则引擎发现**、模型未提及，已补入报告：{preview}"
                    + ("（仅显示前 5 条）" if len(secondary_only) > 5 else "")
                )
            model_only = [
                claim
                for claim in model_result.claims
                if not any(
                    claim.claim_type is rule_claim.claim_type
                    and _spans_overlap(claim.source_span, rule_claim.source_span)
                    for rule_claim in rules_result.claims
                )
            ]
            if model_only:
                preview = "、".join(
                    f"「{claim.source_span.text[:18]}」" for claim in model_only[:5]
                )
                cross_check.append(
                    f"以下主张**只有模型发现**、规则引擎未识别：{preview}"
                    + ("（仅显示前 5 条）" if len(model_only) > 5 else "")
                )
            cross_check.extend(merge_notes)
            report.limitations.extend(model_result.notes)

    report.claims = merged.claims
    # 降级后必须把运行模式改回「离线模拟」——报告已经建好了，所以要显式回写，
    # 否则界面会显示"单模型多角色模拟"，而实际上模型一次都没成功调用（这就是在骗人）。
    report.run_mode = run_mode

    # ---------------- ④ 文内一致性检查 ----------------
    _notify(progress, "④ 文内一致性检查", 0.45)
    analysis = analyze_text(raw)
    conflicts = find_internal_conflicts(raw, report.claims, analysis=analysis)
    evidence, evidence_ids_by_claim = internal_evidence_for_conflicts(conflicts)
    report.evidence = evidence
    _notify(
        progress,
        "④ 文内一致性检查",
        0.55,
        f"发现 {len(conflicts)} 处文内矛盾" if conflicts else "未发现文内自相矛盾",
    )

    # ---------------- ⑤ 规则裁决 ----------------
    _notify(progress, "⑤ 规则裁决", 0.60)
    context = AdjudicationContext(
        text=raw,
        analysis=analysis,
        conflicts_by_claim=conflicts_by_claim(conflicts),
        evidence_ids_by_claim=evidence_ids_by_claim,
        retrieval_enabled=RETRIEVAL_AVAILABLE and settings.retrieve_enabled,
        pipeline_stage="阶段 3",
    )
    report.verification_results = evaluate_all(report.claims, context)
    _notify(
        progress, "⑤ 规则裁决", 0.72, f"完成 {len(report.verification_results)} 条主张的裁决"
    )

    # ---------------- ⑥ 事实审查员（角色 A） ----------------
    if not primary.is_simulated and report.claims:
        _notify(progress, "⑥ 事实审查员复核", 0.78, f"调用 {primary.engine_id}")
        try:
            assessment_report, check_usage = primary.assess_claims(raw, report.claims)
            usages.append(check_usage)
        except VeriMindError as exc:
            fact_check_note = f"事实审查员（角色 A）未能完成，原因：{exc}"
            logger.warning(fact_check_note)
        else:
            cross_check.extend(_apply_assessments(report, assessment_report, primary))
            _notify(progress, "⑥ 事实审查员复核", 0.88, "角色结论已写入报告")
    elif primary.is_simulated:
        cross_check.append(
            "当前为离线规则引擎模式：没有大模型角色参与，因此报告中不存在"
            "「不同模型之间的分歧」——也不会假装存在。"
        )

    # ---------------- ⑦ 生成报告 ----------------
    _notify(progress, "⑦ 生成报告", 0.92)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    prompt_tokens = sum(usage.prompt_tokens or 0 for usage in usages)
    completion_tokens = sum(usage.completion_tokens or 0 for usage in usages)
    cache_hits = sum(1 for usage in usages if usage.cache_hit)
    estimated_cost = settings.estimate_cost(prompt_tokens, completion_tokens)

    if cache_hits and estimated_cost is not None:
        cost_note = (
            f"按配置单价估算（其中 {cache_hits} 次命中缓存未产生费用）。"
            "单价来自 .env 的 VERIMIND_LLM_PRICE_* 配置，请自行核对是否为最新官方价格。"
        )
    elif estimated_cost is not None:
        cost_note = "按 .env 中配置的单价估算，请自行核对是否为最新官方价格。"
    elif primary.is_simulated:
        cost_note = (
            "离线规则引擎不调用任何付费接口，因此不存在费用。"
            "接入真实模型后，这里会显示按实际 token 统计的成本。"
        )
    else:
        cost_note = (
            f"未计算成本：没有配置单价。本次实际消耗 prompt {prompt_tokens} / "
            f"completion {completion_tokens} tokens；"
            "如需成本估算，请在 .env 中填写 VERIMIND_LLM_PRICE_PROMPT_PER_MTOK 与 "
            "VERIMIND_LLM_PRICE_COMPLETION_PER_MTOK（按供应商官网单价）。"
        )

    report.metrics = RunMetrics(
        latency_ms=elapsed_ms,
        llm_calls=sum(1 for usage in usages if not usage.cache_hit),
        cache_hits=cache_hits,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost=estimated_cost,
        cost_note=cost_note,
    )
    report.cross_check_notes = cross_check
    report.limitations = _build_limitations(
        primary=primary,
        extraction=merged,
        max_claims=settings.max_claims,
        has_conflicts=bool(conflicts),
        degraded_reason=degraded_reason,
        cache_hits=cache_hits,
        fact_check_note=fact_check_note,
    ) + report.limitations
    report.llm_config_summary = _config_summary(
        primary=primary,
        rules=rules_engine,
        fact_checker_engine=None if primary.is_simulated else primary,
    )
    report.recompute()

    # ---------------- 落盘 ----------------
    if repo is not None:
        try:
            repo.save(report)
        except VeriMindError as exc:
            report.limitations.append(f"报告未能保存到本地数据库：{exc}")

    _notify(progress, "完成", 1.0, f"耗时 {elapsed_ms} ms")
    return report


__all__ = [
    "PIPELINE_VERSION",
    "RETRIEVAL_AVAILABLE",
    "ProgressCallback",
    "ProgressEvent",
    "RoleName",
    "run_detection",
]
