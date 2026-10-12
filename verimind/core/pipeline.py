"""流水线编排：把「文本 → 报告」的全过程串起来。

阶段 2 实际执行的步骤
--------------------
    ① 校验输入
    ② 拆解原子事实主张        （离线规则引擎 / 阶段 3 换成真实模型）
    ③ 文内一致性检查          （唯一无需外部证据就能给出"来源冲突"的来源）
    ④ 规则裁决                （6 种状态 + 问题类型 + 缺失证据清单）
    ⑤ 生成报告并落盘

阶段 4/5 会在这里插入"外部检索 → 证据匹配"和"修正 → 独立复核"两步。
因为每一步都是纯函数、出入参都是 Pydantic 模型，插入新步骤不会影响已有代码。

两个设计要点
------------
1. **进度回调**：界面通过 `progress` 回调显示每一步的进度，业务层不认识 Streamlit。
2. **失败不静默**：任何一步失败都会抛出带中文说明的异常；能降级的（例如报告落盘失败）
   会记录到 `report.limitations` 里，而不是假装成功。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

from verimind.claims.extractor import ExtractionResult
from verimind.claims.preprocess import analyze_text
from verimind.core.config import Settings, get_settings
from verimind.core.enums import RoleName, RunMode
from verimind.core.errors import InputError, InputTooLongError, VeriMindError
from verimind.core.schemas import DetectionReport, RunMetrics
from verimind.models.base import LLMClient, build_client
from verimind.storage.repository import RunRepository
from verimind.verification.consistency import conflicts_by_claim, find_internal_conflicts
from verimind.verification.rules import (
    AdjudicationContext,
    evaluate_all,
    internal_evidence_for_conflicts,
)

PIPELINE_VERSION = "0.2.0"

#: 阶段 2 尚未实现外部检索。这里显式写死，避免"配置里开了检索"被误当成真的查过了。
#: 阶段 4 实现检索后改成 True。
RETRIEVAL_AVAILABLE = False


@dataclass(frozen=True)
class ProgressEvent:
    """给界面看的进度事件。"""

    stage: str
    ratio: float
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def _notify(
    callback: ProgressCallback | None, stage: str, ratio: float, detail: str = ""
) -> None:
    if callback is not None:
        callback(ProgressEvent(stage=stage, ratio=ratio, detail=detail))


def _config_summary(client: LLMClient) -> dict[str, dict[str, str]]:
    """报告里记录的"谁在干活"。绝不含密钥。"""
    active = {
        "provider": client.provider,
        "model": client.model,
        "simulated": "是（非真实大模型）" if client.is_simulated else "否",
    }
    pending = {"provider": "未启用", "model": "阶段 3 实现"}
    return {
        "claim_extractor": dict(active),
        "fact_checker": dict(pending),
        "evidence_checker": dict(pending),
        "corrector": dict(pending),
        "independent_reviewer": dict(pending),
    }


def _build_limitations(
    *,
    client: LLMClient,
    extraction: ExtractionResult,
    max_claims: int,
    has_conflicts: bool,
) -> list[str]:
    """如实列出本次检测的能力边界。这段文字会原样显示在报告里。"""
    limitations: list[str] = [
        (
            f"本次主张拆解由「{client.engine_id}」完成。"
            + (
                "这是**规则引擎，不是大模型**：它按正则与关键词找线索，不会理解语义，"
                "可能遗漏主张或误判边界（阶段 3 将接入大模型，并与规则结果交叉比对）。"
                if client.is_simulated
                else "由真实大模型完成，并已通过数据结构校验。"
            )
        ),
        (
            "本次**未接入任何外部文献检索**，因此报告不会出现「已核实」，"
            "也不会因外部证据出现「存在矛盾」；需要外部证据的主张一律标记为「证据不足」，"
            "这不代表内容有误（阶段 4 将接入 Crossref / OpenAlex / arXiv）。"
        ),
        "系统未读取任何论文的摘要或全文，因此不对「某文献是否支持该结论」作出判断。",
        "报告只给出有明确含义的状态与定性严重度，**不包含概率化的「幻觉分数」**（系统未做概率校准）。",
    ]

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
            "（文献、年份、数据、机构结论），那么很可能是规则引擎漏检——这是阶段 3 要解决的问题。"
        )
    if has_conflicts:
        limitations.append(
            "文内矛盾的检测基于「同一 DOI 不同年份 / 同一指标不同数值」的规则匹配，"
            "属于启发式方法，可能存在漏检或误配。"
        )
    return limitations


def run_detection(
    text: str,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    progress: ProgressCallback | None = None,
    repository: RunRepository | None = None,
) -> DetectionReport:
    """执行一次完整检测，返回可直接落盘/展示的 `DetectionReport`。

    可能抛出的异常（界面需要分别处理并给出中文提示）：
        InputError / InputTooLongError  输入问题
        LLMError 及其子类               引擎问题（例如阶段 2 配置了真实 provider）
    """
    started = time.perf_counter()
    settings = settings or get_settings()

    # ---------------- ① 校验输入 ----------------
    raw = (text or "").strip()
    if not raw:
        raise InputError("待核验文本为空，请先粘贴或输入内容。")
    if len(raw) > settings.max_input_chars:
        raise InputTooLongError(len(raw), settings.max_input_chars)
    _notify(progress, "① 校验输入", 0.05, f"{len(raw)} 字")

    # ---------------- 准备引擎 ----------------
    engine = client or build_client(settings)
    # 真实模式由客户端身份决定：只要客户端是模拟实现，报告就如实写「离线模拟」
    run_mode = RunMode.MOCK if engine.is_simulated else settings.run_mode

    report = DetectionReport.new(
        run_id=uuid.uuid4().hex,
        input_text=raw,
        run_mode=run_mode,
        app_version=settings.app_version,
        pipeline_version=PIPELINE_VERSION,
        llm_config_summary=_config_summary(engine),
    )

    # ---------------- ② 拆解主张 ----------------
    _notify(progress, "② 拆解原子事实主张", 0.25)
    extraction, usage = engine.extract_claims(raw, max_claims=settings.max_claims)
    report.claims = extraction.claims
    _notify(
        progress,
        "② 拆解原子事实主张",
        0.40,
        f"识别出 {len(extraction.claims)} 条可核验主张",
    )

    # ---------------- ③ 文内一致性检查 ----------------
    _notify(progress, "③ 文内一致性检查", 0.50)
    analysis = analyze_text(raw)
    conflicts = find_internal_conflicts(raw, report.claims, analysis=analysis)
    evidence, evidence_ids_by_claim = internal_evidence_for_conflicts(conflicts)
    report.evidence = evidence
    _notify(
        progress,
        "③ 文内一致性检查",
        0.60,
        f"发现 {len(conflicts)} 处文内矛盾" if conflicts else "未发现文内自相矛盾",
    )

    # ---------------- ④ 规则裁决 ----------------
    _notify(progress, "④ 规则裁决", 0.65)
    context = AdjudicationContext(
        text=raw,
        analysis=analysis,
        conflicts_by_claim=conflicts_by_claim(conflicts),
        evidence_ids_by_claim=evidence_ids_by_claim,
        retrieval_enabled=RETRIEVAL_AVAILABLE and settings.retrieve_enabled,
        pipeline_stage="阶段 2",
    )
    report.verification_results = evaluate_all(report.claims, context)
    _notify(
        progress,
        "④ 规则裁决",
        0.85,
        f"完成 {len(report.verification_results)} 条主张的裁决",
    )

    # ---------------- ⑤ 生成报告 ----------------
    _notify(progress, "⑤ 生成报告", 0.92)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    report.metrics = RunMetrics(
        latency_ms=elapsed_ms,
        # 离线规则引擎不算"大模型调用"。这里如实写 0，而不是凑一个 1，
        # 否则评测页统计"单次检测的模型调用次数"时会虚高。
        llm_calls=0 if engine.is_simulated else 1,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        estimated_cost=None,
        cost_note=(
            "离线规则引擎不调用任何付费接口，因此不存在费用。"
            "接入真实模型后，这里会显示按实际 token 统计的成本。"
            if engine.is_simulated
            else "成本统计将在阶段 3 接入计费单价后提供。"
        ),
    )
    report.limitations = _build_limitations(
        client=engine,
        extraction=extraction,
        max_claims=settings.max_claims,
        has_conflicts=bool(conflicts),
    )
    report.recompute()

    # ---------------- 落盘 ----------------
    if settings.save_runs:
        try:
            repo = repository or RunRepository(settings.sqlite_path)
            repo.save(report)
        except VeriMindError as exc:
            # 存不下不算检测失败，但必须如实写进报告
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
