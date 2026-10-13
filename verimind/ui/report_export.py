"""把检测报告导出成 Markdown（可下载、可归档、可直接贴进材料）。

为什么要导出功能：
    · 比赛要求"结果可复现"——导出的文件本身就是一次运行的证据；
    · 报告页好看，但评委/队友需要能带走的东西；
    · 导出内容与页面**同源**（都来自同一个 DetectionReport 对象），
      所以不会出现"页面说 A、导出文件说 B"。

导出原则（与全项目一致）：
    1. 不确定的内容一律写明"未核实/证据不足"，绝不因为导出了就说得更满；
    2. 每条结论都带上**命中的规则编号**与**证据链接**，方便逐条追问；
    3. 结尾必须带局限性声明与运行信息（含缓存命中、降级说明）。
"""

from __future__ import annotations

from datetime import datetime

from verimind.core.enums import (
    STATUS_DISPLAY_ORDER,
)
from verimind.core.schemas import DetectionReport, VerificationResult


def _status_line(result: VerificationResult) -> str:
    parts = [f"**{result.status.value}**"]
    if result.severity.value != "无":
        parts.append(f"严重度：{result.severity.value}")
    if result.adjudication_rule:
        parts.append(f"命中规则：{result.adjudication_rule}")
    return " ｜ ".join(parts)


def _evidence_lines(report: DetectionReport, ids: list[str]) -> list[str]:
    lines: list[str] = []
    for evidence_id in ids:
        item = report.evidence_by_id(evidence_id)
        if item is None:
            continue
        bits = [f"`{item.evidence_id}`", f"来源：{item.source.value}", f"层级：{item.level.value}"]
        if item.judged_aspect is not None:
            bits.append(f"判断层面：{item.judged_aspect.value}")
        bits.append(f"关系：{item.relation.value}")
        lines.append("  - " + " ｜ ".join(bits))
        if item.title:
            lines.append(f"    - 标题：{item.title}")
        if item.year:
            lines.append(f"    - 年份：{item.year}")
        if item.doi:
            lines.append(f"    - DOI：`{item.doi}`")
        if item.url:
            lines.append(f"    - 链接：{item.url}")
        if item.snippet:
            lines.append(f"    - 片段：{item.snippet[:220]}")
        if item.relation_rationale:
            lines.append(f"    - 判断理由：{item.relation_rationale}")
        if item.raw_ref:
            lines.append(f"    - 原始响应留档：`{item.raw_ref}`")
    return lines


def report_to_markdown(report: DetectionReport) -> str:
    """把一份检测报告渲染成 Markdown 文本。"""
    stats = report.stats
    lines: list[str] = []
    add = lines.append

    created = report.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    add("# 智鉴 AI（VeriMind AI）检测报告")
    add("")
    add(f"- 报告编号：`{report.run_id}`")
    add(f"- 生成时间：{created}")
    add(f"- 运行模式：**{report.run_mode.value}**（引擎：{report.llm_config_summary.get('claim_extractor', {}).get('provider', '—')}）")
    add(f"- 管道版本：{report.pipeline_version}")
    add("")
    add("## 一、结论摘要")
    add("")
    add(f"- 主张总数：**{stats.total_claims}**")
    add(
        f"- 已判定有问题：**{stats.suspicious_claims}**　｜　需关注（含疑点与证据缺口）："
        f"**{stats.flagged_claims}**　｜　证据不足：**{stats.status_counts.get('证据不足', 0)}**"
    )
    add("- 状态分布：" + "　".join(
        f"{status.value} {stats.status_counts.get(status.value, 0)}"
        for status in STATUS_DISPLAY_ORDER
        if stats.status_counts.get(status.value, 0)
    ))
    if stats.corrections_count:
        add(
            f"- 修正建议：{stats.corrections_count} 条（采用 {len(report.applied_correction_ids)} 条"
            f" / 拒绝 {len(report.rejected_correction_ids)} 条）"
        )
    add("")

    add("## 二、原文")
    add("")
    add("```text")
    add(report.input_text)
    add("```")
    add("")

    if report.corrected_text and report.corrected_text != report.input_text:
        add("## 三、修正后全文（仅包含复核通过的修改）")
        add("")
        add("```text")
        add(report.corrected_text)
        add("```")
        add("")

    add("## 四、逐条主张")
    add("")
    for index, claim in enumerate(report.claims, start=1):
        result = report.result_by_claim(claim.claim_id)
        add(f"### {index}. {claim.claim_id}　{claim.source_span.text[:60]}")
        add("")
        add(f"- 原文位置：第 {claim.source_span.start}–{claim.source_span.end} 字符")
        add(f"- 规范化主张：{claim.normalized_claim}")
        add(f"- 主张类型：{claim.claim_type.value}　｜　优先级：{claim.priority.value}")
        if result is None:
            add("- 结论：未裁决")
            add("")
            continue
        add(f"- 结论：{_status_line(result)}")
        if result.problem_types:
            add("- 问题类型：" + "、".join(item.value for item in result.problem_types))
        add(f"- 判断理由：{result.rationale}")
        if result.missing_evidence:
            add("- 还缺什么证据：")
            for item in result.missing_evidence:
                add(f"  - {item}")
        if result.supporting_evidence_ids or result.refuting_evidence_ids or result.related_evidence_ids:
            add("- 相关证据：")
            lines.extend(
                _evidence_lines(
                    report,
                    list(
                        dict.fromkeys(
                            result.supporting_evidence_ids
                            + result.refuting_evidence_ids
                            + result.related_evidence_ids
                        )
                    ),
                )
            )
        add("")

    if report.corrections:
        add("## 五、修正建议与独立复核")
        add("")
        for correction in report.corrections:
            applied = correction.correction_id in report.applied_correction_ids
            add(f"### {correction.correction_id}　{'✅ 已采用' if applied else '❌ 未采用'}")
            add("")
            add(f"- 类型：{correction.change_type.value}")
            add(f"- 原文片段：{correction.original_span.text}")
            add(f"- 修改为：{correction.proposed_text or '（删除该片段）'}")
            add(f"- 修改理由：{correction.rationale}")
            if correction.evidence_ids:
                add(f"- 支撑证据：{'、'.join('`' + item + '`' for item in correction.evidence_ids)}")
            review = next(
                (
                    item
                    for item in report.review_results
                    if item.target_correction_id == correction.correction_id
                ),
                None,
            )
            if review is None:
                add("- 复核：**无复核记录（因此不采用）**")
            else:
                add(f"- 复核结论：**{review.overall.value}**"
                    + (f"（复核者：{review.reviewer.provider}/{review.reviewer.model}）" if review.reviewer else ""))
                for check in review.checks:
                    add(f"  - {'✅' if check.passed else '❌'} {check.check.value}：{check.detail}")
                for item in review.unresolved:
                    add(f"  - 未解决：{item}")
            add("")

    if report.cross_check_notes:
        add("## 六、引擎交叉比对与过程说明")
        add("")
        for note in report.cross_check_notes:
            add(f"- {note}")
        add("")

    if report.limitations:
        add("## 七、系统局限（本次运行适用）")
        add("")
        for item in report.limitations:
            add(f"- {item}")
        add("")

    add("## 八、运行信息")
    add("")
    metrics = report.metrics
    if metrics is not None:
        add(f"- 耗时：{metrics.latency_ms} ms")
        add(f"- 模型调用：{metrics.llm_calls} 次（命中缓存 {metrics.cache_hits} 次）")
        add(f"- token：prompt {metrics.prompt_tokens} + completion {metrics.completion_tokens}")
        add(
            f"- 外部检索：查询 {metrics.retrieval_queries} 次，请求 {metrics.retrieval_requests} 次"
            f"（缓存命中 {metrics.retrieval_cache_hits} 次）"
        )
        if metrics.estimated_cost is not None:
            add(f"- 估算成本：{metrics.estimated_cost}（{metrics.cost_note}）")
        else:
            add("- 估算成本：未计算（未配置单价，因此不给出金额）")
    add("---")
    add("")
    add(
        "本报告由智鉴 AI 自动生成，是**基于可检索到的公开证据作出的辅助判断**，"
        "不是学术结论、法律意见或事实认定的最终依据。"
        "系统无法确知模型内部真实的出错原因；报告中的问题类型是依据可观察证据作出的归因。"
        "当证据不足或无法核验时，系统会明确标注，而不是给出猜测性结论。"
    )
    add("")
    add("> 未核实的部分请以人工复核为准。导出时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "。")
    return "\n".join(lines)


__all__ = ["report_to_markdown"]
