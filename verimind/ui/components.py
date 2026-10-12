"""报告展示组件：原文高亮、统计卡片、主张卡片、证据条目。

把展示逻辑集中在这里，页面文件就只剩"取数据 → 调用组件"，
不至于让 `pages/2_检测报告.py` 变成几百行的渲染脚本。

三条展示纪律（写在代码里，防止以后被绕过）：
    1. **统计数字只能来自 `report.stats`**（由 `DetectionReport.recompute()` 算出），
       页面不允许自己数、更不允许手写。
    2. **原文永远转义后展示**：`html.escape` 之后再拼 HTML，
       用户文本里的 `<script>` 之类不会变成可执行内容。
    3. **没有的东西就明说**：没有证据、没有角色分歧、检索被跳过，都要如实写出来。
"""

from __future__ import annotations

import html

import streamlit as st

from verimind.core.enums import (
    STATUS_DISPLAY_ORDER,
    Severity,
    VerificationStatus,
)
from verimind.core.schemas import Claim, DetectionReport, Evidence, VerificationResult
from verimind.ui.theme import (
    STATUS_STYLE,
    badge,
    severity_badge_html,
    status_badge_html,
)
from verimind.verification.rules import RULE_LIBRARY

_MARK_BG = "#fff3b0"
_MARK_BORDER = "#e0b400"


# ==========================================================================
# 原文高亮
# ==========================================================================
def highlight_html(text: str, spans: list[tuple[int, int]] | None = None) -> str:
    """把原文渲染成 HTML，并高亮给定区间（自动转义、自动合并重叠区间）。"""
    ranges = sorted((s, e) for s, e in (spans or []) if 0 <= s < e <= len(text))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(
            f'<mark style="background:{_MARK_BG};border-bottom:2px solid {_MARK_BORDER};'
            f'padding:0 2px;border-radius:3px;">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_text_panel(text: str, spans: list[tuple[int, int]] | None = None) -> None:
    """带高亮的文本面板（用于原文与证据片段）。"""
    st.markdown(
        '<div style="background:#fbfcfd;border:1px solid #e3e7ec;border-radius:6px;'
        "padding:12px 14px;line-height:1.95;white-space:pre-wrap;font-size:0.95rem;"
        f'color:#1c1f23;">{highlight_html(text, spans)}</div>',
        unsafe_allow_html=True,
    )


def render_sentence_with_highlight(claim: Claim) -> None:
    """显示主张所在整句，并高亮该主张的精确片段。"""
    parent = claim.parent_sentence_span
    if parent is None:
        render_text_panel(claim.source_span.text)
        return
    relative = claim.source_span.relative_to(parent)
    spans = [(relative.start, relative.end)] if relative is not None else []
    render_text_panel(parent.text, spans)


# ==========================================================================
# 统计卡片
# ==========================================================================
def render_stat_cards(report: DetectionReport) -> None:
    """顶部统计。所有数字都来自 report.stats（自动计算，不可手工填写）。"""
    stats = report.stats
    unresolved = stats.status_counts.get(VerificationStatus.INSUFFICIENT.value, 0) + stats.status_counts.get(
        VerificationStatus.UNVERIFIABLE.value, 0
    )
    columns = st.columns(4)
    columns[0].metric("主张总数", stats.total_claims)
    columns[1].metric("已判定有问题", stats.suspicious_claims, help="存在矛盾 / 部分支持 / 来源冲突")
    columns[2].metric("需关注", stats.flagged_claims, help="含尚未证实的疑点与证据缺口")
    columns[3].metric("尚未核验", unresolved, help="证据不足 / 暂无法核验——不等于有错")

    counts = "".join(
        badge(
            f"{STATUS_STYLE[status].icon} {status.value} {stats.status_counts.get(status.value, 0)}",
            fg=STATUS_STYLE[status].fg,
            bg=STATUS_STYLE[status].bg,
            border=STATUS_STYLE[status].border,
        )
        for status in STATUS_DISPLAY_ORDER
    )
    st.markdown(counts, unsafe_allow_html=True)

    if stats.problem_counts:
        problems = "、".join(f"{name}（{count}）" for name, count in stats.problem_counts.items())
        st.caption(f"问题类型分布：{problems}")
    if report.metrics.latency_ms is not None:
        st.caption(
            f"耗时 {report.metrics.latency_ms} ms ｜ 大模型调用 {report.metrics.llm_calls} 次 ｜ "
            f"引擎耗时说明见下方「运行信息」"
        )


# ==========================================================================
# 证据
# ==========================================================================
def render_evidence_item(evidence: Evidence) -> None:
    """一条证据。外部证据给可点击链接；原文内部证据给原文片段。"""
    if evidence.is_internal:
        st.markdown(
            badge(
                f"原文内部 · {evidence.level.value}",
                fg="#5f6b7a",
                bg="#f4f5f7",
                border="#d5d9df",
            ),
            unsafe_allow_html=True,
        )
        render_text_panel(evidence.snippet or "", None)
        st.caption(f"关联关系：{evidence.relation.value}｜{evidence.relation_rationale}")
        return

    aspect = evidence.judged_aspect.value if evidence.judged_aspect else "未做判断"
    st.markdown(
        badge(
            f"{evidence.source.value} · {evidence.level.value}",
            fg="#3a5a8c",
            bg="#eef3fb",
            border="#b9cbe8",
        )
        + badge(
            f"判断层面：{aspect}",
            fg="#5f6b7a",
            bg="#f4f5f7",
            border="#d5d9df",
        )
        + (
            badge(
                f"关系：{evidence.relation.value}",
                fg="#b42318" if evidence.relation is EvidenceRelation.REFUTES else "#1a7f37",
                bg="#fdecea" if evidence.relation is EvidenceRelation.REFUTES else "#e9f7ee",
                border="#f3b3ad" if evidence.relation is EvidenceRelation.REFUTES else "#a7e0b8",
            )
            if evidence.relation is not EvidenceRelation.UNKNOWN
            else ""
        ),
        unsafe_allow_html=True,
    )
    if evidence.title:
        year = f"（{evidence.year}）" if evidence.year else ""
        st.markdown(f"**{evidence.title}**{year}")
    if evidence.authors:
        st.caption("作者：" + "、".join(evidence.authors[:5]) + ("…" if len(evidence.authors) > 5 else ""))
    if evidence.url:
        st.link_button(f"打开来源：{evidence.url}", evidence.url)
    if evidence.snippet:
        label = "引用原句" if evidence.judged_aspect is JudgedAspect.CONTENT else "证据片段"
        st.caption(f"{label}：{evidence.snippet}")
    if evidence.relation_rationale:
        st.caption(f"判断理由：{evidence.relation_rationale}")
    if evidence.query:
        st.caption(f"检索式：`{evidence.query}`")
    if evidence.raw_ref:
        st.caption(f"原始响应留档：`data/{evidence.raw_ref}`")


def render_search_trace(result: VerificationResult) -> None:
    """检索痕迹：如实展示"系统查了什么、还是根本没查"。"""
    records = result.search_trace.records
    if not records:
        st.caption("检索痕迹：无记录。")
        return
    lines = []
    for record in records:
        source = record.source.value if record.source else "未指定来源"
        detail = f"｜{record.detail}" if record.detail else ""
        lines.append(f"- {record.status.value}｜查询：`{record.query}`｜来源：{source}{detail}")
    st.markdown("检索痕迹：\n" + "\n".join(lines))


# ==========================================================================
# 主张卡片
# ==========================================================================
def _problem_type_label(result: VerificationResult) -> str:
    if result.is_suspicious:
        return "已确认的问题类型"
    return "疑似问题类型（尚未证实）"


def render_claim_card(
    report: DetectionReport,
    claim: Claim,
    result: VerificationResult | None,
    *,
    expanded: bool = False,
) -> None:
    """一条主张的完整卡片：定位 → 主张 → 问题 → 理由 → 证据 → 还缺什么。"""
    status = result.status if result else VerificationStatus.UNVERIFIABLE
    style = STATUS_STYLE[status]
    label = (
        f"{style.icon} {claim.claim_id} ｜ {claim.claim_type.value} ｜ "
        f"{status.value} ｜ {claim.source_span.text[:36]}"
    )

    with st.expander(label, expanded=expanded):
        st.markdown(
            status_badge_html(status)
            + severity_badge_html(result.severity if result else Severity.NONE)
            + badge(
                f"优先级 {claim.priority.value}",
                fg="#3a5a8c",
                bg="#eef3fb",
                border="#b9cbe8",
            )
            + badge(
                f"{claim.checkability.value}",
                fg="#5f6b7a",
                bg="#f4f5f7",
                border="#d5d9df",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("**原文定位**（高亮部分为该主张）")
        render_sentence_with_highlight(claim)

        st.markdown(f"**规范化主张**：{claim.normalized_claim}")

        if claim.verification_questions:
            st.markdown("**要核验这条主张，必须回答**")
            for question in claim.verification_questions:
                st.markdown(f"- {question}")

        if result is None:
            st.warning("该主张没有核验结果（可能是流水线被中断）。")
            return

        st.markdown(f"**判断理由**：{result.rationale}")
        if result.adjudication_rule:
            rule_ids = [part.strip() for part in result.adjudication_rule.split("、") if part.strip()]
            explanations = [f"`{rule_id}` {RULE_LIBRARY.get(rule_id, '（未登记的规则）')}" for rule_id in rule_ids]
            st.markdown("**命中的规则**\n\n" + "\n\n".join(explanations))

        if result.problem_types:
            st.markdown(f"**{_problem_type_label(result)}**")
            for problem in result.problem_types:
                st.markdown(f"- {problem.value}")

        evidence_ids = list(
            dict.fromkeys(
                result.refuting_evidence_ids
                + result.supporting_evidence_ids
                + result.related_evidence_ids
            )
        )
        if evidence_ids:
            st.markdown("**相关证据**")
            for evidence_id in evidence_ids:
                evidence = report.evidence_by_id(evidence_id)
                if evidence is not None:
                    st.markdown(f"`{evidence.evidence_id}`")
                    render_evidence_item(evidence)
        else:
            st.caption("相关证据：无。本阶段尚未接入外部检索，因此没有外部证据。")

        if result.missing_evidence:
            st.markdown("**还缺什么证据才能判断**")
            for item in result.missing_evidence:
                st.markdown(f"- {item}")

        render_search_trace(result)

        if result.role_positions:
            st.markdown("**各角色结论**")
            for position in result.role_positions:
                st.markdown(
                    f"- {position.role.value}（{position.provider}/{position.model}）："
                    f"{position.stance.value}｜{position.conclusion}"
                )
        else:
            st.caption(
                "各角色结论：无。当前为离线规则引擎模式，没有大模型角色参与——"
                "所以这里既没有「分歧」也没有「协同」，不会假装有。"
            )

        st.caption(
            f"拆解引擎：{claim.extracted_by or '未知'}｜拆解置信度：{claim.extraction_confidence.value}"
        )


# ==========================================================================
# 报告头部 / 运行信息 / 局限
# ==========================================================================
def render_report_header(report: DetectionReport) -> None:
    local_time = report.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"**运行 ID**：`{report.run_id}`　｜　**时间**：{local_time}（本地时区）　｜　"
        f"**流水线版本**：{report.pipeline_version}"
    )
    from verimind.ui.theme import RUN_MODE_DESCRIPTION

    mode = report.run_mode
    icon, color, description = RUN_MODE_DESCRIPTION[mode]
    st.markdown(
        f'<div style="border-left:5px solid {color};background:{color}14;padding:8px 12px;'
        f'border-radius:6px;margin:6px 0;">'
        f'<b style="color:{color}">{icon} 本次运行模式：{mode.value}</b><br>'
        f'<span style="font-size:0.88rem;color:#333;">{description}</span></div>',
        unsafe_allow_html=True,
    )

    extractor = report.llm_config_summary.get("claim_extractor", {})
    if extractor:
        engine = f"{extractor.get('provider', '')}/{extractor.get('model', '')}"
        st.caption(
            f"主张拆解引擎：**{engine}** ｜ 是否模拟：{extractor.get('simulated', '未知')}"
        )


def render_cross_check(report: DetectionReport) -> None:
    """引擎交叉比对：把"规则引擎和模型谁发现了什么、谁漏了什么"摆到台面上。

    这一节存在的意义是**暴露分歧**，而不是掩盖分歧。
    一个小系统如果连自己的两个部件意见不一致都不敢说，那它给出的"结论"也没什么可信度。
    """
    st.markdown("#### 🔍 引擎交叉比对（规则引擎 vs 大模型）")
    if not report.cross_check_notes:
        st.caption("本次没有可展示的交叉比对信息。")
        return
    st.caption(
        "规则引擎与真实模型分别独立拆解同一段文本，下面是两者的差异与分歧全记录"
        "（不做取舍、不做美化）。"
    )
    for note in report.cross_check_notes:
        st.markdown(f"- {note}")


def render_run_info(report: DetectionReport) -> None:
    """运行信息：谁在干活、花了多少时间、是否产生费用。"""
    with st.expander("运行信息（引擎、耗时、成本）", expanded=False):
        rows = [
            f"| {role} | {config.get('provider', '')} | {config.get('model', '')} | "
            f"{config.get('simulated', '')} |"
            for role, config in report.llm_config_summary.items()
        ]
        st.markdown(
            "| 角色 | provider | model | 是否模拟 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
        )
        st.markdown(
            f"- 端到端耗时：{report.metrics.latency_ms} ms\n"
            f"- 大模型调用次数：{report.metrics.llm_calls}（计费）"
            f" ／ {report.metrics.cache_hits}（命中缓存，未计费）\n"
            f"- 外部检索：{report.metrics.retrieval_queries} 个查询 ／ "
            f"{report.metrics.retrieval_requests} 次真实请求 ／ "
            f"{report.metrics.retrieval_cache_hits} 次命中缓存\n"
            f"- token 用量：prompt {report.metrics.prompt_tokens} / completion {report.metrics.completion_tokens}\n"
            f"- 成本：{report.metrics.estimated_cost if report.metrics.estimated_cost is not None else '—'}\n"
            f"- 成本说明：{report.metrics.cost_note}"
        )
        st.caption(f"输入文本 SHA-256：{report.input_sha256}")


def render_limitations(report: DetectionReport) -> None:
    """系统局限：必须显著展示，不能藏在角落里。"""
    if not report.limitations:
        return
    st.markdown("#### ⚠️ 系统局限（本次检测不能做什么）")
    for item in report.limitations:
        st.markdown(f"- {item}")


def render_disclaimer(report: DetectionReport) -> None:
    st.caption(report.disclaimer)
