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

from datetime import datetime

import html

import streamlit as st

from verimind.core.enums import (
    STATUS_DISPLAY_ORDER,
    EvidenceRelation,
    RoleStance,
    JudgedAspect,
    ReviewOverall,
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
from verimind.verification.evidence_strength import assess_evidence_strength
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

        render_consensus(result)

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

        render_strength_badge(
            result,
            [item for item in (report.evidence_by_id(i) for i in evidence_ids) if item is not None],
        )

        if result.missing_evidence:
            st.markdown("**还缺什么证据才能判断**")
            for item in result.missing_evidence:
                st.markdown(f"- {item}")

        render_evidence_chain(report, claim, result)

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
def render_correction_section(report: DetectionReport) -> None:
    """修正建议与独立复核：修正前后对比 + 五项检查 + 被拒绝的修改。"""
    st.markdown("#### 修正建议与独立复核")
    if not report.corrections:
        st.caption(
            "本次没有生成修正建议——修正只针对**已被判定有问题**的主张；"
            "「证据不足」与「暂无法核验」都不会触发改写（那会把「还没查清」变成「擅自改写」）。"
        )
        return

    applied = set(report.applied_correction_ids)
    rejected = set(report.rejected_correction_ids)
    st.markdown(
        badge(f"共 {len(report.corrections)} 条", fg="#3a5a8c", bg="#eef3fb", border="#b9cbe8")
        + badge(f"已采用 {len(applied)} 条", fg="#1a7f37", bg="#e9f7ee", border="#a7e0b8")
        + badge(f"被拒绝 {len(rejected)} 条", fg="#b42318", bg="#fdecea", border="#f3b3ad"),
        unsafe_allow_html=True,
    )
    st.caption(
        "只有**独立复核通过**的修正才会被采用；复核没通过、或者根本没复核成功的，一律不写进正文。"
    )

    if report.corrected_text is not None:
        st.markdown("**修正后全文**（仅包含已采用的修改）")
        render_text_panel(report.corrected_text)

    for correction in report.corrections:
        accepted = correction.correction_id in applied
        icon = "已采用" if accepted else "未采用"
        with st.expander(
            f"{icon} {correction.correction_id} ｜ {correction.change_type.value} ｜ "
            f"{correction.original_span.text[:28]}"
            + ("" if accepted else "（未采用）"),
            expanded=not accepted,
        ):
            st.markdown(
                badge(
                    "已采用" if accepted else "未采用",
                    fg="#1a7f37" if accepted else "#b42318",
                    bg="#e9f7ee" if accepted else "#fdecea",
                    border="#a7e0b8" if accepted else "#f3b3ad",
                )
                + badge(
                    f"类型：{correction.change_type.value}",
                    fg="#3a5a8c",
                    bg="#eef3fb",
                    border="#b9cbe8",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(f"**原文片段**：{correction.original_span.text}")
            st.markdown(
                f"**修改为**：{correction.proposed_text if correction.proposed_text else '（删除该片段）'}"
            )
            st.markdown(f"**修改理由**：{correction.rationale}")
            if correction.risk_note:
                st.caption(f"该修改自身的风险：{correction.risk_note}")
            if correction.evidence_ids:
                st.markdown("**支撑证据**")
                for evidence_id in correction.evidence_ids:
                    evidence = report.evidence_by_id(evidence_id)
                    if evidence is not None:
                        st.markdown(f"`{evidence.evidence_id}`")
                        render_evidence_item(evidence)
            else:
                st.caption("未引用证据（该修改不改写事实内容）。")

            review = next(
                (
                    item
                    for item in report.review_results
                    if item.target_correction_id == correction.correction_id
                ),
                None,
            )
            if review is None:
                st.warning("该修正没有复核记录（因此不会被采用）。")
                continue
            overall_color = {
                ReviewOverall.PASSED: ("#1a7f37", "#e9f7ee", "#a7e0b8"),
                ReviewOverall.PASSED_WITH_RISKS: ("#a15c07", "#fff6e5", "#f2d08a"),
                ReviewOverall.FAILED: ("#b42318", "#fdecea", "#f3b3ad"),
            }[review.overall]
            st.markdown(
                badge(
                    f"复核结论：{review.overall.value}",
                    fg=overall_color[0],
                    bg=overall_color[1],
                    border=overall_color[2],
                )
                + (
                    badge(
                        f"复核者：{review.reviewer.provider}/{review.reviewer.model}",
                        fg="#5f6b7a",
                        bg="#f4f5f7",
                        border="#d5d9df",
                    )
                    if review.reviewer
                    else ""
                ),
                unsafe_allow_html=True,
            )
            st.markdown("**五项检查**")
            for check in review.checks:
                mark = "通过" if check.passed else "未通过"
                st.markdown(f"- {mark} {check.check.value}：{check.detail}")
            if review.unresolved:
                st.markdown("**仍未解决 / 需要人工确认**")
                for item in review.unresolved:
                    st.markdown(f"- {item}")


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
    st.markdown("#### 引擎交叉比对（规则引擎与模型）")
    if not report.cross_check_notes:
        st.caption("本次没有可展示的交叉比对信息。")
        return
    st.caption(
        "规则引擎与模型分别独立拆解同一段文本，以下为两者差异与分歧的完整记录。"
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
    st.markdown("#### 系统局限（本次检测的适用范围）")
    for item in report.limitations:
        st.markdown(f"- {item}")


def render_disclaimer(report: DetectionReport) -> None:
    st.caption(report.disclaimer)


# --------------------------------------------------------------------------
# 共识程度与证据链（反馈建议：让多模型结论与证据路径可核对）
# --------------------------------------------------------------------------
def claim_consensus(result: VerificationResult) -> tuple[str, str, str, str, str]:
    """依据**已有的**角色立场与证据状态给出共识标识。

    刻意不引入任何概率或"置信度"数字：
        · 角色意见相左 → 存在分歧
        · 意见一致但没有任何外部证据 → 证据不足
        · 意见一致且有证据 → 高度一致
        · 只有规则层结论、无角色意见 → 单层判定
    返回 (标识, 说明, 前景色, 背景色, 边框色)。
    """
    positions = [p for p in result.role_positions if p.stance is not RoleStance.NOT_APPLICABLE]
    if not positions:
        return (
            "单层判定",
            "本条仅有规则层结论，没有可交叉比对的角色意见。",
            "#5f6b7a", "#f4f5f7", "#d5d9df",
        )
    stances = {p.stance for p in positions}
    if RoleStance.SUPPORT in stances and RoleStance.REFUTE in stances:
        detail = "；".join(f"{p.role.value}主张{p.stance.value}" for p in positions)
        return ("存在分歧", f"分歧点：{detail}。", "#a15c07", "#fff6e5", "#f2d08a")
    if not result.supporting_evidence_ids and not result.refuting_evidence_ids:
        return (
            "证据不足",
            "各角色意见一致，但均未取得可支撑判断的外部证据。",
            "#3a5a8c", "#eef3fb", "#b9cbe8",
        )
    return (
        "高度一致",
        f"{len(positions)} 个角色意见一致，且已有证据支撑判断。",
        "#1a7f37", "#e9f7ee", "#a7e0b8",
    )


def render_consensus(result: VerificationResult) -> None:
    """在主张卡片顶部显示共识程度（有分歧时逐条说明是谁与谁不一致）。"""
    label, detail, fg, bg, border = claim_consensus(result)
    st.markdown(
        badge(f"共识程度：{label}", fg=fg, bg=bg, border=border)
        + f'<span style="color:#444;font-size:0.86rem;">{detail}</span>',
        unsafe_allow_html=True,
    )


def render_evidence_chain(report: DetectionReport, claim: Claim, result: VerificationResult) -> None:
    """证据链：论断 → 检索 → 命中文献（含时效性）→ 原文依据 → 裁决与复核。"""
    lines: list[str] = []
    lines.append(f"**① 论断**　{claim.source_span.text}　（原文第 {claim.source_span.start}–{claim.source_span.end} 字符）")

    queries: list[str] = []
    hits = misses = 0
    trace = getattr(result, "search_trace", None)
    if trace is not None:
        for record in trace.records:
            if record.query:
                queries.append(record.query)
            status = str(getattr(record.status, "value", record.status))
            # 精确比对枚举取值，避免模糊匹配把「已检索但无命中」算成命中
            if status == "检索成功":
                hits += 1
            elif status in ("已检索但无命中", "检索失败"):
                misses += 1
    if queries:
        shown = "；".join(queries[:3])
        suffix = f"（共 {len(queries)} 次，命中 {hits}，未命中 {misses}）" if len(queries) > 3 else ""
        lines.append(f"**② 检索关键词**　{shown}{suffix}")
    else:
        lines.append("**② 检索关键词**　无（本条未触发外部检索）")

    evidence_ids = list(
        dict.fromkeys(
            result.refuting_evidence_ids + result.supporting_evidence_ids + result.related_evidence_ids
        )
    )
    evidence_items = [item for item in (report.evidence_by_id(i) for i in evidence_ids) if item is not None]
    current_year = datetime.now().year
    if evidence_items:
        parts = []
        for item in evidence_items[:2]:
            label = item.title or item.url or item.evidence_id
            if item.year:
                age = max(current_year - int(item.year), 0)
                label += f"（{item.year} 年，距今 {age} 年）"
            parts.append(f"{label}　[{item.source.value}·{item.level.value}]")
        lines.append("**③ 命中文献**　" + "；".join(parts))
        snippet = next((item.snippet for item in evidence_items if item.snippet), None)
        lines.append(
            "**④ 原文依据**　" + (snippet[:180] + "…" if snippet and len(snippet) > 180 else snippet)
            if snippet
            else "**④ 原文依据**　无（仅比对到元数据，未取得摘要或全文）"
        )
    else:
        lines.append("**③ 命中文献**　无")
        lines.append("**④ 原文依据**　无")

    rule = result.adjudication_rule or "未命中规则"
    lines.append(f"**⑤ 裁决**　{result.status.value}（规则 {rule}）")

    review = next(
        (
            item
            for item in report.review_results
            if any(
                correction.correction_id == item.target_correction_id
                and correction.claim_id == claim.claim_id
                for correction in report.corrections
            )
        ),
        None,
    )
    if review is not None:
        lines.append(f"**⑥ 独立复核**　{review.overall.value}")
    st.markdown("**证据链**")
    for line in lines:
        st.markdown(f"- {line}")

def data_source_note(report: DetectionReport) -> tuple[str, str]:
    """如实标注本次结果的数据来源：本地缓存还是实时检索。"""
    requests = report.metrics.retrieval_requests if report.metrics else 0
    cache_hits = report.metrics.retrieval_cache_hits if report.metrics else 0
    if requests > 0:
        return (
            f"数据来源：实时检索（本次发出 {requests} 次外部请求）",
            "#1a7f37",
        )
    if cache_hits > 0:
        return (
            f"数据来源：本地缓存（已验证数据，本次未联网；缓存命中 {cache_hits} 次）",
            "#a15c07",
        )
    return ("数据来源：未进行外部检索", "#5f6b7a")


def render_conclusion_summary(report: DetectionReport) -> None:
    """报告开头的一段结论摘要：只给计数与要点，不给任何概率化分数。"""
    stats = report.stats
    problem = [result for result in report.verification_results if result.is_suspicious]
    note, color = data_source_note(report)
    st.markdown("#### 结论摘要")
    st.markdown(
        f"- 本次共核验 **{stats.total_claims}** 条主张：**{stats.suspicious_claims}** 条判定存在问题"
        f"（存在矛盾 / 部分支持 / 来源冲突），**{stats.flagged_claims}** 条需关注，"
        f"**{stats.status_counts.get('证据不足', 0)}** 条证据不足，"
        f"**{stats.status_counts.get('暂无法核验', 0)}** 条暂无法核验。"
    )
    if problem:
        detail = "；".join(
            f"「{report.claim_by_id(item.claim_id).source_span.text[:18]}」→ {item.status.value}"
            for item in problem[:3]
            if report.claim_by_id(item.claim_id) is not None
        )
        st.markdown(f"- 需要优先处理：{detail}。")
    else:
        st.markdown("- 本次没有判定为「存在问题」的主张。")
    st.markdown(
        "- **证据不足不等于错误**：它表示在本次检索范围内还没有取得足以支持或反驳的资料。"
        "本报告不提供未经过校准的概率化分数。"
    )
    st.markdown(
        badge(note, fg=color, bg=color + "14", border=color + "55"),
        unsafe_allow_html=True,
    )


def render_strength_badge(result: VerificationResult, evidence_items: list[Evidence]) -> None:
    """证据强度徽标（A/B/C/未获证据）。"""
    verdict = assess_evidence_strength(result, evidence_items)
    st.markdown(
        badge(verdict.label, fg=verdict.fg, bg=verdict.bg, border=verdict.border)
        + f'<span style="color:#444;font-size:0.86rem;">{verdict.reason}</span>',
        unsafe_allow_html=True,
    )
