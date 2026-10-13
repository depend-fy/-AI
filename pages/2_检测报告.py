"""检测报告：主张、证据、问题类型、系统局限。

这一页是整个 Demo 最重要的展示面，也是最容易"注水"的地方。因此定下三条规矩：
    1. 所有数字来自 `report.stats`（由流水线自动计算），页面不自己数、更不手写；
    2. 每条结论都要能看到：原文位置 → 判断理由 → 命中的规则 → 证据（或"没有证据"）；
    3. 没有的东西就明说（没有外部证据、没有模型角色、没有修正），不摆样子。
"""

from __future__ import annotations

import streamlit as st

from verimind.core.config import Settings
from verimind.core.enums import Severity
from verimind.core.schemas import DetectionReport
from verimind.core.errors import VeriMindError
from verimind.storage.repository import RunRepository
from verimind.ui.components import (
    render_claim_card,
    render_disclaimer,
    render_limitations,
    render_report_header,
    render_run_info,
    render_stat_cards,
)
from verimind.ui.report_export import report_to_markdown
from verimind.ui.theme import inject_css, render_footer, render_status_legend, safe_page_link

st.set_page_config(page_title="检测报告 · 智鉴 AI", page_icon="📄", layout="wide")
inject_css()

settings = Settings()

st.title("📄 检测报告")
st.caption(
    "报告中的每一条判断都能追问：依据是什么、命中了哪条规则、还缺什么证据。"
)

# ---------------------------------------------------------------- 选择报告
session_report: DetectionReport | None = st.session_state.get("last_report")
summaries = []
repository: RunRepository | None = None
try:
    repository = RunRepository(settings.sqlite_path)
    summaries = repository.list_recent(limit=20)
except VeriMindError as exc:
    st.warning(f"读取本地历史记录失败：{exc}")

_CURRENT = "（当前会话最新一次检测）"
options = [_CURRENT] + [
    f"{item.created_at.astimezone():%m-%d %H:%M} ｜ {item.total_claims} 条主张 ｜ "
    f"{item.suspicious_claims} 条有问题 ｜ {item.input_preview[:18]}…"
    for item in summaries
]
choice = st.selectbox(
    "选择要查看的报告",
    options,
    index=0,
    key="report_selector",
    help="历史记录保存在本地 data/verimind.db，不会上传到任何地方。",
)

report: DetectionReport | None = None
if choice == _CURRENT:
    report = session_report
    if report is None and summaries and repository is not None:
        st.info("本次会话还没有新的检测，已自动显示最近一次历史记录。")
        try:
            report = repository.get(summaries[0].run_id)
        except VeriMindError as exc:
            st.warning(str(exc))
else:
    position = options.index(choice) - 1
    if repository is not None and 0 <= position < len(summaries):
        try:
            report = repository.get(summaries[position].run_id)
        except VeriMindError as exc:
            st.warning(str(exc))

if report is None:
    st.info(
        "还没有可查看的检测报告。请先到「检测工作台」输入文本并点击「开始检测」。",
        icon="🧪",
    )
    safe_page_link("pages/1_检测工作台.py", label="前往检测工作台 →", icon="🧪")
    st.stop()

# ---------------------------------------------------------------- 报告头部
render_report_header(report)
render_stat_cards(report)

# ---------------------------------------------------------------- 导出
_export_left, _export_right = st.columns([1, 3])
with _export_left:
    st.download_button(
        "⬇️ 下载完整报告",
        data=report_to_markdown(report),
        file_name=f"verimind-report-{report.run_id[:8]}.md",
        mime="text/markdown",
        key="download_report_md",
        width="stretch",
    )
with _export_right:
    st.caption(
        "导出内容与页面**同源**（同一份数据对象）：逐条主张、命中规则、证据链接与留档路径、"
        "修正与复核结论、系统局限、运行信息（含 token 与缓存命中）全都在里面，可直接归档或贴进材料。"
    )

st.divider()
st.markdown("#### 📖 原文")
with st.container():
    st.caption("高亮部分为系统识别出的、需要核验的关键片段。")
    from verimind.ui.components import render_text_panel

    render_text_panel(
        report.input_text,
        [(claim.source_span.start, claim.source_span.end) for claim in report.claims],
    )

# ---------------------------------------------------------------- 主张分组
flagged: list[tuple] = []
unverified: list[tuple] = []
for claim in report.claims:
    result = report.result_by_claim(claim.claim_id)
    if result is None:
        continue
    if result.requires_attention:
        flagged.append((claim, result))
    else:
        unverified.append((claim, result))

_severity_rank = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.NONE: 3}
flagged.sort(key=lambda pair: (_severity_rank.get(pair[1].severity, 9), pair[0].claim_id))

# ---------------------------------------------------------------- 需关注
st.divider()
st.markdown(f"#### ⚠️ 需关注的主张（{len(flagged)} 条）")
if not flagged:
    st.success(
        "本次检测没有发现「已经可以确认的问题」或「明显疑点」。"
        "但请注意：这**不等于内容正确**——见下方「尚未核验的主张」。"
    )
else:
    st.caption(
        "严重度=高 表示证据充分、可以确认的问题（例如原文自相矛盾）；"
        "严重度=中 表示已发现疑点但尚未证实（例如给出了具体数值却没有出处）。"
    )
    for claim, result in flagged:
        render_claim_card(report, claim, result, expanded=(result.severity is Severity.HIGH))

# ---------------------------------------------------------------- 尚未核验
st.divider()
st.markdown(f"#### 🔎 尚未核验的主张（{len(unverified)} 条）")
if unverified:
    st.caption(
        "这些主张本身没有发现疑点，只是**阶段 2 还没有接入外部检索**，因此无法核实。"
        "把「证据不足」当成「有问题」是本项目最想避免的误判。"
    )
    for claim, result in unverified:
        render_claim_card(report, claim, result, expanded=False)
else:
    st.caption("没有待核验的主张。")

# ---------------------------------------------------------------- 修正与复核
st.divider()
from verimind.ui.components import render_correction_section  # noqa: E402

render_correction_section(report)

# ---------------------------------------------------------------- 交叉比对
st.divider()
from verimind.ui.components import render_cross_check  # noqa: E402

render_cross_check(report)

# ---------------------------------------------------------------- 状态图例
st.divider()
st.markdown("#### 🎨 状态含义对照")
render_status_legend()

# ---------------------------------------------------------------- 局限与运行信息
render_limitations(report)
render_run_info(report)

st.divider()
download_col, note_col = st.columns([1, 3])
with download_col:
    st.download_button(
        "⬇️ 导出报告 JSON",
        data=report.to_json(),
        file_name=f"verimind-report-{report.run_id[:8]}.json",
        mime="application/json",
        key="download_report",
    )
with note_col:
    st.caption(
        "导出的 JSON 包含全部主张、证据、裁决理由与系统局限，可用于复现与归档。"
    )

render_disclaimer(report)
render_footer()
