"""检测工作台：输入待核验文本并启动检测。

阶段 3 起：配置了真实模型就走「模型拆解 + 规则基线交叉比对 + 事实审查员」，
没配置（或模型失败）则降级为离线规则引擎，并在报告里如实声明。
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from verimind.core.config import Settings
from verimind.core.errors import VeriMindError
from verimind.core.pipeline import ProgressEvent, run_detection
from verimind.core.schemas import DetectionReport
from verimind.core.enums import STATUS_DISPLAY_ORDER
from verimind.ui.theme import (
    render_config_warnings,
    render_run_mode_banner,
    safe_page_link,
    status_badge_html,
)
from verimind.ui.theme import inject_css, render_footer

st.set_page_config(page_title="检测工作台 · 智鉴 AI", page_icon="🧪", layout="wide")
inject_css()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = PROJECT_ROOT / "examples" / "sample_inputs.jsonl"


@st.cache_data(show_spinner=False)
def load_samples() -> list[dict[str, str]]:
    """读取示例文本（缓存，避免每次重跑都读盘）。"""
    if not SAMPLES_PATH.exists():
        return []
    samples: list[dict[str, str]] = []
    for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


settings = Settings()

st.title("🧪 检测工作台")
st.caption(
    "粘贴一段需要核验的 AI 生成内容，系统会把它拆解为可单独核验的事实主张，"
    "并给出每一条的核验状态、判断理由与「还缺什么证据」。"
)

render_run_mode_banner(settings)
render_config_warnings(settings)

# ---------------------------------------------------------------- 示例文本
samples = load_samples()
if samples:
    st.markdown("#### 📋 一键载入示例文本")
    columns = st.columns(len(samples))
    for column, sample in zip(columns, samples):
        with column:
            if st.button(sample["title"], key=f"load_{sample['id']}", width="stretch"):
                st.session_state["workbench_text"] = sample["text"]
                st.session_state["workbench_note"] = sample["note"]
                st.session_state["workbench_loaded"] = sample["title"]
            st.caption(sample["note"][:70] + "…")

    if st.session_state.get("workbench_loaded"):
        st.info(
            f"已载入示例：**{st.session_state['workbench_loaded']}**。"
            "可以直接点「开始检测」，也可以先修改文本。",
            icon="✅",
        )

# ---------------------------------------------------------------- 输入
st.markdown("#### ✍️ 待核验文本")
text = st.text_area(
    "待核验文本",
    key="workbench_text",
    height=200,
    max_chars=settings.max_input_chars,
    label_visibility="collapsed",
    placeholder="在此粘贴需要核验的内容，例如包含文献引用、具体数值或机构结论的段落……",
)
char_count = len(text or "")
over_limit = char_count > settings.max_input_chars

count_col, action_col = st.columns([3, 1])
with count_col:
    reviewer_label = (
        f"{settings.llm_reviewer_provider}/{settings.llm_reviewer_model}"
        if settings.reviewer_is_distinct
        else "与主模型相同"
    )
    st.caption(
        f"已输入 {char_count} / {settings.max_input_chars} 字"
        f"　｜　引擎：{settings.llm_provider}/{settings.llm_model}"
        f"　｜　复核：{reviewer_label}"
    )
    st.caption(
        "运行前预估：每条主张约 **2–4 次模型调用**（主张拆解 / 事实审查 / 可能的证据审查、修正与复核）"
        "＋若干次外部检索。**同一段文本重复检测会命中本地缓存（0 次计费调用）**。"
        f"需要更长的输入时，在 `.env` 里设置 `VERIMIND_MAX_INPUT_CHARS`（当前 {settings.max_input_chars}）。"
    )
    if st.button("🧹 清空文本", key="clear_workbench"):
        st.session_state["workbench_text"] = ""
        st.session_state["workbench_loaded"] = None
        st.session_state["workbench_note"] = ""
        st.rerun()

with action_col:
    run_clicked = st.button("🚀 开始检测", type="primary", key="run_detection", width="stretch")

if over_limit:
    st.error(f"文本超过 {settings.max_input_chars} 字的限制，请缩短后重试。")
    run_clicked = False

# ---------------------------------------------------------------- 执行
if run_clicked:
    if not (text or "").strip():
        st.warning("请先输入待核验的文本。")
    else:
        progress_bar = st.progress(0.0, text="准备中…")
        log_box = st.container()

        def on_progress(event: ProgressEvent) -> None:
            progress_bar.progress(min(max(event.ratio, 0.0), 1.0), text=event.stage)
            detail = f"：{event.detail}" if event.detail else ""
            log_box.markdown(f"- {event.stage}{detail}")

        log_box.markdown("**检测进度**")
        try:
            report = run_detection(text, settings=settings, progress=on_progress)
        except VeriMindError as exc:
            progress_bar.empty()
            st.error(f"检测失败：{exc}")
            report = None
        except Exception as exc:  # noqa: BLE001 - 兜底，避免界面白屏
            progress_bar.empty()
            st.error(f"检测过程中出现未预期的错误：{type(exc).__name__}: {exc}")
            report = None

        if report is not None:
            progress_bar.progress(1.0, text="检测完成")
            st.session_state["last_run_id"] = report.run_id
            st.session_state["last_report"] = report
            st.success(f"检测完成，共 {report.stats.total_claims} 条主张。")
            # 不调用 st.rerun()：下面的摘要区块在同一次运行里就能读到 session_state

# ---------------------------------------------------------------- 结果摘要
last_report: DetectionReport | None = st.session_state.get("last_report")
if last_report is not None:
    st.divider()
    st.markdown("#### 📊 本次检测结果摘要")
    st.caption(f"运行 ID：`{last_report.run_id}`")

    stats = last_report.stats
    columns = st.columns(4)
    columns[0].metric("主张总数", stats.total_claims)
    columns[1].metric("已判定有问题", stats.suspicious_claims)
    columns[2].metric("需关注", stats.flagged_claims)
    columns[3].metric("耗时(ms)", last_report.metrics.latency_ms or 0)

    st.markdown(
        "".join(
            status_badge_html(status) + f" {stats.status_counts.get(status.value, 0)}　"
            for status in STATUS_DISPLAY_ORDER
        ),
        unsafe_allow_html=True,
    )

    safe_page_link("pages/2_检测报告.py", label="查看完整检测报告 →", icon="📄")
    st.caption(
        "提示：报告页也可以打开历史记录（保存在本地 `data/verimind.db`，不会上传任何地方）。"
    )

# ---------------------------------------------------------------- 说明
st.divider()
with st.expander("这一版（阶段 3）到底做了什么、没做什么", expanded=False):
    st.markdown(
        "**已经能做的**\n"
        "1. 用**真实大模型**把文本拆解成可单独核验的原子主张（含精确原文位置），"
        "同时用规则引擎独立拆解一遍，两个引擎的差异全部写进报告的「引擎交叉比对」；\n"
        "2. 识别 DOI、年份、具体数值、因果表述、模糊引用、观点预测等关键要素；\n"
        "3. 检查**原文内部的矛盾**（同一文献两个年份、同一指标两个数值）——"
        "这类矛盾不需要任何外部证据就能确认；\n"
        "4. 事实审查员（角色 A）逐条审查并给出怀疑方向；最终核验状态仍由**可解释的规则表**裁决，"
        "模型不能直接判定对错，模型的怀疑与规则层的差异会如实展示；\n"
        "5. 记录真实 token 用量与耗时；在 .env 中配置单价后可估算成本；\n"
        "6. 模型调用失败（超时/限流/格式错误）时**降级为规则引擎并在报告里显著声明**，"
        "绝不悄悄用规则结果冒充模型结果。\n\n"
        "**还没有做的（会在后续阶段补上）**\n"
        "- 阶段 4：接入 Crossref / OpenAlex / arXiv，真正去核对文献是否存在、年份是否一致；\n"
        "- 阶段 5：基于证据生成修正建议，并做修正后的独立复核（角色 B / C）；\n"
        "- 阶段 6：在人工标注集上给出精确率、误报率等指标。\n\n"
        "**为什么现在报告里没有「已核实」**\n"
        "因为还没有查过任何外部来源。没有证据就说「已核实」，是本项目最不能接受的事。"
    )

render_footer()
