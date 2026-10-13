"""展示与答辩：把阶段 7 的展示材料做成网页，并附上**实时**项目状态。

设计要点：
    1. **单一数据源**：本页直接渲染 `docs/07-*.md`，不为网页再抄一份内容——
       改文档即改网页，永远不会出现"文档和网页说的不一样"。
    2. **实时状态**：顶部展示的不是写死的数字，而是**现在**读到的配置、
       标注集确认进度与最近几次检测记录（来自本地 SQLite）。
    3. **可带走**：每份材料都能一键下载 Markdown 原文，方便贴进 PPT 或提交材料。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from verimind.core.config import Settings
from verimind.core.enums import STATUS_DISPLAY_ORDER
from verimind.core.errors import VeriMindError
from verimind.evaluation.dataset import categories, load_golden_set
from verimind.storage.repository import RunRepository
from verimind.ui.theme import badge, render_config_warnings, render_run_mode_banner
from verimind.ui.theme import inject_css, render_footer

st.set_page_config(page_title="展示与答辩 · 智鉴 AI", page_icon="🎬", layout="wide")
inject_css()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    "architecture": PROJECT_ROOT / "docs" / "07-architecture-and-innovation.md",
    "demo": PROJECT_ROOT / "docs" / "07-demo-script.md",
    "experiment": PROJECT_ROOT / "docs" / "07-experiment-and-defense.md",
}
GOLDEN_PATH = PROJECT_ROOT / "examples" / "golden_set.jsonl"

settings = Settings()

st.title("展示与答辩")
st.caption(
    "比赛用的演示脚本、技术架构与创新点、实验报告与答辩材料。"
    "所有数字都来自真实运行；局限与未完成事项也一并写在这里。"
)

render_run_mode_banner(settings)
render_config_warnings(settings)

tabs = st.tabs(
    [
        "实时状态",
        "架构与创新点",
        "演示脚本",
        "实验报告与答辩",
    ]
)


def render_doc(path: Path, *, download_label: str) -> None:
    """渲染一份 Markdown 材料（去掉首行 H1，避免与页面标题重复）。"""
    if not path.exists():
        st.error(f"找不到材料文件：{path}（是否被移动或删除？）")
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    st.markdown("\n".join(lines).lstrip("\n"))
    st.divider()
    st.download_button(
        download_label,
        data=text,
        file_name=path.name,
        mime="text/markdown",
        key=f"download-{path.stem}",
    )


# ==========================================================================
# ① 实时状态
# ==========================================================================
with tabs[0]:
    st.markdown("#### 运行配置")
    engine_cols = st.columns(4)
    engine_cols[0].metric("运行模式", settings.run_mode.value)
    engine_cols[1].metric("主模型", f"{settings.llm_provider}")
    engine_cols[2].metric(
        "独立复核员",
        f"{settings.llm_reviewer_provider}" if settings.reviewer_is_distinct else "与主模型相同",
    )
    engine_cols[3].metric("模型调用上限", f"{settings.llm_max_calls_per_run} 次/次检测")

    if settings.reviewer_is_distinct:
        st.success(
            f"独立复核已启用：修正员用 `{settings.llm_provider}/{settings.llm_model}`，"
            f"复核员用 `{settings.llm_reviewer_provider}/{settings.llm_reviewer_model}`——"
            "**改的人**与**检查改得对不对的人**不是同一个模型。",
        )
    else:
        st.warning(
            "尚未配置独立的复核模型：复核由主模型兼任，报告中会如实标注「复核独立性有限」。"
            "在 `.env` 中配置 `VERIMIND_LLM_REVIEWER_*` 即可升级为多模型协同。",
        )

    st.divider()
    st.markdown("#### 人工标注测试集确认进度")
    try:
        cases = load_golden_set(GOLDEN_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"读取标注集失败：{exc}")
        cases = []

    if cases:
        confirmed = sum(1 for case in cases if case.human_confirmed)
        label_cols = st.columns(4)
        label_cols[0].metric("用例总数", len(cases))
        label_cols[1].metric("覆盖类别", len(categories(cases)))
        label_cols[2].metric("人工已确认", confirmed)
        label_cols[3].metric("待人工确认", len(cases) - confirmed)
        st.markdown(
            badge(f"人工已确认 {confirmed} 条", fg="#1a7f37", bg="#e9f7ee", border="#a7e0b8")
            + badge(
                f"待确认 {len(cases) - confirmed} 条",
                fg="#a15c07" if len(cases) - confirmed else "#1a7f37",
                bg="#fff6e5" if len(cases) - confirmed else "#e9f7ee",
                border="#f2d08a" if len(cases) - confirmed else "#a7e0b8",
            ),
            unsafe_allow_html=True,
        )
        if len(cases) - confirmed:
            st.info(
                "在**全部**用例经人工确认之前，评测页与实验报告里的指标都只能称为"
                "「AI 起草标注上的初步结果」。确认入口：`examples/golden_set_review.csv`"
                "（Excel 直接打开，填最后两列）。",
            )

    st.divider()
    st.markdown("#### 最近检测记录")
    try:
        repository = RunRepository(settings.sqlite_path)
        summaries = repository.list_recent(limit=8)
        usage = repository.usage_summary()
    except VeriMindError as exc:
        st.warning(f"读取本地记录失败：{exc}")
        summaries, usage = [], {}

    if summaries:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "时间": item.created_at.astimezone().strftime("%m-%d %H:%M"),
                        "模式": item.run_mode,
                        "主张数": item.total_claims,
                        "已判定有问题": item.suspicious_claims,
                        "需关注": item.flagged_claims,
                        "输入摘要": item.input_preview,
                    }
                    for item in summaries
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"累计运行 {repository.count()} 次　｜　计费模型调用 {usage.get('calls', 0)} 次　｜　"
            f"累计 token {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
            "（读自本地 `llm_calls` 表；命中缓存的调用不计费也不计 token）"
        )
    else:
        st.caption("还没有检测记录。到「检测工作台」跑一次就会出现在这里。")

    st.divider()
    st.markdown("#### 能力范围")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**已验证可用**\n"
            "1. 真实模型拆解主张 + 规则引擎交叉比对\n"
            "2. Crossref / OpenAlex / arXiv 真实检索（含原始响应留档）\n"
            "3. 区分「文献是否存在」与「内容是否被支持」\n"
            "4. 文内自相矛盾检测（无需外部证据）\n"
            "5. 证据驱动的最小改动修正 + **跨厂商独立复核**\n"
            "6. 只有复核通过的修改才写进「修正后全文」\n"
            "7. 真实 token / 检索次数 / 耗时统计，结果可缓存复用\n"
            "8. 在 20 条标注集上跑评测（含免费下界基线）"
        )
    with right:
        st.markdown(
            "**尚未做到**\n"
            "- 论文**全文**阅读（现在最多到摘要）\n"
            "- 中文文献核验（数据源没有公开 API）\n"
            "- 非学术事实核验（需要付费检索服务）\n"
            "- 作者 / 机构类核验（目前只实现年份与期刊）\n"
            "- B0 基线（让原模型自检自己）未实现\n"
            "- 标注仍有 16 条待人工确认（存在自我实现偏差风险）"
            "\n\n状态取值顺序：" + " → ".join(item.value for item in STATUS_DISPLAY_ORDER)
        )

# ==========================================================================
# ②③④ 三份材料
# ==========================================================================
with tabs[1]:
    render_doc(DOCS["architecture"], download_label="⬇️ 下载《技术架构与核心创新点》Markdown")

with tabs[2]:
    render_doc(DOCS["demo"], download_label="⬇️ 下载《演示脚本》Markdown")

with tabs[3]:
    render_doc(DOCS["experiment"], download_label="⬇️ 下载《实验报告与答辩材料》Markdown")

render_footer()
