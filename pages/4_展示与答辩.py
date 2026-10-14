"""展示与答辩：按"评委模式"组织为三档——30 秒看懂 / 3 分钟演示 / 答辩备查。

设计依据（评审建议）：
    · 首页只负责"第一印象"，技术细节搬到本页，避免首屏信息过载；
    · 评委的时间预算不同，因此给三档入口，而不是一堆平铺的标签页；
    · 所有数字都实时读取（标注集、评测存档、运行配置），不写死。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from verimind.core.config import Settings
from verimind.core.enums import STATUS_DISPLAY_ORDER
from verimind.core.errors import VeriMindError
from verimind.evaluation.dataset import dataset_stats
from verimind.storage.repository import RunRepository
from verimind.ui.method_notes import (
    render_differences,
    render_environment,
    render_pipeline_diagram,
    render_problem_definition,
    render_technical_route,
)
from verimind.ui.theme import (
    badge,
    inject_css,
    render_config_warnings,
    render_footer,
    render_run_mode_banner,
)

st.set_page_config(page_title="展示与答辩 · 智鉴 AI", page_icon="🎬", layout="wide")
inject_css()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    "architecture": PROJECT_ROOT / "docs" / "07-architecture-and-innovation.md",
    "demo": PROJECT_ROOT / "docs" / "07-demo-script.md",
    "experiment": PROJECT_ROOT / "docs" / "07-experiment-and-defense.md",
}
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
GOLDEN_PATH = PROJECT_ROOT / "examples" / "golden_set.jsonl"

settings = Settings()

st.title("展示与答辩")
st.caption("按评审时间预算分三档：结论、流程、细节。全部数据来自真实运行。")

render_run_mode_banner(settings)
render_config_warnings(settings)


def latest_metrics() -> dict:
    """读取最近一次全量评测的关键指标（没有存档时返回空字典）。"""
    if not EVALUATIONS_DIR.exists() or not GOLDEN_PATH.exists():
        return {}
    candidates = sorted(EVALUATIONS_DIR.glob("b2*.json")) or sorted(EVALUATIONS_DIR.glob("*.json"))
    if not candidates:
        return {}
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8")).get("metrics", {})
    except (OSError, ValueError):
        return {}


def render_doc(path: Path, *, download_label: str) -> None:
    if not path.exists():
        st.error(f"找不到材料文件：{path}")
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


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


tab_intro, tab_demo, tab_detail = st.tabs(["30 秒看懂", "3 分钟演示", "答辩备查"])

# ==========================================================================
# 30 秒看懂
# ==========================================================================
with tab_intro:
    st.markdown("### 让每条 AI 事实主张都有可追溯的证据与复核结论")
    st.markdown("系统不把模型怀疑当作结论：**每条判断均须经过外部证据、明确规则与独立复核。**")

    stats = dataset_stats(GOLDEN_PATH) if GOLDEN_PATH.exists() else None
    metrics = latest_metrics()

    cols = st.columns(4)
    if stats is not None:
        cols[0].metric("标注测试集", f"{stats.total} 条", f"已人工确认 {stats.confirmed} 条")
    if metrics:
        cols[1].metric(
            "主张级精确率",
            _percent(metrics.get("detection_precision")),
            f"召回率 {_percent(metrics.get('detection_recall'))}",
        )
        cols[2].metric("误报率", _percent(metrics.get("false_positive_rate")), "越低越好")
        calls = metrics.get("avg_llm_calls_per_case")
        cols[3].metric(
            "单条核验成本",
            "—" if calls is None else f"{calls} 次调用",
            "重复核验命中缓存则不产生费用",
        )
    else:
        cols[1].metric("评测结果", "尚未生成", "可在评测页运行生成")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### 怎么用（两步）")
        st.markdown(
            "1. 在**检测工作台**粘贴待核验文本（或直接使用内置样例）；\n"
            "2. 打开**检测报告**：先看结论摘要，再逐条展开主张、证据、修正与复核。"
        )
        st.markdown("##### 系统能回答的问题")
        st.markdown(
            "- 这段话里的哪一句是可核验的事实主张？\n"
            "- 它引用的文献真实存在吗？年份、期刊对得上吗？\n"
            "- 文献的内容真的支持这句话吗（还是只是看起来相关）？\n"
            "- 有哪些地方**没有证据**、需要人工复核？"
        )
    with right:
        st.markdown("##### 明确的边界（同样重要）")
        st.markdown(
            "- 只核验**可检索的学术文献**层面；中文文献与非学术事实无法核验；\n"
            "- 最多读到**摘要**，不通读全文；\n"
            "- 「文献是真的」与「文献真的这么说」是两个问题，系统分别处理；\n"
            "- 「证据不足」不等于「错误」；\n"
            "- 不提供未经过校准的概率化分数。"
        )
        with st.expander("模型与角色配置（评委追问时展开）"):
            engine_rows = [
                {"角色": "生成模型", "当前配置": f"{settings.llm_provider} / {settings.llm_model}"}
            ]
            if settings.reviewer_is_distinct:
                engine_rows.append(
                    {
                        "角色": "独立复核模型",
                        "当前配置": f"{settings.llm_reviewer_provider} / {settings.llm_reviewer_model}",
                    }
                )
            else:
                engine_rows.append({"角色": "独立复核模型", "当前配置": "未单独配置，由生成模型兼任"})
            engine_rows.append({"角色": "规则裁决层", "当前配置": "确定性代码，不是模型"})
            st.dataframe(pd.DataFrame(engine_rows), width="stretch", hide_index=True)
            st.caption(
                "只有生成与复核由**不同厂商**的模型承担时，界面才显示「多模型协同」；"
                "否则如实标注为「单模型多角色模拟」。"
            )

    st.divider()
    st.markdown("##### 核心表述")
    st.info("智鉴 AI 不把模型怀疑当作结论；每条判断都需经过可追溯证据、明确规则和独立复核。", icon=None)

# ==========================================================================
# 3 分钟演示
# ==========================================================================
with tab_demo:
    st.markdown("### 3 分钟闭环演示")
    st.caption("固定流程：输入文本 → 拆分主张 → 命中问题 → 打开来源 → 展示独立复核 → 导出报告。")

    st.markdown("##### 演示步骤与要点")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "步骤": "① 输入",
                    "操作": "工作台载入内置样例",
                    "要看什么": "样例同时包含年份矛盾、表述过度与证据不足",
                },
                {
                    "步骤": "② 拆分",
                    "操作": "点「开始核验」",
                    "要看什么": "阶段进度条推进；展开「阶段明细」可看每一步",
                },
                {
                    "步骤": "③ 命中",
                    "操作": "打开检测报告",
                    "要看什么": "结论摘要给出问题条数；主张卡片显示状态、共识程度与证据强度",
                },
                {
                    "步骤": "④ 来源",
                    "操作": "展开主张卡片",
                    "要看什么": "证据链：论断 → 检索词 → 命中文献（含年份）→ 摘要片段 → 裁决规则",
                },
                {
                    "步骤": "⑤ 复核",
                    "操作": "滚到「修正建议与独立复核」",
                    "要看什么": "五项检查逐条结论；复核未通过则标注「未采用」",
                },
                {
                    "步骤": "⑥ 导出",
                    "操作": "点「下载完整报告」",
                    "要看什么": "Markdown 报告含规则、证据链接、局限与 token 统计",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("##### 可当场复现的命令")
    st.code(
        ".\\.venv\\Scripts\\python.exe scripts\\check_llm.py\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_demo.py --id mixed_status_demo\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --baseline b1 --all\n"
        ".\\.venv\\Scripts\\python.exe -m pytest -q",
        language="powershell",
    )
    st.caption(
        "演示前先执行一次即可把结果写入本地缓存；重复演示为 **0 次计费调用**。"
        "若现场断网且缓存为空，可切换为离线规则引擎模式，报告中会如实标注「离线模拟」。"
    )

# ==========================================================================
# 答辩备查
# ==========================================================================
with tab_detail:
    st.markdown("#### 当前配置")
    config_cols = st.columns(4)
    config_cols[0].metric("运行模式", settings.run_mode.value)
    config_cols[1].metric("生成模型", settings.llm_provider)
    config_cols[2].metric(
        "独立复核模型",
        settings.llm_reviewer_provider if settings.reviewer_is_distinct else "未单独配置",
    )
    config_cols[3].metric("调用上限", f"{settings.llm_max_calls_per_run} 次/次核验")

    if settings.reviewer_is_distinct:
        st.success("生成与复核由不同厂商的模型承担，独立复核成立。", icon=None)
    else:
        st.warning("未配置独立的复核模型，复核由生成模型兼任，报告中会如实标注独立性有限。", icon=None)

    if GOLDEN_PATH.exists():
        stats = dataset_stats(GOLDEN_PATH)
        st.markdown("#### 标注集确认进度")
        label_cols = st.columns(4)
        label_cols[0].metric("用例总数", stats.total)
        label_cols[1].metric("覆盖类别", stats.categories)
        label_cols[2].metric("人工已确认", stats.confirmed)
        label_cols[3].metric("待确认", stats.awaiting)
        st.markdown(
            badge(f"人工已确认 {stats.confirmed} 条", fg="#1a7f37", bg="#e9f7ee", border="#a7e0b8")
            + badge(
                f"待确认 {stats.awaiting} 条",
                fg="#a15c07" if stats.awaiting else "#1a7f37",
                bg="#fff6e5" if stats.awaiting else "#e9f7ee",
                border="#f2d08a" if stats.awaiting else "#a7e0b8",
            ),
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### 方法说明")
    render_problem_definition()
    st.divider()
    render_technical_route()
    st.divider()
    render_differences()
    st.divider()
    render_pipeline_diagram()

    st.divider()
    st.markdown("#### 最近核验记录")
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
                        "判定有问题": item.suspicious_claims,
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
            f"累计运行 {repository.count()} 次；计费模型调用 {usage.get('calls', 0)} 次；"
            f"累计 token {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
        )
    else:
        st.caption("暂无检测记录。")

    st.divider()
    render_environment(settings)

    st.divider()
    st.markdown("#### 完整材料")
    render_doc(DOCS["architecture"], download_label="下载《技术架构与核心创新点》Markdown")
    render_doc(DOCS["demo"], download_label="下载《演示脚本》Markdown")
    render_doc(DOCS["experiment"], download_label="下载《实验报告与答辩材料》Markdown")

    st.caption("状态取值顺序：" + " → ".join(item.value for item in STATUS_DISPLAY_ORDER))

render_footer()
