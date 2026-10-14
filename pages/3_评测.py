"""评测：在人工标注测试集上给出**真实跑出来**的指标。

这一页的三条纪律（写在代码里，也写在页面上）：
    1. 数字只能来自真实运行：点「运行评测」才会跑，跑完把逐条结果与指标一起展示；
    2. 样本量、标注者身份、同源偏差、失败案例全部显示，不做任何粉饰；
    3. 分母为 0 的指标显示「—」，绝不拿 0 或 100% 去凑一个好看的表格。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from verimind.core.config import Settings
from verimind.core.enums import STATUS_DISPLAY_ORDER
from verimind.core.errors import VeriMindError
from verimind.evaluation.dataset import categories, load_golden_set
from verimind.evaluation.runner import (
    BASELINE_B0,
    BASELINE_B1,
    BASELINE_B2,
    EvaluationReport,
    offline_settings,
    run_evaluation,
)
from verimind.ui.theme import badge, render_config_warnings, render_run_mode_banner
from verimind.ui.theme import inject_css, render_footer

st.set_page_config(page_title="评测 · 智鉴 AI", page_icon="📈", layout="wide")
inject_css()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "examples" / "golden_set.jsonl"

settings = Settings()

st.title("评测")
st.caption(
    "在人工标注测试集上真实跑一遍完整流程，给出主张级指标、失败案例与实验说明。"
    "所有数字都来自运行结果，**不做任何粉饰**。"
)

render_run_mode_banner(settings)
render_config_warnings(settings)

# ---------------------------------------------------------------- 标注集
try:
    cases = load_golden_set(GOLDEN_PATH)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"读取标注集失败：{exc}")
    st.stop()

category_list = categories(cases)
ai_labelled = sum(1 for case in cases if case.labeler_is_ai)
human_confirmed = sum(1 for case in cases if case.human_confirmed)

st.markdown("#### 测试集概览")
stat_cols = st.columns(4)
stat_cols[0].metric("用例总数", len(cases))
stat_cols[1].metric("覆盖类别", len(category_list))
stat_cols[2].metric("人工已确认", human_confirmed)
stat_cols[3].metric("待人工确认", len(cases) - human_confirmed)

st.markdown(
    badge(f"来源：{GOLDEN_PATH.name}", fg="#3a5a8c", bg="#eef3fb", border="#b9cbe8")
    + badge(
        f"人工已确认 {human_confirmed} 条" if human_confirmed else "尚无人工确认",
        fg="#1a7f37" if human_confirmed else "#5f6b7a",
        bg="#e9f7ee" if human_confirmed else "#f4f5f7",
        border="#a7e0b8" if human_confirmed else "#d5d9df",
    )
    + badge(
        f"待人工确认 {len(cases) - human_confirmed} 条",
        fg="#a15c07" if len(cases) - human_confirmed else "#1a7f37",
        bg="#fff6e5" if len(cases) - human_confirmed else "#e9f7ee",
        border="#f2d08a" if len(cases) - human_confirmed else "#a7e0b8",
    ),
    unsafe_allow_html=True,
)
st.info(
    "**这些标注怎么来的**：凡涉及文献的用例，真值都用 Crossref **实测**确认过"
    "（该 DOI 是否存在、实际发表年份是多少），并把实测结论写进每条用例的「真值来源」字段。"
    f"其中 {human_confirmed} 条已经过人工逐条确认，"
    f"仍有 {len(cases) - human_confirmed} 条为 AI 起草、等待确认——"
    "在**全部**确认之前，请不要把整体指标对外表述为「人工标注结果」。",
)

with st.expander(f"查看全部 {len(cases)} 条标注（含真值来源、标注者与人工确认意见）", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "编号": case.case_id,
                    "类别": case.category,
                    "文本": case.text,
                    "标注片段": case.gold_claim_quote,
                    "标注状态": case.gold_status.value,
                    "标注问题类型": "、".join(item.value for item in case.gold_problem_types),
                    "真值来源": case.source_of_truth,
                    "标注者": case.labeler,
                    "人工确认": "已确认" if case.human_confirmed else "待确认",
                    "确认意见": case.human_comment,
                }
                for case in cases
            ]
        ),
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------- 运行评测
st.divider()
st.markdown("#### 运行评测")

run_col1, run_col2, run_col3 = st.columns([2, 2, 1])
with run_col1:
    baseline_label = st.radio(
        "选择评测配置",
        [BASELINE_B2, BASELINE_B1, BASELINE_B0],
        index=0,
        help="B2 是完整流程（模型 + 外部检索）；B1 是离线规则引擎 + 不检索（免费下界）；B0 是直接问模型、不给任何外部证据（对照「直接问模型」）。",
    )
with run_col2:
    chosen_categories = st.multiselect(
        "限制类别（留空 = 全部）", category_list, default=[]
    )
    limit = st.number_input(
        "最多跑多少条（先小样本试跑可省钱）",
        min_value=1,
        max_value=len(cases),
        value=len(cases),
        step=1,
    )
with run_col3:
    st.write("")
    run_clicked = st.button("运行评测", type="primary", key="run_evaluation")

selected_count = len(
    [case for case in cases if not chosen_categories or case.category in chosen_categories]
)
selected_count = min(selected_count, int(limit))
if baseline_label == BASELINE_B2 and settings.llm_provider != "mock":
    st.caption(
        f"预计调用模型 {selected_count * 2}–{selected_count * 4} 次"
        "（每条用例：拆解 + 事实审查 + 可能还有证据审查/修正/复核）。"
        "模型与检索结果都会缓存，**重复评测不再重复付费**。"
    )
else:
    st.caption(f"当前配置不调用真实模型（{selected_count} 条用例，零成本）。")

if run_clicked:
    active_settings = (
        offline_settings(settings) if baseline_label == BASELINE_B1 else settings
    )
    progress_bar = st.progress(0.0, text="准备中…")
    status_box = st.status("正在评测…", expanded=True)

    def on_case(index: int, total: int, case) -> None:  # noqa: ANN001
        progress_bar.progress(index / max(total, 1), text=f"[{index}/{total}] {case.case_id}")
        status_box.write(f"{case.case_id}（{case.category}）：{case.text[:40]}…")

    try:
        report = run_evaluation(
            settings=active_settings,
            cases=cases,
            baseline=baseline_label,
            limit=int(limit),
            only_categories=chosen_categories or None,
            on_case=on_case,
        )
    except VeriMindError as exc:
        status_box.update(label="评测失败", state="error")
        st.error(f"评测过程中出错：{exc}")
    else:
        progress_bar.progress(1.0, text="评测完成")
        status_box.update(label=f"评测完成（{report.case_count} 条用例）", state="complete")
        st.session_state["evaluation_report"] = report
        st.session_state.setdefault("evaluation_history", []).append(report)

# ---------------------------------------------------------------- 已保存的结果
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
saved_files = sorted(EVALUATIONS_DIR.glob("*.json")) if EVALUATIONS_DIR.exists() else []

st.divider()
st.markdown("#### 查看评测结果")
SESSION_OPTION = "（本次会话刚运行的）"
source_options = ([SESSION_OPTION] if st.session_state.get("evaluation_report") else []) + [
    item.name for item in saved_files
]
if source_options:
    st.caption(
        "evaluations/ 目录下的每一份 JSON 都是一次真实运行的存档，可以直接查看，不需要重跑（也就不会花钱）。"
    )
    source = st.selectbox("数据来源", source_options, index=0, key="evaluation_source")
else:
    source = SESSION_OPTION

report: EvaluationReport | None = None
if source == SESSION_OPTION:
    report = st.session_state.get("evaluation_report")
else:
    try:
        raw = json.loads((EVALUATIONS_DIR / source).read_text(encoding="utf-8"))
        report = EvaluationReport.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - 存档损坏时给出可读提示，不让整页崩掉
        st.error(f"读取评测存档失败：{exc}")
        report = None

# ---------------------------------------------------------------- 结果
if report is None:
    st.info("还没有评测结果。选择配置后点「运行评测」即可——先跑 3–5 条小样本确认流程正常，再跑全量。")
    st.stop()

if report.metric("total_llm_calls") == 0 and settings.llm_provider != "mock":
    st.warning(
        "这份存档的模型调用为 **0 次**（全部命中本地缓存），因此「平均耗时」与「token」两项反映的是缓存路径，"
        "不代表冷启动成本。冷启动实测数据见 docs/07-experiment-and-defense.md。",
    )

st.divider()
st.markdown(f"#### 评测结果：{report.baseline}")
st.caption(
    f"运行时间：{report.created_at.astimezone():%Y-%m-%d %H:%M:%S}　｜　"
    f"引擎：{report.engine_summary}　｜　用例数：{report.case_count}"
)

metrics = report.metrics


def show(value: object, *, percent: bool = True) -> str:
    if value is None:
        return "—"
    if percent and isinstance(value, float):
        return f"{value * 100:.1f}%"
    return str(value)


cards = st.columns(5)
cards[0].metric("检测精确率", show(metrics.get("detection_precision")))
cards[1].metric("检测召回率", show(metrics.get("detection_recall")))
cards[2].metric("误报率 FPR", show(metrics.get("false_positive_rate")))
cards[3].metric("状态完全一致", show(metrics.get("status_accuracy")))
cards[4].metric("证据可追溯率", show(metrics.get("evidence_traceability_rate")))

cards2 = st.columns(5)
cards2[0].metric("修正成功率", show(metrics.get("correction_success_rate")))
cards2[1].metric("剩余错误率", show(metrics.get("residual_error_rate")))
cards2[2].metric("新增错误率", show(metrics.get("new_error_rate")))
cards2[3].metric("平均耗时(ms)", show(metrics.get("avg_latency_ms"), percent=False))
cards2[4].metric("平均模型调用", show(metrics.get("avg_llm_calls_per_case"), percent=False))

st.caption(
    "各指标定义：**检测精确率/召回率**以「确实有问题」（存在矛盾 / 部分支持 / 来源冲突）为正类；"
    "**误报率** = 本身没问题却被判成有问题的比例（本项目最看重的诚实指标）；"
    "**修正成功率** = 标注期望修正的用例里，修正后文本既包含应出现的正确内容、又不残留错误内容的比例；"
    "**新增错误率** = 修正后出现了原文与证据里都没有的事实（年份/数值/DOI）的用例占比。"
)

# 混淆矩阵
st.markdown("##### 混淆矩阵（主张级）")
confusion = pd.DataFrame(
    [
        [metrics.get("true_positive", 0), metrics.get("false_negative", 0)],
        [metrics.get("false_positive", 0), metrics.get("true_negative", 0)],
    ],
    index=["实际：有问题", "实际：没问题"],
    columns=["预测：有问题", "预测：没问题"],
)
st.dataframe(confusion, width="stretch")
st.caption(
    f"未命中标注片段的用例：{metrics.get('unmatched_count', 0)} 条"
    "（拆解环节没找到与标注对齐的那条主张——这本身就是一个需要如实报告的结果）。"
)

# 分类别
st.markdown("##### 分类别统计")
by_category = metrics.get("by_category") or {}
st.dataframe(
    pd.DataFrame(
        [
            {
                "类别": name,
                "用例数": item["cases"],
                "对齐成功": item["matched"],
                "状态一致": item["correct"],
                "状态一致率": "—" if not item["matched"] else f"{item['correct'] / item['matched'] * 100:.0f}%",
            }
            for name, item in by_category.items()
        ]
    ),
    width="stretch",
    hide_index=True,
)

# 逐条结果
st.markdown("##### 逐条结果")
category_filter = st.multiselect(
    "按类别筛选", sorted({row.category for row in report.rows}), default=[], key="row_filter"
)
rows = [row for row in report.rows if not category_filter or row.category in category_filter]
st.dataframe(
    pd.DataFrame(
        [
            {
                "编号": row.case_id,
                "类别": row.category,
                "标注": row.gold_status,
                "预测": row.predicted_status or "未命中",
                "一致": "是" if row.gold_status == row.predicted_status else "否",
                "对齐到的主张": row.matched_claim_quote or "—",
                "预测问题类型": "、".join(row.predicted_problem_types) or "—",
                "证据数": len(row.evidence_ids),
                "修正/采用": f"{row.corrections}/{row.corrections_applied}",
                "新增事实": "、".join(row.new_fact_tokens) or "—",
                "耗时(ms)": row.latency_ms,
                "模型调用": row.llm_calls,
            }
            for row in rows
        ]
    ),
    width="stretch",
    hide_index=True,
)

if report.failures:
    st.markdown(f"##### 失败案例（{len(report.failures)} 条）")
    for item in report.failures:
        st.markdown(f"- **{item['case_id']}**（{item['kind']}）：{item['detail']}")
else:
    st.success("本次评测没有失败案例（样本量小，这并不代表系统在所有情况下都可靠）。")

st.markdown("##### 实验说明与局限")
for caveat in report.caveats:
    st.markdown(f"- {caveat}")

st.divider()
st.markdown("##### 基线说明")
st.markdown(
    f"- **{BASELINE_B2}**：本项目的完整流程（拆解 → 检索 → 裁决 → 修正 → 复核）。\n"
    f"- **{BASELINE_B1}**：离线规则引擎 + 不检索，**完全免费**，用来给出一个下界参照"
    "（如果完整流程连它都赢不了，那就说明外部证据与模型没有带来价值）。\n"
    f"- **{BASELINE_B0}**：同一个模型、单次调用、**不给任何外部证据**——对照「直接问模型」这件事本身。它的证据可追溯率必然为 0（定义如此），而误报率显著高于完整流程。\n"
    "宁可缺一行，也不编一行。"
)
st.caption(
    "状态取值顺序：" + " → ".join(item.value for item in STATUS_DISPLAY_ORDER)
)

render_footer()
