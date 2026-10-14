"""智鉴 AI（VeriMind AI）· 首页

运行方式（在项目根目录）：
    .\\.venv\\Scripts\\streamlit.exe run app.py

本页只展示两件事：
    1. 项目要解决什么问题、怎么做、边界在哪；
    2. 当前这台电脑上系统的**真实**运行状态（配置、模式、进度）。
不展示任何虚构的指标或演示数据——检测功能要到阶段 2 才可用。
"""

from __future__ import annotations

import streamlit as st

from verimind import __version__
from verimind.core.config import Settings
from verimind.evaluation.dataset import dataset_stats
from verimind.core.enums import (
    EXPECTED_PROBLEM_TYPE_COUNT,
    EXPECTED_STATUS_COUNT,
    ProblemType,
    STATUS_DISPLAY_ORDER,
)
from verimind.ui.theme import (
    render_config_warnings,
    render_run_mode_banner,
    inject_css,
    hero,
    render_footer,
    safe_page_link,
    status_badge_html,
)

st.set_page_config(
    page_title="智鉴 AI（VeriMind AI）",
    page_icon="🔍",
    layout="wide",
)

settings = Settings()
inject_css()
dataset = dataset_stats()

# ---------------------------------------------------------------- 页头
hero(
    "智鉴 AI（VeriMind AI）",
    "多模型协同的 <b>AI 幻觉检测、证据溯源、错误归因与自动修正</b>系统"
    f"　<span style='color:#6b7480;font-size:0.9rem;'>Demo 版 v{__version__}</span><br>"
    "对文本中的事实性主张进行证据核验、错误归因与可复核的修正。"
    "系统不将模型判断作为结论：每项判定均须经外部证据、明确规则与独立复核。",
    [
        "① 拆解主张",
        "② 检索真实文献",
        "③ 区分「存在」与「内容被支持」",
        "④ 可解释规则裁决",
        "⑤ 最小改动修正",
        "⑥ 跨厂商独立复核",
        "⑦ 20 条标注集评测",
    ],
)

render_run_mode_banner(settings)
render_config_warnings(settings)

# ---------------------------------------------------------------- 首屏入口
st.markdown("#### 开始核验")
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    safe_page_link("pages/1_检测工作台.py", label="检测工作台", icon=None)
    st.caption("粘贴待核验文本，启动核验")
with nav2:
    safe_page_link("pages/2_检测报告.py", label="检测报告", icon=None)
    st.caption("查看主张、证据与修正建议")
with nav3:
    safe_page_link("pages/3_评测.py", label="评测", icon=None)
    st.caption("人工标注测试集上的实测指标")
with nav4:
    safe_page_link("pages/4_展示与答辩.py", label="展示与答辩", icon=None)
    st.caption("技术架构、实验报告与答辩材料")

# ---------------------------------------------------------------- 首屏只留入口
# 问题定义、技术路线、与常见做法的区别、核验流程、运行环境与配置明细
# 已移至「展示与答辩」页的「答辩备查」档（见 verimind/ui/method_notes.py）。
st.caption("技术路线、方法比较、核验流程与运行环境明细，见「展示与答辩 · 答辩备查」。")
safe_page_link("pages/4_展示与答辩.py", label="查看方法与答辩材料", icon=None)



# ---------------------------------------------------------------- 状态说明
st.divider()
st.markdown("#### 六类核验状态")
st.caption(
    f"每种状态都有明确含义，并且必须给出理由与证据（共 {EXPECTED_STATUS_COUNT} 种状态、"
    f"{EXPECTED_PROBLEM_TYPE_COUNT} 类问题类型）。"
)
badges = "".join(status_badge_html(s) for s in STATUS_DISPLAY_ORDER)
st.markdown(badges, unsafe_allow_html=True)

with st.expander("查看 8 类问题类型"):
    for problem in ProblemType:
        st.markdown(f"- {problem.value}")



# ---------------------------------------------------------------- 快速体验
st.divider()
st.markdown("#### 样例核验")
st.caption(
    "执行完整流程：主张拆分、外部检索、规则裁决、修正建议与独立复核。"
    "内置样例为「真实 DOI 与注册记录不符的年份」。"
)
if st.button("执行样例核验", key="one_click_demo", type="primary"):
    import json as _json
    from pathlib import Path as _Path

    from verimind.core.errors import VeriMindError as _VErr
    from verimind.core.pipeline import run_detection as _run_detection

    _samples = _Path(__file__).resolve().parent / "examples" / "sample_inputs.jsonl"
    _demo_text = ""
    if _samples.exists():
        for _line in _samples.read_text(encoding="utf-8").splitlines():
            if _line.strip():
                _item = _json.loads(_line)
                if _item.get("id") == "mixed_status_demo":
                    _demo_text = str(_item["text"])
                    break
    if not _demo_text:
        st.error("找不到样例文本 examples/sample_inputs.jsonl，请改用「检测工作台」。")
    else:
        with st.spinner("正在执行流水线（命中本地缓存时约 0.3 秒）…"):
            try:
                _report = _run_detection(_demo_text, settings=settings)
            except _VErr as exc:
                st.error(f"演示运行失败：{exc}")
            else:
                st.session_state["last_report"] = _report
                st.session_state["last_run_id"] = _report.run_id
                st.success(
                    f"完成：{_report.stats.total_claims} 条主张，"
                    f"判定有问题 {_report.stats.suspicious_claims} 条，"
                    f"需关注 {_report.stats.flagged_claims} 条；"
                    f"模型调用 {_report.metrics.llm_calls} 次（缓存命中 {_report.metrics.cache_hits} 次）。"
                )
                safe_page_link("pages/2_检测报告.py", label="查看这份报告", icon=None)
st.caption(
    "该样例的模型与检索结果通常会命中本地缓存（**0 次计费调用**）；"
    "若缓存已失效则会真实调用模型与外部检索，并在报告中如实记录次数与 token。"
)




# ---------------------------------------------------------------- 项目进度（诚实版）
st.divider()
st.markdown("#### 项目进度")
st.markdown(
    """
| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 0 | 需求冻结与技术方案 | 已完成 |
| 阶段 1 | 开发环境与项目骨架 | 已完成 |
| 阶段 2 | 基础检测流程（离线可跑） | 已完成 |
| 阶段 3 | 接入真实大模型（交叉比对、缓存、成本统计） | 已完成 |
| 阶段 4 | 文献检索与证据核验（Crossref / OpenAlex / arXiv） | 已完成 |
| 阶段 5 | 修正与独立复核（角色 C 修正 / 角色 D 复核） | 已完成 |
| 阶段 6 | 测试与评测（人工标注集、指标、基线对比） | 已完成 |
| 阶段 7 | 比赛展示准备（演示脚本、架构、实验报告、答辩材料） | 已完成 |
"""
)
st.info(
    "**当前能力（阶段 7 完成）**：完整闭环已经打通——拆解主张 → 检索外部证据 → 可解释规则裁决 → "
    "生成最小改动的修正建议 → **跨厂商独立复核** → 只有复核通过的修改才会写进「修正后全文」；"
    "并在 20 条人工标注集上给出真实指标与失败案例。\n\n"
    "**两条关键边界**：「文献是真的」与「文献真的这么说」是两个问题；"
    "「改了」与「改对了」也是两个问题。修正是**代码级的保守**——"
    "编造的数字会被确定性检查拦下，复核未通过的修改一律不改。\n\n"
    "**尚未做到**：论文全文阅读（现在最多读到摘要）、中文文献检索（数据源没有公开 API）、"
    f"非学术事实核验（需要付费检索服务）；标注集共 {dataset.total} 条，其中 {dataset.confirmed} 条已经人工逐条确认。",
)

st.divider()
st.caption(
    "免责声明：本项目为 Demo，用于方法验证与竞赛演示；系统输出是辅助判断，"
    "不构成学术结论、法律意见或事实认定的最终依据。"
)

render_footer()
