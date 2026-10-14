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
from verimind.ui.method_notes import render_competition_alignment
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
    "面向科研文本的 <b>事实核验 Agent</b>：对论文、综述与研究报告中的事实性主张逐条核验，"
    "输出可追溯的证据、可解释的判定与可复核的修正建议。"
    f"　<span style='color:#6b7480;font-size:0.9rem;'>v{__version__}</span><br>"
    "系统不将模型判断作为结论：每项判定均须经外部证据、明确规则与独立复核。",
    [
        "① 任务规划",
        "② 工具调用",
        "③ 执行反馈",
        "④ 结果验证",
        "⑤ 异常处理",
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



# ---------------------------------------------------------------- 任务闭环对照
st.divider()
render_competition_alignment()

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




# ---------------------------------------------------------------- 能力与不足
st.divider()
st.markdown("#### 当前能力")
st.markdown(
    """
| 能力 | 实现方式 |
| --- | --- |
| 主张拆分 | 真实大模型拆解为可单独核验的原子主张（含字符级原文位置），规则引擎独立拆解一遍，两者差异全部记录 |
| 外部检索 | 真实调用 Crossref / OpenAlex / arXiv，核对文献是否存在、年份与期刊是否一致；检索痕迹与原始响应留档可查 |
| 内容核验 | 取得摘要时判断文献内容是否支持该表述，**必须引用原文片段**才算数 |
| 文内矛盾 | 检出同一文献两个年份、同一指标两个数值等自相矛盾，无需外部证据即可确认 |
| 规则裁决 | 6 种状态 / 8 类问题 / 严重度由可解释的规则表决定，每条结论显示命中的规则编号 |
| 修正建议 | 生成最小改动的修改方案；改事实必须有证据，否则只能弱化或标注不确定性 |
| 独立复核 | 由**另一家厂商的模型**复核修正（三项确定性检查 + 两项语义检查），未通过或未完成一律不写入正文 |
| 评测 | 100 条标注测试集上的三基线对照实验（含直接问模型的对照），失败案例全部公开 |
"""
)
st.markdown("#### 不足与缺陷")
st.markdown(
    """
- **只读到摘要**：未通读全文，摘要层面的判断不等于全文核实；拿不到摘要时只核验文献是否存在与书目信息，**不判断内容**；
- **中文文献无法核验**：Crossref / OpenAlex 未提供中文库的公开接口，此类来源如实标注「暂无法核验」；
- **非学术事实无法核验**：政策、新闻、产品参数等需付费检索服务，当前不在覆盖范围内；
- **作者与机构类核验未实现**：目前仅实现年份与期刊比对；
- **跨语言期刊名不判断**：中文刊名与英文刊名的比对误报风险高，系统选择保守处理；
- **检索召回仍有限**：100 条测试中 23 条状态不一致，含漏检与误判，已全部公开；
- **标注一致性未计算**：新增 80 条为口头确认，未做双人独立标注（kappa）；
- **成本未折算金额**：项目不内置单价，仅记录真实 token 用量。
"""
)
st.divider()
st.caption(
    "免责声明：本项目为 Demo，用于方法验证与竞赛演示；系统输出是辅助判断，"
    "不构成学术结论、法律意见或事实认定的最终依据。"
)

render_footer()
