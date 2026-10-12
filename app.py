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
from verimind.core.enums import (
    EXPECTED_PROBLEM_TYPE_COUNT,
    EXPECTED_STATUS_COUNT,
    ProblemType,
    STATUS_DISPLAY_ORDER,
)
from verimind.ui.theme import render_config_warnings, render_run_mode_banner, status_badge_html

st.set_page_config(
    page_title="智鉴 AI（VeriMind AI）",
    page_icon="🔍",
    layout="wide",
)

settings = Settings()

# ---------------------------------------------------------------- 页头
st.title("🔍 智鉴 AI（VeriMind AI）")
st.markdown(
    "**多模型协同的 AI 幻觉检测、证据溯源、错误归因与自动修正系统**（Demo 版，v"
    f"{__version__}）"
)
st.caption(
    "把「相信一个模型」换成「一条可审计的校验流水线」——"
    "每条结论都能追问：你凭什么这么说？"
)

render_run_mode_banner(settings)
render_config_warnings(settings)

# ---------------------------------------------------------------- 核心逻辑
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ❗ 要解决的问题")
    st.markdown(
        "大模型生成的内容可能存在：\n"
        "- 事实错误\n"
        "- 虚构文献 / 不存在的 DOI\n"
        "- 引用与结论不匹配\n"
        "- 数字、时间、人物、机构错误\n"
        "- 逻辑推断过度\n"
        "- 证据不足或来源互相冲突\n\n"
        "而**同一个模型很难可靠地识别自己的错误**。"
    )

with col2:
    st.markdown("#### 🔧 我们的做法")
    st.markdown(
        "把「检测幻觉」拆成 7 个职责单一、可单独测试的环节：\n"
        "1. 拆解为原子事实主张\n"
        "2. 多角色协同分析\n"
        "3. 检索可信外部来源\n"
        "4. 匹配证据与主张\n"
        "5. 规则裁决：状态 + 问题类型\n"
        "6. 基于证据生成修正\n"
        "7. 独立复核修正结果\n\n"
        "每个环节的输入输出都是**结构化数据**。"
    )

with col3:
    st.markdown("#### ⭐ 与常见做法的区别")
    st.markdown(
        "- **证据可点击追溯**：报告中每条判断都能点开到真实来源\n"
        "- **模型不下最终结论**：大模型只做结构化判断，状态由可解释的规则表裁决\n"
        "- **无证据不判错**：把「证据不足 / 暂无法核验」当作一等公民状态\n"
        "- **敢展示失败**：内置修正后独立复核，如实报告「修正失败 / 引入新错误」\n"
        "- **不给假概率**：没有校准过的概率模型，就不显示貌似精确的「幻觉概率」"
    )

# ---------------------------------------------------------------- 状态说明
st.divider()
st.markdown("#### 📊 报告中的 6 种核验状态")
st.caption(
    f"每种状态都有明确含义，并且必须给出理由与证据（共 {EXPECTED_STATUS_COUNT} 种状态、"
    f"{EXPECTED_PROBLEM_TYPE_COUNT} 类问题类型）。"
)
badges = "".join(status_badge_html(s) for s in STATUS_DISPLAY_ORDER)
st.markdown(badges, unsafe_allow_html=True)

with st.expander("查看 8 类问题类型"):
    for problem in ProblemType:
        st.markdown(f"- {problem.value}")

# ---------------------------------------------------------------- 流程
st.divider()
st.markdown("#### 🔄 检测流程（闭环）")
st.markdown(
    """
```text
用户输入文本
   → ① 拆解为原子事实主张（带原文位置）
   → ② 多角色协同：事实审查员提出核验问题
   → ③ 外部检索：Crossref / OpenAlex / arXiv（免密钥公开接口）
   → ④ 证据审查员：判断证据是支持 / 部分支持 / 反驳 / 无关
   → ⑤ 规则裁决：6 种状态 + 8 类问题 + 严重度 + 缺失证据清单
   → ⑥ 基于证据生成修正（无证据不许改事实）
   → ⑦ 独立复核：原问题是否解决 / 是否引入新错误
   → 可追溯检测报告（含未解决问题与系统局限）
```
"""
)

# ---------------------------------------------------------------- 导航
st.divider()
st.markdown("#### 🚀 开始使用")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.page_link("pages/1_检测工作台.py", label="检测工作台", icon="🧪")
    st.caption("粘贴待核验文本，启动检测")
with nav2:
    st.page_link("pages/2_检测报告.py", label="检测报告", icon="📄")
    st.caption("查看主张、证据、问题与修正")
with nav3:
    st.page_link("pages/3_评测.py", label="评测", icon="📈")
    st.caption("在人工标注集上的真实指标")

# ---------------------------------------------------------------- 真实环境状态
st.divider()
st.markdown("#### 🖥️ 当前环境与配置（真实读取，非示意）")

env_col1, env_col2 = st.columns(2)
with env_col1:
    st.markdown(
        f"""
| 配置项 | 当前值 |
| --- | --- |
| 运行模式 | **{settings.run_mode.value}** |
| 大模型 provider | `{settings.llm_provider}` |
| 大模型 model | `{settings.llm_model}` |
| API Key | `{settings.masked_api_key}` |
| 单次模型调用上限 | {settings.llm_max_calls_per_run} 次 |
| 文献检索 | {"已启用" if settings.retrieve_enabled else "已关闭"}（Crossref / OpenAlex / arXiv，免密钥） |
"""
    )
with env_col2:
    st.markdown(
        f"""
| 运行限制 | 当前值 |
| --- | --- |
| 最大输入字数 | {settings.max_input_chars} |
| 单次最多核验主张数 | {settings.max_claims} |
| 检索超时 | {settings.retrieval_timeout_seconds} 秒 |
| 检索结果缓存 | {settings.retrieval_cache_hours} 小时 |
| 保存检测记录 | {"是" if settings.save_runs else "否"} |
| 数据目录 | `{settings.data_dir}` |
"""
    )

with st.expander("环境自检与配置方法"):
    st.markdown(
        "**一条命令做完整环境自检**（含依赖版本、配置、可选的外部 API 连通性）：\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\check_env.py --network\n"
        "```\n"
        "**要接入真实大模型**（阶段 3 才需要）：\n"
        "```powershell\n"
        "Copy-Item .env.example .env     # 然后编辑 .env，填入 VERIMIND_LLM_API_KEY\n"
        "```\n"
        "> `.env` 已被 `.gitignore` 排除，密钥不会进入 Git；界面上只显示脱敏值。"
    )

# ---------------------------------------------------------------- 项目进度（诚实版）
st.divider()
st.markdown("#### 🗺️ 项目进度（如实标注）")
st.markdown(
    """
| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 0 | 需求冻结与技术方案 | ✅ 已完成 |
| 阶段 1 | 开发环境与项目骨架 | ✅ 已完成（当前） |
| 阶段 2 | 基础检测流程（离线可跑） | ⏳ 未开始 |
| 阶段 3 | 接入真实大模型 | ⏳ 未开始 |
| 阶段 4 | 文献检索与证据核验 | ⏳ 未开始 |
| 阶段 5 | 修正与独立复核 | ⏳ 未开始 |
| 阶段 6 | 测试与 Demo 完善 | ⏳ 未开始 |
| 阶段 7 | 比赛展示准备 | ⏳ 未开始 |
"""
)
st.info(
    "**当前状态说明**：阶段 1 只完成了工程骨架与环境验证。检测流程、证据检索、修正与复核"
    "均为**尚未实现**，因此「检测工作台 / 检测报告 / 评测」三页目前是占位页，不显示任何结果。"
    "这样做的原因很简单：宁可页面空着，也不放看起来像真数据的假内容。",
    icon="ℹ️",
)

st.divider()
st.caption(
    "免责声明：本项目为 Demo，用于方法验证与竞赛演示；系统输出是辅助判断，"
    "不构成学术结论、法律意见或事实认定的最终依据。"
)
