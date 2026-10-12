"""检测工作台：输入待核验文本并启动检测。

阶段 1 状态：占位页。阶段 2 实现「文本输入 → 主张拆解 → 结构化结果」的离线闭环。
"""

from __future__ import annotations

import streamlit as st

from verimind.ui.theme import placeholder_page

st.set_page_config(page_title="检测工作台 · 智鉴 AI", page_icon="🧪", layout="wide")

placeholder_page(
    title="🧪 检测工作台",
    planned_stage="阶段 2（基础检测流程）",
    will_do=[
        "多行文本输入框（带实时字数统计，上限由配置决定）",
        "示例文本按钮：一键载入预设的待核验内容（含虚构 DOI、真实文献错年份、证据不足三类）",
        "检测强度选择：快速模式（只做检索核验）/ 标准模式（走完整闭环）",
        "「开始检测」按钮 + 逐步进度显示（拆解中 → 检索中 → 判定中 → 修正中 → 复核中）",
        "错误提示：模型调用失败、检索失败、文本过长、主张数超限等情况的具体原因",
        "检测完成后跳转到「检测报告」页",
    ],
    why=(
        "工作台只负责**收集输入**和**展示进度**，不做任何业务判断——"
        "所有检测逻辑都在 `verimind/core/pipeline.py` 里，界面只传一个 `run_id`。\n\n"
        "这样拆分的好处：业务逻辑可以被 pytest 直接测试，不需要启动网页；"
        "将来要改成 FastAPI 接口或做别的界面，业务代码一行都不用动。"
    ),
    dependency=(
        "- 阶段 2 起可在**完全离线**（不接任何大模型）的情况下跑通，用离线模拟模式演示流程\n"
        "- 接入真实模型需要配置 `.env` 中的 `VERIMIND_LLM_API_KEY`（阶段 3）\n"
        "- 检索外部证据需要能访问 Crossref / OpenAlex / arXiv（阶段 4）"
    ),
)
