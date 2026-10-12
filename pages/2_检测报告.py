"""检测报告：展示主张、证据、问题类型、修正前后对比与复核结论。

阶段 1 状态：占位页。阶段 2 实现基础报告（主张 + 理由 + 定位），
阶段 4/5 补齐证据链接、修正对比与复核结果。
"""

from __future__ import annotations

import streamlit as st

from verimind.core.enums import STATUS_DISPLAY_ORDER
from verimind.ui.theme import placeholder_page, render_status_legend

st.set_page_config(page_title="检测报告 · 智鉴 AI", page_icon="📄", layout="wide")

placeholder_page(
    title="📄 检测报告",
    planned_stage="阶段 2（基础报告）→ 阶段 4/5（证据、修正与复核）",
    will_do=[
        "顶部统计卡片：主张总数、6 种核验状态各自的数量（数字由实际结果自动统计）",
        "可疑主张列表（按严重度排序），点击可展开",
        "每条主张的完整卡片：原文高亮定位 → 规范化主张 → 核验问题 → 判断理由 → 命中的裁决规则",
        "证据表格：来源名称、标题、层级（仅元数据 / 摘要 / 全文）、检索时间、**可点击的原始链接**",
        "三个检测角色的结论与分歧（同一模型多角色时也会如实标注）",
        "修正前后对比：原文 / 建议文本 / 修改理由 / 关联证据 / 该修改自身的风险",
        "修正后独立复核的 5 项检查结论；未通过时明确展示失败原因",
        "尚未解决的问题清单与系统局限声明（例如「未获得全文，仅依据摘要判断」）",
        "导出完整报告 JSON（可复现、可归档）",
    ],
    why=(
        "报告是本 Demo 最重要的展示部分，也是最容易造假的部分。因此定下三条硬规则：\n\n"
        "1. **统计数字只能由结果算出**：`DetectionReport.recompute()` 统一计算，"
        "页面不允许手填任何指标。\n"
        "2. **每条结论必须挂证据**：数据模型层已强制校验——"
        "判定「存在矛盾 / 部分支持」时若没有证据 ID，程序会直接报错。\n"
        "3. **没有的东西就留白**：证据不足就显示「证据不足」并说明还缺什么，"
        "不硬凑一个看起来完整的故事。"
    ),
    dependency=(
        "- 需要先在工作台完成一次检测（阶段 2 起）\n"
        "- 报告落盘在本地 `data/` 目录，刷新页面不会丢失（阶段 2 实现）"
    ),
)

st.divider()
st.markdown("#### 报告中的状态含义")
render_status_legend()
st.caption(
    "状态展示顺序固定为：" + " → ".join(s.value for s in STATUS_DISPLAY_ORDER)
)
