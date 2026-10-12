"""界面主题与通用组件。

为什么把配色/徽标集中在一个文件？
    检测状态有 6 种，报告、工作台、评测页都要用同一套颜色和图标。
    散落在各页面里迟早会出现"同一状态两种颜色"。集中定义 = 一致性。

注意：这里只做"展示"，不做任何业务判断。
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from verimind.core.config import Settings
from verimind.core.enums import (
    STATUS_DISPLAY_ORDER,
    RoleStance,
    RunMode,
    Severity,
    VerificationStatus,
)


@dataclass(frozen=True)
class StatusStyle:
    """一种核验状态的展示样式。"""

    icon: str
    fg: str
    bg: str
    border: str
    desc: str


#: 6 种核验状态的统一视觉定义。
#: 原则：绿色=有证据支持；黄色=部分支持；红色=有矛盾；橙色=来源打架；
#:       蓝灰=证据不足；灰色=查不到。颜色只表达"证据强度"，不表达"模型信心"。
STATUS_STYLE: dict[VerificationStatus, StatusStyle] = {
    VerificationStatus.VERIFIED: StatusStyle(
        icon="✅", fg="#1a7f37", bg="#e9f7ee", border="#a7e0b8",
        desc="现有证据能够支持该主张。",
    ),
    VerificationStatus.PARTIALLY_SUPPORTED: StatusStyle(
        icon="🟡", fg="#a15c07", bg="#fff6e5", border="#f2d08a",
        desc="证据只支持主张的一部分（例如方向对、数字不对）。",
    ),
    VerificationStatus.CONTRADICTED: StatusStyle(
        icon="🔴", fg="#b42318", bg="#fdecea", border="#f3b3ad",
        desc="找到可信证据与该主张冲突。",
    ),
    VerificationStatus.SOURCE_CONFLICT: StatusStyle(
        icon="🟠", fg="#c2410c", bg="#fff1e6", border="#f5c09a",
        desc="不同可信来源之间给出了不一致的信息。",
    ),
    VerificationStatus.INSUFFICIENT: StatusStyle(
        icon="🔵", fg="#3a5a8c", bg="#eef3fb", border="#b9cbe8",
        desc="已有材料不足以支持或反驳该主张。",
    ),
    VerificationStatus.UNVERIFIABLE: StatusStyle(
        icon="⚪", fg="#5f6b7a", bg="#f4f5f7", border="#d5d9df",
        desc="当前检索条件下无法作出可靠判断（注意：这不等于「错误」）。",
    ),
}


@dataclass(frozen=True)
class SeverityStyle:
    label: str
    fg: str
    bg: str


SEVERITY_STYLE: dict[Severity, SeverityStyle] = {
    Severity.HIGH: SeverityStyle("高", "#b42318", "#fdecea"),
    Severity.MEDIUM: SeverityStyle("中", "#a15c07", "#fff6e5"),
    Severity.LOW: SeverityStyle("低", "#3a5a8c", "#eef3fb"),
    Severity.NONE: SeverityStyle("无", "#5f6b7a", "#f4f5f7"),
}


STANCE_ICON: dict[RoleStance, str] = {
    RoleStance.SUPPORT: "👍",
    RoleStance.REFUTE: "👎",
    RoleStance.UNCERTAIN: "❔",
    RoleStance.NOT_APPLICABLE: "—",
}


RUN_MODE_DESCRIPTION: dict[RunMode, tuple[str, str, str]] = {
    # 模式 -> (图标, 颜色, 说明)
    RunMode.MOCK: (
        "🧪",
        "#5f6b7a",
        "离线模拟：不调用任何真实大模型，用本地模拟结果跑通流程。用于开发与断网演示，结果不具备事实核验意义。",
    ),
    RunMode.SINGLE_MODEL_MULTI_ROLE: (
        "🧩",
        "#3a5a8c",
        "单模型多角色模拟：同一个模型分别扮演事实审查员 / 证据审查员 / 独立复核员。"
        "角色之间有提示词与流程隔离，但**不是真正的多模型协同**。",
    ),
    RunMode.MULTI_PROVIDER: (
        "🔀",
        "#1a7f37",
        "多模型协同：不同角色由不同供应商/模型承担，独立复核员与生成方不是同一个模型。",
    ),
}


# --------------------------------------------------------------------------
# 组件
# --------------------------------------------------------------------------
def badge(text: str, *, fg: str, bg: str, border: str) -> str:
    """生成一个内联徽标 HTML。"""
    return (
        f'<span style="display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;'
        f'border-radius:12px;font-size:0.85rem;line-height:1.5;'
        f'color:{fg};background:{bg};border:1px solid {border};white-space:nowrap;">{text}</span>'
    )


def status_badge_html(status: VerificationStatus) -> str:
    """状态徽标（报告里到处都要用）。"""
    style = STATUS_STYLE[status]
    return badge(f"{style.icon} {status.value}", fg=style.fg, bg=style.bg, border=style.border)


def severity_badge_html(severity: Severity) -> str:
    style = SEVERITY_STYLE[severity]
    return badge(f"严重度 {style.label}", fg=style.fg, bg=style.bg, border=style.fg + "33")


def render_status_legend() -> None:
    """6 种状态的图例。放在报告页显眼处，让评委一眼看懂颜色含义。"""
    parts = []
    for status in STATUS_DISPLAY_ORDER:
        style = STATUS_STYLE[status]
        parts.append(
            f'<div style="margin:6px 0;">{status_badge_html(status)}'
            f'<span style="color:#444;font-size:0.88rem;">{style.desc}</span></div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_run_mode_banner(settings: Settings) -> None:
    """如实展示本次运行的模型模式——这是本项目的诚信底线，必须显眼。"""
    mode = settings.run_mode
    icon, color, text = RUN_MODE_DESCRIPTION[mode]
    st.markdown(
        f'<div style="border-left:5px solid {color};background:{color}14;'
        f'padding:10px 14px;border-radius:6px;margin:8px 0 16px 0;">'
        f'<div style="font-weight:600;color:{color};margin-bottom:4px;">{icon} 当前运行模式：{mode.value}</div>'
        f'<div style="color:#333;font-size:0.9rem;">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_config_warnings(settings: Settings) -> None:
    """把配置层面的风险如实告诉用户（不隐藏、不美化）。"""
    for message in settings.runtime_warnings:
        st.warning(message, icon="⚠️")


def section(title: str, caption: str | None = None, *, divider: bool = True) -> None:
    """统一的小标题样式。"""
    if divider:
        st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


def placeholder_page(
    *,
    title: str,
    planned_stage: str,
    will_do: list[str],
    why: str,
    dependency: str,
) -> None:
    """尚未实现的页面占位内容。

    刻意写清楚"这里现在没有内容、什么时候会有"，
    而不是摆一堆假图表让页面看起来"很完整"——那样只会误导演示与答辩。
    """
    st.title(title)
    st.info(
        f"🚧 本页尚未实现。计划在 **{planned_stage}** 完成。\n\n"
        f"当前不会显示任何检测结果，避免出现「看起来像真数据其实不是」的内容。",
        icon="🚧",
    )

    st.subheader("这个页面将来会显示什么")
    for item in will_do:
        st.markdown(f"- {item}")

    st.subheader("为什么这样设计")
    st.markdown(why)

    st.subheader("依赖条件")
    st.markdown(dependency)
