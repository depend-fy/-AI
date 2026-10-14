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
        icon="●", fg="#1a7f37", bg="#e9f7ee", border="#a7e0b8",
        desc="现有证据能够支持该主张。",
    ),
    VerificationStatus.PARTIALLY_SUPPORTED: StatusStyle(
        icon="●", fg="#a15c07", bg="#fff6e5", border="#f2d08a",
        desc="证据只支持主张的一部分（例如方向对、数字不对）。",
    ),
    VerificationStatus.CONTRADICTED: StatusStyle(
        icon="●", fg="#b42318", bg="#fdecea", border="#f3b3ad",
        desc="找到可信证据与该主张冲突。",
    ),
    VerificationStatus.SOURCE_CONFLICT: StatusStyle(
        icon="●", fg="#c2410c", bg="#fff1e6", border="#f5c09a",
        desc="不同可信来源之间给出了不一致的信息。",
    ),
    VerificationStatus.INSUFFICIENT: StatusStyle(
        icon="●", fg="#3a5a8c", bg="#eef3fb", border="#b9cbe8",
        desc="已有材料不足以支持或反驳该主张。",
    ),
    VerificationStatus.UNVERIFIABLE: StatusStyle(
        icon="●", fg="#5f6b7a", bg="#f4f5f7", border="#d5d9df",
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
    RoleStance.SUPPORT: "支持",
    RoleStance.REFUTE: "反对",
    RoleStance.UNCERTAIN: "未定",
    RoleStance.NOT_APPLICABLE: "—",
}


RUN_MODE_DESCRIPTION: dict[RunMode, tuple[str, str, str]] = {
    # 模式 -> (图标位保留, 颜色, 说明)；正式界面不显示装饰性图标，故图标位留空
    RunMode.MOCK: (
        "",
        "#5f6b7a",
        "离线模拟：不调用任何真实大模型，用本地模拟结果跑通流程。用于开发与断网演示，结果不具备事实核验意义。",
    ),
    RunMode.SINGLE_MODEL_MULTI_ROLE: (
        "",
        "#3a5a8c",
        "单模型多角色模拟：同一个模型分别扮演事实审查员 / 证据审查员 / 独立复核员。"
        "角色之间有提示词与流程隔离，但**不是真正的多模型协同**。",
    ),
    RunMode.MULTI_PROVIDER: (
        "",
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
        f'<div style="font-weight:600;color:{color};margin-bottom:4px;">当前运行模式：{mode.value}</div>'
        f'<div style="color:#333;font-size:0.9rem;">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_config_warnings(settings: Settings) -> None:
    """把配置层面的风险如实告诉用户（不隐藏、不美化）。"""
    for message in settings.runtime_warnings:
        st.warning(message)


def section(title: str, caption: str | None = None, *, divider: bool = True) -> None:
    """统一的小标题样式。"""
    if divider:
        st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


def safe_page_link(page: str, *, label: str, icon: str | None = None) -> None:
    """多页跳转链接（带降级保护）。

    为什么需要它：`st.page_link` 的路径是**相对于主入口文件**解析的。
    如果某个页面被单独运行（开发时很常见的 `streamlit run pages/2_检测报告.py`），
    或者运行在不注册页面表的环境里，它会抛出 StreamlitPageNotFoundError 并把整页打挂。
    与其崩掉，不如降级成一句提示——用户仍然能从左侧导航打开目标页面。
    """
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - 任何跳转失败都不值得让页面崩溃
        st.caption(f"（当前运行方式下无法生成跳转链接，请在左侧页面导航中打开「{label}」）")


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
        f"本页尚未实现，计划在 **{planned_stage}** 完成。\n\n"
        f"当前不会显示任何检测结果，页面不展示任何虚构数据。",
    )

    st.subheader("这个页面将来会显示什么")
    for item in will_do:
        st.markdown(f"- {item}")

    st.subheader("为什么这样设计")
    st.markdown(why)

    st.subheader("依赖条件")
    st.markdown(dependency)


# --------------------------------------------------------------------------
# 成品化：统一视觉、页面头、页脚
# --------------------------------------------------------------------------
_CSS = """
<style>
  /* ============================================================
     排版基线：衬线标题 + 系统字体正文/元数据（学术出版物惯例）
     字体族优先使用中文衬线（宋体/思源宋体），回退到通用衬线
     ============================================================ */
  :root {
      --vm-ink: #1a1a1a;          /* 正文墨色 */
      --vm-ink-soft: #44484d;     /* 次级文字 */
      --vm-meta: #6b7075;         /* 元数据 */
      --vm-rule: #d8dbdf;         /* 细分隔线 */
      --vm-rule-soft: #e8eaed;
      --vm-accent: #1a3d63;       /* 单一强调色（深蓝） */
      --vm-paper: #ffffff;
      --vm-wash: #f7f8f9;         /* 极浅底 */
      --vm-serif: "Songti SC", "SimSun", "Source Han Serif SC", "Noto Serif CJK SC",
                  Georgia, "Times New Roman", serif;
      --vm-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei",
                 "PingFang SC", "Hiragino Sans GB", sans-serif;
  }

  /* 内容列：出版物式窄栏 + 充足留白 */
  .block-container {
      max-width: 1120px;
      padding-top: 2.6rem;
      padding-bottom: 4rem;
  }

  /* ---------- 标题：衬线、字重对比、句首大写 ---------- */
  h1, h2, h3, h4, h5 {
      font-family: var(--vm-serif) !important;
      color: var(--vm-ink) !important;
      letter-spacing: -0.01em;
  }
  h1 {font-size: 2.0rem !important; font-weight: 700 !important; line-height: 1.25 !important;}
  h2 {font-size: 1.45rem !important; font-weight: 700 !important; line-height: 1.3 !important;
      margin-top: 1.6rem !important;}
  h3 {font-size: 1.15rem !important; font-weight: 700 !important; line-height: 1.35 !important;
      margin-top: 1.4rem !important;}
  h4 {font-size: 1.02rem !important; font-weight: 700 !important; margin-top: 1.2rem !important;}
  h5 {font-size: 0.98rem !important; font-weight: 700 !important; color: var(--vm-ink-soft) !important;}

  /* ---------- 正文 ---------- */
  .stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"] p {
      font-family: var(--vm-sans);
      color: var(--vm-ink);
      font-size: 0.95rem;
      line-height: 1.72;
  }
  .stMarkdown li {margin-bottom: 0.28rem;}

  /* ---------- 交互元素与元数据：系统字体、克制 ---------- */
  button, .stButton button, [data-testid="stBaseButton-secondary"],
  [data-testid="stBaseButton-primary"], input, textarea, select {
      font-family: var(--vm-sans) !important;
  }
  .stButton button {
      border-radius: 3px !important;
      border: 1px solid var(--vm-rule) !important;
      font-weight: 500 !important;
      box-shadow: none !important;
  }
  .stButton button:hover {border-color: var(--vm-accent) !important; color: var(--vm-accent) !important;}
  /* 主按钮：比正文强调色浅一档，避免在浅色页面上显得过重 */
  [data-testid="stBaseButton-primary"] {
      background: #e8eef5 !important;
      border: 1px solid #b9c9dc !important;
      color: #1a3d63 !important;
      font-weight: 600 !important;
  }
  [data-testid="stBaseButton-primary"]:hover {
      background: #dbe6f1 !important;
      border-color: #8fa9c6 !important;
      color: #14314f !important;
  }
  [data-testid="stBaseButton-primary"]:focus:not(:active) {
      border-color: #8fa9c6 !important;
      color: #14314f !important;
  }

  /* ---------- 指标卡：去掉彩色底，改为出版物式数据行 ---------- */
  div[data-testid="stMetric"] {
      background: transparent;
      border: none;
      border-top: 2px solid var(--vm-ink);
      border-radius: 0;
      padding: 10px 2px 6px 2px;
  }
  div[data-testid="stMetricLabel"] p {
      font-family: var(--vm-sans);
      font-size: 0.78rem !important;
      color: var(--vm-meta) !important;
      letter-spacing: 0.02em;
  }
  div[data-testid="stMetricValue"] {
      font-family: var(--vm-serif) !important;
      font-weight: 700 !important;
      color: var(--vm-ink) !important;
  }

  /* ---------- 分隔线：细分隔线是学术排版的主要结构手段 ---------- */
  hr, [data-testid="stDivider"] hr {
      border: none;
      border-top: 1px solid var(--vm-rule);
      margin: 1.5rem 0;
  }

  /* ---------- 折叠区：像论文附录 ---------- */
  details, [data-testid="stExpander"] details {
      border: 1px solid var(--vm-rule-soft) !important;
      border-radius: 3px !important;
      background: var(--vm-paper);
  }
  [data-testid="stExpander"] summary {
      font-family: var(--vm-sans);
      font-size: 0.9rem;
      color: var(--vm-ink-soft);
  }

  /* ---------- 表格：三线表（学术论文表格惯例） ---------- */
  [data-testid="stDataFrame"] {border: none !important;}
  table {border-collapse: collapse; font-family: var(--vm-sans); font-size: 0.88rem;}
  thead tr th {
      border-top: 2px solid var(--vm-ink) !important;
      border-bottom: 1px solid var(--vm-ink) !important;
      background: transparent !important;
      color: var(--vm-ink) !important;
      font-weight: 700 !important;
  }
  tbody tr td {border-bottom: 1px solid var(--vm-rule-soft) !important;}
  tbody tr:last-child td {border-bottom: 2px solid var(--vm-ink) !important;}

  /* ---------- 代码块：等宽、浅底、细边 ---------- */
  code, pre, .stCode {
      font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace !important;
      font-size: 0.84rem !important;
  }
  pre {border: 1px solid var(--vm-rule-soft) !important; border-radius: 3px !important;}

  /* ---------- 提示条：去掉彩色气泡感，改为左边线 + 浅底 ---------- */
  [data-testid="stAlert"] {
      border-radius: 0 3px 3px 0 !important;
      border-left: 3px solid var(--vm-accent) !important;
      background: var(--vm-wash) !important;
      box-shadow: none !important;
  }
  [data-testid="stAlert"] p {font-size: 0.9rem;}

  /* ---------- 页头：出版物题头（细线分隔，不用渐变） ---------- */
  .vm-hero {
      border: none;
      border-bottom: 1px solid var(--vm-rule);
      border-left: none;
      border-radius: 0;
      padding: 0 0 1.1rem 0;
      background: transparent;
      margin-bottom: 1.4rem;
  }
  .vm-hero h1 {
      font-family: var(--vm-serif);
      margin: 0 0 0.5rem 0;
      font-size: 2.1rem;
      font-weight: 700;
      line-height: 1.25;
      color: var(--vm-ink);
      letter-spacing: -0.015em;
  }
  .vm-hero p {
      margin: 0;
      color: var(--vm-ink-soft);
      font-size: 0.98rem;
      line-height: 1.7;
      max-width: 62em;
  }
  .vm-step {
      display: inline-block;
      padding: 2px 0;
      margin: 0 14px 4px 0;
      border-radius: 0;
      background: transparent;
      border: none;
      border-bottom: 1px solid var(--vm-rule);
      color: var(--vm-meta);
      font-family: var(--vm-sans);
      font-size: 0.8rem;
      letter-spacing: 0.01em;
  }

  /* ---------- 页脚：小字、细线、克制 ---------- */
  .vm-footer {
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--vm-rule);
      color: var(--vm-meta);
      font-family: var(--vm-sans);
      font-size: 0.78rem;
      line-height: 1.75;
  }

  /* ---------- 徽标：降饱和、去圆角气泡 ---------- */
  .vm-badge, span[style*="border-radius:12px"] {
      border-radius: 2px !important;
      font-family: var(--vm-sans);
      letter-spacing: 0.01em;
  }
</style>
"""


def inject_css() -> None:
    """注入统一的样式（每个页面开头调用一次）。"""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, points: list[str] | None = None) -> None:
    """页面题头：学术出版物式（衬线标题 + 细线分隔 + 关键词行）。

    刻意不用渐变、不用彩色卡片——把注意力留给内容本身。
    """
    body = f"<p>{subtitle}</p>"
    if points:
        chips = "".join(f'<span class="vm-step">{item}</span>' for item in points)
        body += f'<div style="margin-top:14px;">{chips}</div>'
    st.markdown(
        f'<div class="vm-hero"><h1>{title}</h1>{body}</div>',
        unsafe_allow_html=True,
    )


def render_footer(*, extra: str | None = None) -> None:
    """页脚声明：适用范围与边界。每页统一出现。"""
    notes = [
        "本系统的输出为**基于公开可检索证据的辅助判断**，不构成学术结论、法律意见或事实认定的最终依据。",
        "「文献是否存在」与「文献内容是否支持该表述」为两个独立问题；「已修正」与「修正正确」亦为两个独立问题，"
        "后者一律须经独立复核。",
        "证据不足或无法核验时予以明确标注，不输出推测性结论，亦不提供未经校准的概率化评分。",
    ]
    if extra:
        notes.append(extra)
    items = "".join(f"<div>{item}</div>" for item in notes)
    st.markdown(f'<div class="vm-footer">{items}</div>', unsafe_allow_html=True)
