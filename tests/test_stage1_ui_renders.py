"""阶段 1 界面冒烟测试：4 个页面必须都能真实渲染，且不抛任何异常。

用 Streamlit 官方的 AppTest（不需要浏览器）真正执行页面脚本：
    · 能抓出"页面看起来没报错、其实一打开就崩"的问题
    · 不需要人工点界面，也不会引入假数据

运行：
    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGE_SCRIPTS = [
    "app.py",
    "pages/1_检测工作台.py",
    "pages/2_检测报告.py",
    "pages/3_评测.py",
]


@pytest.mark.parametrize("script", PAGE_SCRIPTS)
def test_page_renders_without_exception(script: str) -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=120)
    app.run()
    messages = [str(element.value) for element in app.exception]
    assert not app.exception, f"{script} 渲染时抛出异常：{messages}"


def test_home_page_shows_real_project_info() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    app.run()

    titles = [element.value for element in app.title]
    assert any("智鉴 AI" in title for title in titles)

    # 首页必须如实声明当前运行模式（默认无密钥 → 离线模拟）
    markdown_text = "\n".join(element.value for element in app.markdown)
    assert "当前运行模式" in markdown_text
    assert "离线模拟" in markdown_text

    # 首页必须明确说明检测功能尚未实现，不允许让占位页看起来像已经能用了
    info_text = "\n".join(element.value for element in app.info)
    assert "尚未实现" in info_text


def test_placeholder_pages_refuse_to_fake_results() -> None:
    """三个占位页都必须显式说明"本页尚未实现"，不允许显示编造的数据。"""
    for script in PAGE_SCRIPTS[1:]:
        app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=120)
        app.run()
        info_text = "\n".join(element.value for element in app.info)
        assert "尚未实现" in info_text, f"{script} 没有说明自己是占位页"
