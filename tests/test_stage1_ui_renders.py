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


@pytest.fixture(autouse=True)
def isolate_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """界面渲染测试不该依赖开发机上的本地数据，也不该真的去检索外部来源。"""
    monkeypatch.setenv("VERIMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VERIMIND_RETRIEVE_ENABLED", "false")


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

    # 首页必须如实声明当前运行模式（具体是哪种取决于 .env 配置，因此只断言"明确标注了"）
    markdown_text = "\n".join(element.value for element in app.markdown)
    assert "当前运行模式" in markdown_text
    assert any(
        mode in markdown_text for mode in ("离线模拟", "单模型多角色模拟", "多模型协同")
    ), "运行模式必须是三种明确模式之一，不能含糊"

    # 首页必须如实声明当前能力边界（阶段 2：未接入外部检索与修正复核）
    info_text = "\n".join(element.value for element in app.info)
    assert "尚未接入" in info_text


def test_evaluation_page_is_still_an_honest_placeholder() -> None:
    """尚未实现的页面必须显式声明"本页尚未实现"，不允许显示编造的数据。

    阶段 2 之后，工作台与报告页已经实现；「评测」页要到阶段 6 才有真实数据，
    因此这里只对它做断言。
    """
    script = "pages/3_评测.py"
    app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=120)
    app.run()
    info_text = "\n".join(element.value for element in app.info)
    assert "尚未实现" in info_text, f"{script} 没有说明自己是占位页"


def test_implemented_pages_do_not_claim_to_be_placeholders() -> None:
    """已经实现的页面不能还挂着"尚未实现"的牌子——那同样是不诚实的展示。"""
    for script in ("pages/1_检测工作台.py", "pages/2_检测报告.py"):
        app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=120)
        app.run()
        info_text = "\n".join(element.value for element in app.info)
        assert "尚未实现" not in info_text, f"{script} 已经是实现页，却仍显示占位提示"
