"""阶段 2 测试：界面真的能跑（用 Streamlit 官方 AppTest，不需要浏览器）。

界面测试专门盯住两类问题：
    · 能跑：输入文本 → 点开始检测 → 真的出结果、真的存进 session_state；
    · 诚实：页面上必须标注"离线模拟"，不能假装调用了大模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from verimind.core.config import Settings
from verimind.core.pipeline import run_detection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = "pages/1_检测工作台.py"
REPORT = "pages/2_检测报告.py"

SAMPLES = [
    json.loads(line)
    for line in (PROJECT_ROOT / "examples" / "sample_inputs.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
SAMPLE_BY_ID = {sample["id"]: sample for sample in SAMPLES}
CONFLICT_SAMPLE = SAMPLE_BY_ID["internal_conflict"]


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """让界面测试写到临时目录，别污染真实的 data/verimind.db。

    同时关闭外部检索：界面测试不应该真的去打 Crossref / OpenAlex
    （检索行为由 test_stage4_*.py 用 respx 模拟验证）。
    """
    monkeypatch.setenv("VERIMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VERIMIND_LLM_PROVIDER", "mock")
    monkeypatch.setenv("VERIMIND_LLM_API_KEY", "")
    monkeypatch.setenv("VERIMIND_RETRIEVE_ENABLED", "false")


def open_page(script: str) -> AppTest:
    app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=180)
    app.run()
    return app


# ==========================================================================
# 工作台
# ==========================================================================
def test_workbench_loads_sample_text_by_button() -> None:
    app = open_page(WORKBENCH)
    assert not app.exception
    app.button(key="load_internal_conflict").click().run()
    assert not app.exception
    assert app.text_area(key="workbench_text").value == CONFLICT_SAMPLE["text"]


def test_workbench_runs_detection_and_stores_report() -> None:
    app = open_page(WORKBENCH)
    app.text_area(key="workbench_text").set_value(CONFLICT_SAMPLE["text"]).run()
    app.button(key="run_detection").click().run()

    assert not app.exception
    report = app.session_state["last_report"]
    assert report is not None
    assert app.session_state["last_run_id"] == report.run_id
    assert report.stats.total_claims >= 4
    assert report.stats.suspicious_claims >= 4, "文内矛盾应当被判定为问题"

    assert any("检测完成" in element.value for element in app.success)
    assert any("核验结果摘要" in element.value for element in app.markdown)


def test_workbench_warns_instead_of_running_on_empty_input() -> None:
    app = open_page(WORKBENCH)
    app.button(key="run_detection").click().run()
    assert not app.exception
    assert any("请先输入" in element.value for element in app.warning)
    assert not app.success, "空输入时不能出现成功提示"


def test_workbench_declares_offline_mode() -> None:
    """界面上必须如实标注当前模式，否则用户会以为真的调用了大模型。"""
    app = open_page(WORKBENCH)
    markdown_all = "\n".join(element.value for element in app.markdown)
    assert "当前运行模式：离线模拟" in markdown_all


# ==========================================================================
# 报告页
# ==========================================================================
def test_report_page_renders_a_real_report(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, save_runs=False)
    report = run_detection(CONFLICT_SAMPLE["text"], settings=settings)

    app = AppTest.from_file(str(PROJECT_ROOT / REPORT), default_timeout=180)
    app.session_state["last_report"] = report
    app.run()

    assert not app.exception
    assert any("检测报告" in element.value for element in app.title)

    markdown_all = "\n".join(element.value for element in app.markdown)
    assert "本次运行模式：离线模拟" in markdown_all
    assert "系统局限" in markdown_all, "报告必须展示系统局限"
    assert "原文内部" in markdown_all or "相关证据" in markdown_all

    # 统计卡片与报告数据一致（不是页面自己数的）
    metrics = {element.label: element.value for element in app.metric}
    assert str(metrics.get("主张总数")) == str(report.stats.total_claims)
    assert str(metrics.get("已判定有问题")) == str(report.stats.suspicious_claims)


def test_report_page_tells_user_to_run_detection_first() -> None:
    app = open_page(REPORT)
    assert not app.exception
    assert any("还没有可查看的检测报告" in element.value for element in app.info)


def test_report_page_hides_report_when_nothing_selected() -> None:
    """没有报告时应当明确提示并停止，而不是渲染一堆空卡片。"""
    app = open_page(REPORT)
    assert not app.metric, "没有报告时不应出现统计卡片"
