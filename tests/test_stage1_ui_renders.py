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
    "pages/4_展示与答辩.py",
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

    assert not app.exception

    # 首页标题改用自定义 hero 渲染（不再用 st.title），因此断言 Markdown 内容
    markdown_text = "\n".join(element.value for element in app.markdown)
    assert "智鉴 AI" in markdown_text
    assert "当前运行模式" in markdown_text
    assert any(
        mode in markdown_text for mode in ("离线模拟", "单模型多角色模拟", "多模型协同")
    ), "运行模式必须是三种明确模式之一，不能含糊"

    # 首页必须如实声明当前能力边界（现以「当前能力 / 不足与缺陷」两块呈现）
    assert "不足与缺陷" in markdown_text, "首页必须如实声明做不到什么"
    assert "当前能力" in markdown_text, "首页必须声明已实现的能力"
    assert "中文文献" in markdown_text, "必须写明中文文献无法核验"

    # 成品体验：首页要有一键演示入口（不用先粘文本也能看到完整效果）
    assert any(button.key == "one_click_demo" for button in app.button)


def test_evaluation_page_shows_the_labeled_dataset() -> None:
    """评测页必须能真实渲染，并把"标注是谁做的"如实展示出来。

    注意：这一页**不会**在打开时自动运行评测（那会直接花钱），
    因此这里断言"只有概览卡片、还没有指标卡片"。
    """
    script = "pages/3_评测.py"
    app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=180)
    app.run()
    assert not app.exception

    markdown_all = "\n".join(element.value for element in app.markdown)
    assert "测试集构成" in markdown_all
    assert "人工已确认" in markdown_all, "必须显示有多少条标注已经过人工确认"

    info_text = "\n".join(element.value for element in app.info)
    assert "AI 起草" in info_text or "人工逐条确认" in info_text, (
        "标注的来源与确认状态必须如实写明，不能含糊成「人工标注」"
    )

    labels = [element.label for element in app.metric]
    assert "用例总数" in labels

    # evaluations/ 下有存档时，页面可以直接展示而不必重跑（省钱）；
    # 没有任何存档时才要求"不出现指标数字"。
    has_saved = (PROJECT_ROOT / "evaluations").exists() and any(
        (PROJECT_ROOT / "evaluations").glob("*.json")
    )
    if not has_saved:
        assert "检测精确率" not in labels, "没有存档、也没点运行时不应出现任何指标数字"

    assert any(button.key == "run_evaluation" for button in app.button)


def test_presentation_page_renders_the_stage7_materials() -> None:
    """「展示与答辩」页必须真的把三份材料渲染出来（单一数据源，改文档即改网页）。

    这样一旦 docs/ 里的文件被改名或删除，测试会立刻失败，而不是页面上悄悄显示一个报错框。
    """
    app = AppTest.from_file(str(PROJECT_ROOT / "pages/4_展示与答辩.py"), default_timeout=180)
    app.run()
    assert not app.exception

    markdown_all = "\n".join(element.value for element in app.markdown)
    assert "五个核心创新点" in markdown_all, "架构与创新点材料没有被渲染"
    assert "四幕脚本" in markdown_all, "演示脚本材料没有被渲染"
    assert "答辩 Q&A" in markdown_all, "实验与答辩材料没有被渲染"

    labels = [element.label for element in app.metric]
    assert "运行模式" in labels, "必须展示实时运行模式"
    assert "用例总数" in labels, "必须展示标注集的实时确认进度"


def test_implemented_pages_do_not_claim_to_be_placeholders() -> None:
    """已经实现的页面不能还挂着"尚未实现"的牌子——那同样是不诚实的展示。"""
    for script in ("pages/1_检测工作台.py", "pages/2_检测报告.py"):
        app = AppTest.from_file(str(PROJECT_ROOT / script), default_timeout=120)
        app.run()
        info_text = "\n".join(element.value for element in app.info)
        assert "尚未实现" not in info_text, f"{script} 已经是实现页，却仍显示占位提示"

def test_report_page_renders_a_real_report_with_evidence_and_corrections(tmp_path: Path) -> None:
    """回归测试：报告页在**真的有报告**时必须能渲染完整内容。

    这个用例是补出来的：之前 236 个测试全绿，但报告页走的都是「没有报告」的分支，
    于是 components.py 里 render_evidence_item 用到的 EvidenceRelation 一直没被导入
    （NameError），用户一在浏览器里打开报告就崩。
    教训：**UI 测试必须喂进真实数据，只测空状态等于没测。**
    """
    import json as _json

    from verimind.core.config import Settings as _Settings
    from verimind.core.enums import (
        ChangeType as _ChangeType,
        ReviewCheckType as _CheckType,
        ReviewOverall as _Overall,
        RoleName as _RoleName,
        RoleStance as _Stance,
    )
    from verimind.core.pipeline import run_detection as _run
    from verimind.core.schemas import (
        Correction as _Correction,
        ReviewCheck as _ReviewCheck,
        ReviewResult as _ReviewResult,
        RolePosition as _RolePosition,
    )

    # 1) 用离线引擎真跑一份带「文内矛盾 + 内部证据」的报告（零成本、不联网）
    samples = PROJECT_ROOT / "examples" / "sample_inputs.jsonl"
    text = ""
    for line in samples.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = _json.loads(line)
            if item.get("id") == "internal_conflict":
                text = str(item["text"])
                break
    assert text, "找不到 internal_conflict 样例"

    settings = _Settings(
        _env_file=None,
        data_dir=tmp_path,
        save_runs=False,
        llm_provider="mock",
        retrieve_enabled=False,
    )
    report = _run(text, settings=settings)
    assert report.claims and report.evidence, "样例应当产生主张与内部证据"

    # 2) 再挂一条「已复核但未采用」的修正，把修正区的渲染路径也覆盖到
    claim = report.claims[0]
    correction = _Correction(
        correction_id="K001",
        claim_id=claim.claim_id,
        original_span=claim.source_span,
        proposed_text="",
        change_type=_ChangeType.DELETE,
        rationale="回归测试用：删除该片段",
    )
    review = _ReviewResult(
        review_id="V001",
        target_correction_id="K001",
        checks=[
            _ReviewCheck(check=_CheckType.CORRECTION_EVIDENCE_BACKED, passed=True, detail="测试：有证据"),
            _ReviewCheck(check=_CheckType.EVIDENCE_MATCHES_TEXT, passed=True, detail="测试：匹配"),
            _ReviewCheck(check=_CheckType.NO_NEW_FACT_ERROR, passed=True, detail="测试：无新事实"),
            _ReviewCheck(check=_CheckType.ORIGINAL_ISSUE_RESOLVED, passed=False, detail="测试：未解决"),
            _ReviewCheck(check=_CheckType.NO_NEW_OVERREACH, passed=True, detail="测试：无过度推断"),
        ],
        overall=_Overall.FAILED,
        unresolved=["需要人工确认"],
        reviewer=_RolePosition(
            role=_RoleName.INDEPENDENT_REVIEWER,
            provider="test",
            model="test-model",
            stance=_Stance.REFUTE,
            conclusion="未通过",
        ),
    )
    report = report.model_copy(
        update={
            "corrections": [correction],
            "review_results": [review],
            "rejected_correction_ids": ["K001"],
        }
    )

    # 3) 把报告塞进会话状态后再渲染报告页
    app = AppTest.from_file(str(PROJECT_ROOT / "pages/2_检测报告.py"), default_timeout=180)
    app.session_state["last_report"] = report
    app.run()
    messages = [str(element.value) for element in app.exception]
    assert not app.exception, f"报告页使用真实报告渲染时报错：{messages}"
