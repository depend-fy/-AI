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
    assert "适用范围与局限" in markdown_all, "报告必须展示适用范围与局限"
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

def test_claim_consensus_is_derived_from_existing_role_data() -> None:
    """共识程度必须由**已有的**角色立场与证据状态推出，不得引入任何概率数字。"""
    from verimind.core.enums import EvidenceSource, RoleName, RoleStance, VerificationStatus
    from verimind.core.schemas import RolePosition, SearchRecord, SearchTrace, VerificationResult
    from verimind.ui.components import claim_consensus

    def position(role: RoleName, stance: RoleStance) -> RolePosition:
        return RolePosition(
            role=role,
            provider="test",
            model="test-model",
            stance=stance,
            conclusion="测试结论",
        )

    trace = SearchTrace(records=[SearchRecord(source=EvidenceSource.CROSSREF, query="测试", status="检索成功")])

    # 只有规则层结论，没有角色意见
    alone = VerificationResult(
        claim_id="C001",
        status=VerificationStatus.INSUFFICIENT,
        rationale="测试",
        missing_evidence=["需要更多证据"],
        search_trace=trace,
    )
    assert claim_consensus(alone)[0] == "单层判定"

    # 两个角色意见相左
    split = VerificationResult(
        claim_id="C002",
        status=VerificationStatus.INSUFFICIENT,
        rationale="测试",
        missing_evidence=["需要更多证据"],
        search_trace=trace,
        role_positions=[
            position(RoleName.FACT_CHECKER, RoleStance.SUPPORT),
            position(RoleName.EVIDENCE_CHECKER, RoleStance.REFUTE),
        ],
    )
    label, detail, *_ = claim_consensus(split)
    assert label == "存在分歧"
    assert "事实审查员" in detail and "证据审查员" in detail, "分歧必须指明是谁与谁不一致"

    # 意见一致，但没有任何外部证据
    no_evidence = VerificationResult(
        claim_id="C003",
        status=VerificationStatus.INSUFFICIENT,
        rationale="测试",
        missing_evidence=["需要更多证据"],
        search_trace=trace,
        role_positions=[position(RoleName.FACT_CHECKER, RoleStance.SUPPORT)],
    )
    assert claim_consensus(no_evidence)[0] == "证据不足"

    # 意见一致且有证据
    agreed = VerificationResult(
        claim_id="C004",
        status=VerificationStatus.VERIFIED,
        rationale="测试",
        supporting_evidence_ids=["E001"],
        search_trace=trace,
        role_positions=[position(RoleName.FACT_CHECKER, RoleStance.SUPPORT)],
    )
    assert claim_consensus(agreed)[0] == "高度一致"

def test_evidence_strength_levels_are_rule_based() -> None:
    """证据强度必须由规则判定：A=有原文片段并形成判断，B=仅元数据，C=只有线索，无=没有资料。"""
    import uuid

    from verimind.core.enums import (
        EvidenceLevel,
        EvidenceRelation,
        EvidenceSource,
        JudgedAspect,
        VerificationStatus,
    )
    from verimind.core.schemas import (
        Evidence,
        SearchRecord,
        SearchTrace,
        Span,
        VerificationResult,
    )
    from verimind.verification.evidence_strength import assess_evidence_strength

    trace = SearchTrace(
        records=[SearchRecord(source=EvidenceSource.CROSSREF, query="测试", status="检索成功")]
    )

    def result(supporting: list[str], refuting: list[str], related: list[str]) -> VerificationResult:
        return VerificationResult(
            claim_id="C001",
            status=VerificationStatus.VERIFIED if supporting else VerificationStatus.INSUFFICIENT,
            rationale="测试",
            supporting_evidence_ids=supporting,
            refuting_evidence_ids=refuting,
            related_evidence_ids=related,
            missing_evidence=[] if supporting else ["测试"],
            search_trace=trace,
        )

    def evidence(identifier: str, *, level: EvidenceLevel, relation: EvidenceRelation, aspect=None, snippet=None) -> Evidence:
        return Evidence(
            evidence_id=identifier,
            claim_ids=["C001"],
            source=EvidenceSource.OPENALEX,
            level=level,
            url="https://example.org/x",
            title="测试文献",
            year=2020,
            relation=relation,
            judged_aspect=aspect,
            snippet=snippet,
            relation_rationale="测试",
        )

    # 无证据
    assert assess_evidence_strength(result([], [], []), []).level == "未获证据"

    # A：摘要级证据 + 内容层面判断 + 有片段
    item_a = evidence(
        "E001",
        level=EvidenceLevel.ABSTRACT,
        relation=EvidenceRelation.REFUTES,
        aspect=JudgedAspect.CONTENT,
        snippet="摘要中的原始片段",
    )
    verdict_a = assess_evidence_strength(result([], ["E001"], []), [item_a])
    assert verdict_a.level == "A", verdict_a

    # B：仅元数据、且只有一个来源
    item_b = evidence("E002", level=EvidenceLevel.METADATA_ONLY, relation=EvidenceRelation.SUPPORTS, aspect=JudgedAspect.BIBLIOGRAPHIC)
    assert assess_evidence_strength(result(["E002"], [], []), [item_b]).level == "B"

    # C：只有相关资料、未形成支持或反驳判断
    item_c = evidence("E003", level=EvidenceLevel.METADATA_ONLY, relation=EvidenceRelation.UNKNOWN)
    assert assess_evidence_strength(result([], [], ["E003"]), [item_c]).level == "C"
