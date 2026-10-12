"""阶段 6 测试：标注集、指标计算与评测运行器。

重点守住三件事：
    · **标注集自洽**：每条用例的"标注片段"必须真的出现在文本里，否则评测根本对不上；
      每条都必须写清真值来源与标注者；
    · **指标算法正确**：精确率/召回率/误报率/修正成功率/新增错误率都用手算过的例子验证；
      分母为 0 时必须是 None（界面显示「—」），不许用 0 或 1 蒙混；
    · **评测跑的就是生产线**：用离线配置真跑一遍，确认能产出报告，并且把样本量与标注者局限写进结论。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from verimind.core.config import Settings
from verimind.core.enums import (
    ClaimType,
    EvidenceLevel,
    EvidenceSource,
    ProblemType,
    RunMode,
    SearchStatus,
    VerificationStatus,
)
from verimind.core.schemas import (
    Claim,
    DetectionReport,
    Evidence,
    SearchRecord,
    SearchTrace,
    Span,
    VerificationResult,
)
from verimind.evaluation.dataset import GoldenCase, load_golden_set, match_claim
from verimind.evaluation.metrics import CaseOutcome, compute_metrics, summarize_failures
from verimind.evaluation.runner import BASELINE_B1, offline_settings, run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT_ROOT / "examples" / "golden_set.jsonl"


# ==========================================================================
# 标注集自洽性
# ==========================================================================
def test_golden_set_loads_and_is_self_consistent() -> None:
    cases = load_golden_set(GOLDEN)
    assert len(cases) >= 20, "测试集至少要 20 条"

    for case in cases:
        assert case.gold_claim_quote in case.text, (
            f"{case.case_id} 的标注片段「{case.gold_claim_quote}」不在文本里，评测将无法对齐"
        )
        assert case.source_of_truth, f"{case.case_id} 缺少真值来源说明"
        assert case.labeler, f"{case.case_id} 缺少标注者"
        if case.gold_evidence_doi:
            assert case.gold_evidence_doi.startswith("10."), f"{case.case_id} 的 DOI 格式可疑"


def test_golden_set_covers_all_required_categories() -> None:
    cases = load_golden_set(GOLDEN)
    covered = {case.category for case in cases}
    required = {
        "正确事实",
        "明确错误事实",
        "虚构文献",
        "真实文献但错误引用",
        "证据不足",
        "来源冲突",
        "修正质量",
    }
    assert required <= covered, f"缺少这些类别：{required - covered}"


def test_golden_set_flags_ai_drafted_labels() -> None:
    """AI 起草的标注必须能被识别；已人工确认的必须有确认记录。"""
    cases = load_golden_set(GOLDEN)
    assert any(case.labeler_is_ai for case in cases)
    assert all(
        "待人工抽查确认" in case.labeler
        for case in cases
        if case.labeler_is_ai and not case.human_confirmed
    ), "未经人工确认的标注必须写明「待人工抽查确认」"

    confirmed = [case for case in cases if case.human_confirmed]
    assert confirmed, "至少应该有用例经过人工确认"
    for case in confirmed:
        assert "人工确认" in case.labeler
        assert case.human_comment, f"{case.case_id} 标为已确认，却没有确认意见"
        assert case.awaiting_confirmation is False


# ==========================================================================
# 预测与标注的对齐
# ==========================================================================
def make_report(claim_quote: str, status: VerificationStatus, text: str) -> DetectionReport:
    """构造一份最小但**合法**的报告（数据契约要求"已核实"必须挂支持证据）。"""
    start = text.index(claim_quote)
    claim = Claim(
        claim_id="C001",
        source_span=Span.from_text(text, start, start + len(claim_quote)),
        parent_sentence_span=Span.from_text(text, 0, len(text)),
        normalized_claim="（测试）",
        claim_type=ClaimType.METADATA_YEAR,
        verification_questions=["是否正确？"],
    )
    evidence = Evidence(
        evidence_id="E001",
        claim_ids=["C001"],
        source=EvidenceSource.CROSSREF,
        level=EvidenceLevel.METADATA_ONLY,
        url="https://api.crossref.org/works/10.1038/nature14539",
        title="Deep learning",
        year=2015,
    )
    trace = SearchTrace(
        records=[SearchRecord(source=None, query="测试", status=SearchStatus.SKIPPED)]
    )
    if status is VerificationStatus.CONTRADICTED:
        result = VerificationResult(
            claim_id="C001",
            status=status,
            problem_types=[ProblemType.METADATA_ERROR],
            rationale="（测试）",
            refuting_evidence_ids=["E001"],
            search_trace=trace,
        )
    else:
        result = VerificationResult(
            claim_id="C001",
            status=status,
            rationale="（测试）",
            supporting_evidence_ids=["E001"],
            search_trace=trace,
        )
    return DetectionReport(
        run_id=uuid.uuid4().hex,
        input_text=text,
        run_mode=RunMode.MOCK,
        claims=[claim],
        evidence=[evidence],
        verification_results=[result],
    )


def test_match_claim_aligns_by_quote_containment() -> None:
    text = "该研究发表于 2015 年。"
    report = make_report("2015 年", VerificationStatus.VERIFIED, text)
    case = GoldenCase(
        case_id="T01",
        category="正确事实",
        text=text,
        gold_claim_quote="2015 年",
        gold_status=VerificationStatus.VERIFIED,
        source_of_truth="测试",
        labeler="测试",
    )
    claim, result = match_claim(report, case)
    assert claim is not None and result is not None


def test_match_claim_returns_none_when_quote_missing() -> None:
    text = "该研究发表于 2015 年。"
    report = make_report("2015 年", VerificationStatus.VERIFIED, text)
    case = GoldenCase(
        case_id="T02",
        category="正确事实",
        text=text,
        gold_claim_quote="1999 年",
        gold_status=VerificationStatus.VERIFIED,
        source_of_truth="测试",
        labeler="测试",
    )
    assert match_claim(report, case) == (None, None)


# ==========================================================================
# 指标计算
# ==========================================================================
def outcome(
    case_id: str,
    gold: str,
    predicted: str | None,
    *,
    category: str = "正确事实",
    contains_ok: bool = True,
    excludes_ok: bool = True,
    new_tokens: tuple[str, ...] = (),
    traceable: int = 1,
    matched: bool = True,
) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        category=category,
        gold_status=gold,
        predicted_status=predicted,
        matched=matched,
        traceable_evidence_count=traceable,
        correction_contains_ok=contains_ok,
        correction_excludes_ok=excludes_ok,
        new_fact_tokens=new_tokens,
    )


def test_metrics_confusion_matrix_is_correct() -> None:
    outcomes = [
        outcome("A", "存在矛盾", "存在矛盾"),          # TP
        outcome("B", "证据不足", "存在矛盾"),          # FP
        outcome("C", "存在矛盾", "证据不足"),          # FN
        outcome("D", "已核实", "已核实"),              # TN
    ]
    metrics = compute_metrics(outcomes, needs_correction_ids=set())
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["detection_precision"] == 0.5
    assert metrics["detection_recall"] == 0.5
    assert metrics["detection_f1"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["status_accuracy"] == 0.5


def test_metrics_use_none_instead_of_fake_zero() -> None:
    """分母为 0 时必须是 None（界面显示「—」），不许用 0 或 1 蒙混。"""
    metrics = compute_metrics([outcome("A", "已核实", "已核实")], needs_correction_ids=set())
    assert metrics["detection_precision"] is None
    assert metrics["detection_recall"] is None
    assert metrics["detection_f1"] is None
    assert metrics["correction_success_rate"] is None
    assert metrics["new_error_rate"] == 0.0


def test_correction_metrics_count_only_cases_expecting_correction() -> None:
    outcomes = [
        outcome("A", "存在矛盾", "存在矛盾", contains_ok=True, excludes_ok=True),
        outcome("B", "存在矛盾", "存在矛盾", contains_ok=True, excludes_ok=False),
        outcome("C", "已核实", "已核实"),
    ]
    metrics = compute_metrics(outcomes, needs_correction_ids={"A", "B"})
    assert metrics["correction_expected_cases"] == 2
    assert metrics["correction_success_rate"] == 0.5
    assert metrics["residual_error_rate"] == 0.5


def test_new_error_rate_and_failure_list() -> None:
    outcomes = [
        outcome("A", "存在矛盾", "存在矛盾"),
        outcome("B", "证据不足", "证据不足", new_tokens=("25%",)),
        outcome("C", "已核实", None, matched=False),
    ]
    metrics = compute_metrics(outcomes, needs_correction_ids=set())
    assert metrics["new_error_rate"] == 0.5  # 2 条命中里 1 条引入了新事实
    assert metrics["unmatched_count"] == 1

    failures = summarize_failures(outcomes)
    kinds = {item["kind"] for item in failures}
    assert "未命中主张" in kinds
    assert "修正引入新事实" in kinds


def test_metrics_group_by_category() -> None:
    outcomes = [
        outcome("A", "存在矛盾", "存在矛盾", category="明确错误事实"),
        outcome("B", "证据不足", "存在矛盾", category="证据不足"),
    ]
    metrics = compute_metrics(outcomes, needs_correction_ids=set())
    assert metrics["by_category"]["明确错误事实"]["correct"] == 1
    assert metrics["by_category"]["证据不足"]["correct"] == 0


# ==========================================================================
# 运行器（离线配置真跑一遍，零成本）
# ==========================================================================
def test_offline_evaluation_runs_and_reports_caveats(tmp_path: Path) -> None:
    cases = load_golden_set(GOLDEN)
    base = Settings(
        _env_file=None,
        data_dir=tmp_path,
        save_runs=False,
        retrieve_enabled=False,
        llm_provider="mock",
    )
    report = run_evaluation(
        settings=offline_settings(base),
        cases=cases,
        baseline=BASELINE_B1,
        limit=3,
    )

    assert report.case_count == 3
    assert len(report.rows) == 3
    assert report.metrics["case_count"] == 3
    assert report.baseline == BASELINE_B1
    assert "离线规则引擎" in report.engine_summary

    joined = " ".join(report.caveats)
    assert "样本量" in joined, "必须说明样本量很小"
    assert "AI 起草" in joined, "必须说明标注是 AI 起草、待人工确认"
    assert "同源偏差" in joined, "必须说明被测系统与标注者同源的风险"

    # 每条用例都要能在报告里看到标注与预测的对照
    for row in report.rows:
        assert row.gold_status
        assert row.case_id


def test_offline_evaluation_can_filter_categories(tmp_path: Path) -> None:
    cases = load_golden_set(GOLDEN)
    base = Settings(_env_file=None, data_dir=tmp_path, save_runs=False, retrieve_enabled=False)
    report = run_evaluation(
        settings=offline_settings(base),
        cases=cases,
        only_categories=["来源冲突"],
    )
    assert report.case_count == len([case for case in cases if case.category == "来源冲突"])
    assert all(row.category == "来源冲突" for row in report.rows)
    # 文内矛盾不需要外部证据，离线也能判出来
    assert any(row.predicted_status == "来源冲突" for row in report.rows)
