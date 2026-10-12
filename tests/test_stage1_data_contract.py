"""阶段 1 冒烟测试：验证"数据契约"和"配置"确实按设计工作。

为什么阶段 1 就要写测试？
    因为本项目最关键的设计不在界面上，而在数据结构里：
        · 没有证据不许判"存在矛盾"
        · 仅元数据的证据不许判断"支持/反驳"
        · 报告里不许引用不存在的证据
    这些规则如果只是写在文档里，两个月后一定会被绕过。
    现在用测试把它们钉死，任何人（包括我）以后写错代码都会立刻失败。

运行：
    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from verimind.core.config import Settings
from verimind.core.enums import (
    EXPECTED_PROBLEM_TYPE_COUNT,
    EXPECTED_REVIEW_CHECK_COUNT,
    EXPECTED_STATUS_COUNT,
    SUSPICIOUS_STATUSES,
    ChangeType,
    Checkability,
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    JudgedAspect,
    MatchBasis,
    ProblemType,
    ReviewCheckType,
    ReviewOverall,
    RunMode,
    SearchStatus,
    VerificationStatus,
)
from verimind.core.schemas import (
    Claim,
    Correction,
    DetectionReport,
    Evidence,
    ReviewCheck,
    ReviewResult,
    SearchRecord,
    SearchTrace,
    Span,
    VerificationResult,
    find_span,
)

INPUT_TEXT = "某研究（DOI: 10.xxxx/example）发表于 2023 年，证明该方法可将准确率提高 40%。"


# ==========================================================================
# 一、枚举词汇表
# ==========================================================================
def test_enum_vocabulary_matches_requirements() -> None:
    """用户明确要求：6 种核验状态、8 类问题类型、5 项复核检查。少一个都不行。"""
    assert len(list(VerificationStatus)) == EXPECTED_STATUS_COUNT == 6
    assert len(list(ProblemType)) == EXPECTED_PROBLEM_TYPE_COUNT == 8
    assert len(list(ReviewCheckType)) == EXPECTED_REVIEW_CHECK_COUNT == 5


def test_only_real_errors_count_as_suspicious() -> None:
    """「证据不足」「暂无法核验」不是错误，绝不能被算作检测出的问题。"""
    values = {s.value for s in SUSPICIOUS_STATUSES}
    assert "存在矛盾" in values
    assert "证据不足" not in values
    assert "暂无法核验" not in values


# ==========================================================================
# 二、原文定位
# ==========================================================================
def test_span_from_text_keeps_offsets_consistent() -> None:
    span = Span.from_text(INPUT_TEXT, 0, 3)
    assert span.text == "某研究"
    assert span.is_inside(INPUT_TEXT)


def test_span_rejects_inconsistent_text() -> None:
    """片段长度和区间对不上时必须报错——否则界面上会高亮错位置。"""
    with pytest.raises(ValidationError):
        Span(start=0, end=5, text="abc")


def test_span_rejects_reversed_range() -> None:
    with pytest.raises(ValidationError):
        Span(start=5, end=2, text="abc")


def test_find_span_locates_snippet_or_gives_up() -> None:
    assert find_span(INPUT_TEXT, "发表于 2023 年") is not None
    # 找不到就返回 None，绝不硬凑一个错误区间
    assert find_span(INPUT_TEXT, "这段文字根本不在原文里") is None


# ==========================================================================
# 三、主张
# ==========================================================================
def make_claim(**overrides: object) -> Claim:
    data: dict[str, object] = {
        "claim_id": "C001",
        "source_span": Span.from_text(INPUT_TEXT, 0, 5),
        "normalized_claim": "存在一篇 DOI 为 10.xxxx/example 的研究。",
        "claim_type": ClaimType.WORK_EXISTENCE,
        "verification_questions": ["该 DOI 是否对应一篇真实存在的文献？"],
    }
    data.update(overrides)
    return Claim(**data)  # type: ignore[arg-type]


def test_claim_requires_verification_questions() -> None:
    """可核验的主张必须给出核验问题，否则系统不知道要去查什么。"""
    with pytest.raises(ValidationError):
        make_claim(verification_questions=[])


def test_not_checkable_claim_may_skip_questions() -> None:
    claim = make_claim(
        claim_type=ClaimType.OPINION,
        checkability=Checkability.NOT_CHECKABLE,
        verification_questions=[],
    )
    assert claim.checkability is Checkability.NOT_CHECKABLE


def test_claim_id_format_is_enforced() -> None:
    with pytest.raises(ValidationError):
        make_claim(claim_id="claim-1")


# ==========================================================================
# 四、证据（本项目最重要的诚实性约束）
# ==========================================================================
def make_evidence(**overrides: object) -> Evidence:
    data: dict[str, object] = {
        "evidence_id": "E001",
        "claim_ids": ["C001"],
        "source": EvidenceSource.CROSSREF,
        "level": EvidenceLevel.METADATA_ONLY,
        "url": "https://api.crossref.org/works/10.xxxx/example",
        "query": "10.xxxx/example",
    }
    data.update(overrides)
    return Evidence(**data)  # type: ignore[arg-type]


def test_evidence_url_must_be_clickable() -> None:
    with pytest.raises(ValidationError):
        make_evidence(url="not-a-real-url")


def test_metadata_only_evidence_cannot_judge_content() -> None:
    """只有元数据，就不能说"这篇论文支持/反驳了该结论"——那是在假装读过全文。

    阶段 4 起这条约束改得更精确了：元数据**可以**用来判断
    「文献是否存在」「书目信息是否一致」，但**不能**判断「内容结论是否被支持」。
    """
    with pytest.raises(ValidationError):
        make_evidence(
            relation=EvidenceRelation.SUPPORTS,
            relation_rationale="标题看起来一致",
            judged_aspect=JudgedAspect.CONTENT,
        )


def test_metadata_only_evidence_may_judge_existence() -> None:
    """反过来：注册机构的元数据足以判断"文献是否存在"，这是允许且有价值的。"""
    evidence = make_evidence(
        relation=EvidenceRelation.REFUTES,
        relation_rationale="Crossref 明确返回 404：该 DOI 未注册。",
        judged_aspect=JudgedAspect.EXISTENCE,
    )
    assert evidence.judged_aspect is JudgedAspect.EXISTENCE


def test_judgement_requires_stating_the_aspect() -> None:
    """给出了支持/反驳关系，就必须说明判断的是哪个层面。"""
    with pytest.raises(ValidationError):
        make_evidence(
            level=EvidenceLevel.ABSTRACT,
            relation=EvidenceRelation.SUPPORTS,
            relation_rationale="摘要里提到了这一点",
        )


def test_content_judgement_requires_quoting_the_evidence() -> None:
    """判断内容层面必须能引用原句（否则无法复核）。"""
    with pytest.raises(ValidationError):
        make_evidence(
            level=EvidenceLevel.ABSTRACT,
            relation=EvidenceRelation.REFUTES,
            relation_rationale="摘要与主张相反",
            judged_aspect=JudgedAspect.CONTENT,
            snippet=None,
        )


def test_supporting_evidence_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        make_evidence(
            level=EvidenceLevel.ABSTRACT,
            relation=EvidenceRelation.SUPPORTS,
            relation_rationale="",
            judged_aspect=JudgedAspect.CONTENT,
        )


def test_abstract_evidence_may_support_with_rationale_and_snippet() -> None:
    ev = make_evidence(
        level=EvidenceLevel.ABSTRACT,
        relation=EvidenceRelation.SUPPORTS,
        relation_rationale="摘要明确给出该结论。",
        judged_aspect=JudgedAspect.CONTENT,
        snippet="Our results show ...",
    )
    assert ev.relation is EvidenceRelation.SUPPORTS


def test_evidence_url_must_be_clickable() -> None:
    with pytest.raises(ValidationError):
        make_evidence(url="not-a-real-url")


def test_internal_evidence_needs_a_span_instead_of_url() -> None:
    """原文内部证据不许有 URL，但必须有原文定位与片段。"""
    internal = Evidence(
        evidence_id="E010",
        source=EvidenceSource.INTERNAL,
        level=EvidenceLevel.USER_TEXT,
        url=None,
        source_span=Span.from_text(INPUT_TEXT, 0, 3),
        snippet=INPUT_TEXT[0:3],
        relation=EvidenceRelation.REFUTES,
        relation_rationale="原文前后矛盾",
        match_basis=MatchBasis.INTERNAL,
        judged_aspect=JudgedAspect.INTERNAL_CONSISTENCY,
    )
    assert internal.is_internal is True

    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="E011",
            source=EvidenceSource.INTERNAL,
            level=EvidenceLevel.USER_TEXT,
            source_span=None,
            snippet="x",
            relation=EvidenceRelation.REFUTES,
            relation_rationale="缺定位",
            judged_aspect=JudgedAspect.INTERNAL_CONSISTENCY,
        )


def test_external_evidence_requires_a_url() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="E012",
            source=EvidenceSource.CROSSREF,
            level=EvidenceLevel.METADATA_ONLY,
            url=None,
        )


# ==========================================================================
# 五、核验结果：没有证据不许判错
# ==========================================================================
def make_trace() -> SearchTrace:
    return SearchTrace(
        records=[
            SearchRecord(
                source=EvidenceSource.CROSSREF,
                query="10.xxxx/example",
                status=SearchStatus.NO_HIT,
                hit_count=0,
            )
        ]
    )


def test_contradicted_requires_problem_type_and_evidence() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C001",
            status=VerificationStatus.CONTRADICTED,
            rationale="该 DOI 未在 Crossref 注册。",
            problem_types=[ProblemType.CITATION_MISSING],
            refuting_evidence_ids=[],  # 没有证据
        )


def test_contradicted_requires_at_least_one_problem_type() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C001",
            status=VerificationStatus.CONTRADICTED,
            rationale="有问题，但说不清是哪类问题。",
            problem_types=[],
            refuting_evidence_ids=["E001"],
        )


def test_verified_requires_supporting_evidence() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C001",
            status=VerificationStatus.VERIFIED,
            rationale="看起来没问题。",
            supporting_evidence_ids=[],
        )


def test_insufficient_requires_missing_evidence_and_search_trace() -> None:
    """说"证据不足"时必须交代：查过什么、还缺什么。"""
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C001",
            status=VerificationStatus.INSUFFICIENT,
            rationale="查不到。",
            missing_evidence=[],
            search_trace=SearchTrace(),
        )


def test_insufficient_with_full_explanation_is_valid() -> None:
    result = VerificationResult(
        claim_id="C001",
        status=VerificationStatus.INSUFFICIENT,
        rationale="未检索到该 DOI 对应的文献，无法确认其真实性。",
        missing_evidence=["需要该 DOI 在 Crossref 的注册记录"],
        search_trace=make_trace(),
    )
    assert result.is_suspicious is False


def test_source_conflict_requires_two_evidence_items() -> None:
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C001",
            status=VerificationStatus.SOURCE_CONFLICT,
            rationale="两个来源不一致。",
            problem_types=[ProblemType.SOURCE_CONFLICT],
            related_evidence_ids=["E001"],
        )


# ==========================================================================
# 六、修正与复核
# ==========================================================================
def make_correction(**overrides: object) -> Correction:
    data: dict[str, object] = {
        "correction_id": "K001",
        "claim_id": "C001",
        "original_span": Span.from_text(INPUT_TEXT, 0, 5),
        "proposed_text": "（原文此处引用了一篇无法核实的文献）",
        "change_type": ChangeType.ADD_UNCERTAINTY,
        "rationale": "该 DOI 无法在 Crossref 中查到，应标注不确定性。",
        "evidence_ids": [],
    }
    data.update(overrides)
    return Correction(**data)  # type: ignore[arg-type]


def test_fact_changing_correction_requires_evidence() -> None:
    """会写入新事实内容的修改，必须有证据；拿不到证据就只能弱化或删除。"""
    with pytest.raises(ValidationError):
        make_correction(change_type=ChangeType.FIX_NUMBER, proposed_text="准确率提高了 12%", evidence_ids=[])


def test_uncertainty_correction_needs_no_evidence() -> None:
    assert make_correction().change_type is ChangeType.ADD_UNCERTAINTY


def make_checks(*, passed: bool) -> list[ReviewCheck]:
    return [
        ReviewCheck(check=check_type, passed=passed, detail=f"测试用说明：{check_type.value}")
        for check_type in ReviewCheckType
    ]


def test_review_requires_all_five_checks() -> None:
    with pytest.raises(ValidationError):
        ReviewResult(
            review_id="V001",
            target_correction_id="K001",
            checks=make_checks(passed=True)[:4],
            overall=ReviewOverall.PASSED_WITH_RISKS,
            unresolved=["有一项检查缺失"],
        )


def test_review_passed_requires_every_check_passed() -> None:
    checks = make_checks(passed=True)
    checks[2] = ReviewCheck(
        check=ReviewCheckType.NO_NEW_FACT_ERROR, passed=False, detail="修正引入了新的数字"
    )
    with pytest.raises(ValidationError):
        ReviewResult(
            review_id="V001",
            target_correction_id="K001",
            checks=checks,
            overall=ReviewOverall.PASSED,
        )


def test_review_not_passed_must_list_unresolved() -> None:
    checks = make_checks(passed=True)
    checks[0] = ReviewCheck(
        check=ReviewCheckType.ORIGINAL_ISSUE_RESOLVED, passed=False, detail="原始问题没解决"
    )
    with pytest.raises(ValidationError):
        ReviewResult(
            review_id="V001",
            target_correction_id="K001",
            checks=checks,
            overall=ReviewOverall.FAILED,
            unresolved=[],
        )


# ==========================================================================
# 七、配置：默认离线、密钥不泄露、模式如实
# ==========================================================================
def make_settings(**overrides: object) -> Settings:
    data: dict[str, object] = {"_env_file": None}
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


def test_default_settings_run_offline() -> None:
    settings = make_settings(llm_provider="mock", llm_api_key="")
    assert settings.run_mode is RunMode.MOCK
    assert settings.masked_api_key == "未配置"


def test_missing_key_degrades_to_offline_instead_of_crashing() -> None:
    settings = make_settings(llm_provider="deepseek", llm_api_key="")
    assert settings.missing_credentials is True
    assert settings.run_mode is RunMode.MOCK
    assert any("API_KEY" in w for w in settings.runtime_warnings)


def test_masked_key_never_leaks_full_key() -> None:
    secret = "sk-1234567890abcdefghij"
    settings = make_settings(llm_provider="deepseek", llm_api_key=secret)
    assert secret not in settings.masked_api_key
    assert settings.masked_api_key.startswith("sk-")
    assert settings.masked_api_key.endswith("ghij")


def test_single_provider_is_reported_as_single_model_multi_role() -> None:
    settings = make_settings(llm_provider="deepseek", llm_api_key="sk-" + "x" * 20)
    assert settings.run_mode is RunMode.SINGLE_MODEL_MULTI_ROLE


def test_distinct_reviewer_upgrades_to_multi_provider() -> None:
    settings = make_settings(
        llm_provider="deepseek",
        llm_api_key="sk-" + "x" * 20,
        llm_reviewer_provider="zhipu",
        llm_reviewer_model="glm-4-plus",
        llm_reviewer_api_key="zp-" + "y" * 20,
    )
    assert settings.run_mode is RunMode.MULTI_PROVIDER


def test_llm_config_summary_contains_no_real_key() -> None:
    settings = make_settings(llm_provider="deepseek", llm_api_key="sk-abcdefghijklmnop")
    summary = settings.llm_config_summary()
    dumped = str(summary)
    assert "sk-abcdefghijklmnop" not in dumped
    assert set(summary) == {
        "claim_extractor",
        "fact_checker",
        "evidence_checker",
        "corrector",
        "independent_reviewer",
    }


# ==========================================================================
# 八、完整报告：自洽性
# ==========================================================================
def make_report(**overrides: object) -> DetectionReport:
    data: dict[str, object] = {
        "run_id": uuid.uuid4().hex,
        "input_text": INPUT_TEXT,
        "run_mode": RunMode.MOCK,
        "app_version": "0.1.0",
        "pipeline_version": "0.1.0",
        "claims": [make_claim()],
        "evidence": [make_evidence()],
        "verification_results": [
            VerificationResult(
                claim_id="C001",
                status=VerificationStatus.INSUFFICIENT,
                rationale="未检索到该 DOI 对应的文献。",
                missing_evidence=["需要该 DOI 在 Crossref 的注册记录"],
                search_trace=make_trace(),
            )
        ],
    }
    data.update(overrides)
    return DetectionReport(**data)  # type: ignore[arg-type]


def test_report_recompute_fills_stats_from_real_data() -> None:
    report = make_report().recompute()
    assert report.stats.total_claims == 1
    assert report.stats.status_counts[VerificationStatus.INSUFFICIENT.value] == 1
    assert report.stats.suspicious_claims == 0
    assert report.unresolved_claim_ids == ["C001"]


def test_report_rejects_dangling_evidence_reference() -> None:
    """报告里指向一条不存在的证据——必须在构造时就报错。"""
    with pytest.raises(ValidationError) as excinfo:
        make_report(
            verification_results=[
                VerificationResult(
                    claim_id="C001",
                    status=VerificationStatus.CONTRADICTED,
                    rationale="该 DOI 不存在。",
                    problem_types=[ProblemType.CITATION_MISSING],
                    refuting_evidence_ids=["E999"],
                )
            ]
        )
    assert "E999" in str(excinfo.value)


def test_report_rejects_verification_for_unknown_claim() -> None:
    with pytest.raises(ValidationError):
        make_report(
            verification_results=[
                VerificationResult(
                    claim_id="C777",
                    status=VerificationStatus.UNVERIFIABLE,
                    rationale="查不到。",
                    missing_evidence=["缺少可检索的标识信息"],
                    search_trace=make_trace(),
                )
            ]
        )


def test_report_json_roundtrip_preserves_everything() -> None:
    report = make_report().recompute()
    restored = DetectionReport.from_json(report.to_json())
    assert restored.claims[0].source_span == report.claims[0].source_span
    assert restored.run_mode is RunMode.MOCK
    assert restored.stats.total_claims == 1
    assert restored.verification_results[0].status is VerificationStatus.INSUFFICIENT


def test_report_new_computes_input_hash() -> None:
    report = DetectionReport.new(
        run_id=uuid.uuid4().hex,
        input_text=INPUT_TEXT,
        run_mode=RunMode.MOCK,
    )
    assert len(report.input_sha256) == 64  # sha256 十六进制长度
