"""阶段 2 测试：完整流水线、本地存储、模型适配层。

这些测试回答三个问题：
    1. 整条链路真的能跑通吗（4 条示例文本端到端）？
    2. 报告自洽吗（统计与数据一致、JSON 能原样还原）？
    3. 适配层守规矩吗（离线引擎不许冒充大模型、未实现的角色必须报错）？
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verimind.core.config import Settings
from verimind.core.enums import LlmCallStatus, RoleName, RunMode, VerificationStatus
from verimind.core.errors import (
    InputError,
    InputTooLongError,
    LLMError,
    LLMUnsupportedRoleError,
)
from verimind.core.pipeline import ProgressEvent, run_detection
from verimind.core.schemas import ClaimExtraction, DetectionReport
from verimind.models.base import build_client
from verimind.models.mock_provider import RuleBasedOfflineClient
from verimind.storage.repository import RunRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = PROJECT_ROOT / "examples" / "sample_inputs.jsonl"


def load_samples() -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


SAMPLES = load_samples()


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    data: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path,
        "save_runs": False,
        "llm_provider": "mock",
        "llm_api_key": "",
    }
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


# ==========================================================================
# 端到端
# ==========================================================================
def test_sample_file_exists_and_is_wellformed() -> None:
    assert len(SAMPLES) >= 4
    for sample in SAMPLES:
        assert sample["id"] and sample["title"] and sample["text"]
        assert sample["note"], "每个示例都要说明它想演示什么"


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample["id"] for sample in SAMPLES])
def test_pipeline_runs_end_to_end_on_every_sample(sample: dict[str, str], tmp_path: Path) -> None:
    report = run_detection(sample["text"], settings=make_settings(tmp_path))

    assert isinstance(report, DetectionReport)
    assert report.run_mode is RunMode.MOCK, "离线规则引擎必须如实标注为离线模拟"
    assert report.claims, f"示例 {sample['id']} 应当能拆出主张"
    assert len(report.verification_results) == len(report.claims)

    # 每条主张都必须有结论，且结论里的证据引用真实存在
    known_evidence = {item.evidence_id for item in report.evidence}
    for claim in report.claims:
        result = report.result_by_claim(claim.claim_id)
        assert result is not None
        assert set(
            result.supporting_evidence_ids + result.refuting_evidence_ids
        ) <= known_evidence

    # 阶段 2 不可能给出需要外部证据才能有的结论
    assert all(
        result.status is not VerificationStatus.VERIFIED for result in report.verification_results
    )
    assert report.limitations, "必须如实声明能力边界"


def test_report_metrics_are_honest_about_offline_engine(tmp_path: Path) -> None:
    report = run_detection(SAMPLES[0]["text"], settings=make_settings(tmp_path))
    assert report.metrics.llm_calls == 0, "规则引擎不是大模型调用，不能虚报调用次数"
    assert report.metrics.prompt_tokens == 0
    assert report.metrics.estimated_cost is None, "没有单价就不该编一个成本数字"
    assert "离线规则引擎" in report.metrics.cost_note
    assert report.metrics.latency_ms is not None


def test_report_stats_match_the_actual_data(tmp_path: Path) -> None:
    report = run_detection(SAMPLES[-1]["text"], settings=make_settings(tmp_path))
    stats = report.stats
    assert stats.total_claims == len(report.claims)
    assert sum(stats.status_counts.values()) == len(report.verification_results)
    assert stats.suspicious_claims == sum(
        1 for result in report.verification_results if result.is_suspicious
    )
    assert stats.flagged_claims >= stats.suspicious_claims
    assert stats.problem_counts == {
        name: sum(1 for result in report.verification_results for problem in result.problem_types
                  if problem.value == name)
        for name, count in stats.problem_counts.items()
    }


def test_engine_config_summary_never_contains_a_key(tmp_path: Path) -> None:
    report = run_detection(SAMPLES[0]["text"], settings=make_settings(tmp_path))
    summary_text = json.dumps(report.llm_config_summary, ensure_ascii=False)
    assert report.llm_config_summary["claim_extractor"]["provider"] == "mock-rule-engine"
    assert report.llm_config_summary["independent_reviewer"]["provider"] == "未启用"
    assert "api_key" not in summary_text.lower()


def test_progress_callback_reports_every_stage_and_finishes(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []
    run_detection(
        SAMPLES[0]["text"], settings=make_settings(tmp_path), progress=events.append
    )
    assert events, "必须汇报进度，否则界面只能干等"
    ratios = [event.ratio for event in events]
    assert ratios == sorted(ratios), "进度不能倒退"
    assert ratios[0] > 0 and ratios[-1] == pytest.approx(1.0)
    assert any("拆解" in event.stage for event in events)
    assert any("规则裁决" in event.stage for event in events)


def test_truncation_is_reported_in_limitations(tmp_path: Path) -> None:
    report = run_detection(
        SAMPLES[0]["text"], settings=make_settings(tmp_path, max_claims=1)
    )
    assert len(report.claims) == 1
    assert report.stats.total_claims == 1
    assert any("只保留了前 1 条" in item for item in report.limitations)


def test_report_json_roundtrip(tmp_path: Path) -> None:
    report = run_detection(SAMPLES[-1]["text"], settings=make_settings(tmp_path))
    restored = DetectionReport.from_json(report.to_json())
    assert restored.stats.model_dump() == report.stats.model_dump()
    assert [claim.claim_id for claim in restored.claims] == [c.claim_id for c in report.claims]
    assert [item.evidence_id for item in restored.evidence] == [
        item.evidence_id for item in report.evidence
    ]


# ==========================================================================
# 输入校验
# ==========================================================================
def test_empty_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        run_detection("   ", settings=make_settings(tmp_path))


def test_too_long_input_is_rejected_with_actionable_message(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_input_chars=100)
    with pytest.raises(InputTooLongError) as excinfo:
        run_detection("很长" * 100, settings=settings)
    assert "VERIMIND_MAX_INPUT_CHARS" in str(excinfo.value)


# ==========================================================================
# 本地存储
# ==========================================================================
def test_repository_save_get_list_and_delete(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs.db")
    report = run_detection(
        SAMPLES[-1]["text"], settings=make_settings(tmp_path, save_runs=True), repository=repository
    )

    assert repository.count() == 1
    loaded = repository.get(report.run_id)
    assert loaded is not None
    assert loaded.stats.total_claims == report.stats.total_claims
    assert loaded.claims[0].source_span.text == report.claims[0].source_span.text

    summaries = repository.list_recent()
    assert len(summaries) == 1
    assert summaries[0].run_id == report.run_id
    assert summaries[0].total_claims == report.stats.total_claims
    assert summaries[0].flagged_claims == report.stats.flagged_claims
    assert summaries[0].run_mode == RunMode.MOCK.value

    assert repository.delete(report.run_id) is True
    assert repository.get(report.run_id) is None
    assert repository.count() == 0


def test_repository_returns_none_for_unknown_id(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs.db")
    assert repository.get("不存在的-run-id") is None
    assert repository.get("") is None
    assert repository.delete("不存在的-run-id") is False


def test_repository_schema_migration_is_idempotent(tmp_path: Path) -> None:
    """老数据库文件缺列时要能自动补列，重复打开也不能出错。"""
    db_path = tmp_path / "runs.db"
    RunRepository(db_path)
    RunRepository(db_path)
    RunRepository(db_path)
    repository = RunRepository(db_path)
    assert repository.count() == 0


def test_pipeline_saves_report_when_enabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, save_runs=True)
    report = run_detection(SAMPLES[0]["text"], settings=settings)
    repository = RunRepository(settings.sqlite_path)
    assert repository.count() == 1
    assert repository.get(report.run_id) is not None


# ==========================================================================
# 模型适配层
# ==========================================================================
def test_build_client_returns_offline_engine_for_mock(tmp_path: Path) -> None:
    client = build_client(make_settings(tmp_path))
    assert isinstance(client, RuleBasedOfflineClient)
    assert client.is_simulated is True
    assert "mock" in client.engine_id


def test_build_client_requires_key_for_real_provider(tmp_path: Path) -> None:
    """选了需要密钥的 provider 却没填密钥时，必须明确报错，绝不偷偷降级。"""
    settings = make_settings(tmp_path, llm_provider="deepseek", llm_api_key="")
    with pytest.raises(LLMError) as excinfo:
        build_client(settings)
    assert "VERIMIND_LLM_API_KEY" in str(excinfo.value)
    assert "mock" in str(excinfo.value)


def test_build_client_returns_real_client_when_key_present(tmp_path: Path) -> None:
    """阶段 3 起，填了密钥就应该拿到真实模型客户端（而不是报错或规则引擎）。"""
    settings = make_settings(
        tmp_path, llm_provider="deepseek", llm_api_key="sk-" + "x" * 20
    )
    client = build_client(settings)
    assert client.is_simulated is False
    assert client.engine_id == "deepseek/deepseek-chat"


def test_offline_client_rejects_roles_it_cannot_perform(tmp_path: Path) -> None:
    client = build_client(make_settings(tmp_path))
    for role in (RoleName.FACT_CHECKER, RoleName.EVIDENCE_CHECKER, RoleName.INDEPENDENT_REVIEWER):
        with pytest.raises(LLMUnsupportedRoleError):
            client.complete_structured(
                role=role, prompt="测试", schema=ClaimExtraction, context={"text": "文本"}
            )


def test_offline_client_reports_simulated_usage(tmp_path: Path) -> None:
    client = build_client(make_settings(tmp_path))
    result, usage = client.extract_claims(
        "该研究于 2023 年发表（DOI: 10.1038/nature12373）。", max_claims=8
    )
    # 高层入口返回的是已落地的 Claim（带精确位置），而不是草稿
    assert result.claims
    assert all(claim.source_span.is_inside("该研究于 2023 年发表（DOI: 10.1038/nature12373）。") for claim in result.claims)
    assert usage.simulated is True
    assert usage.status is LlmCallStatus.OK
    assert usage.prompt_tokens is None, "规则引擎不消耗 token，不能编一个数字"
    assert usage.total_tokens == 0
    assert usage.latency_ms >= 0


def test_offline_client_output_lands_with_exact_offsets(tmp_path: Path) -> None:
    text = "该研究于 2023 年发表（DOI: 10.1038/nature12373）。"
    client = build_client(make_settings(tmp_path))
    result, _ = client.extract_claims(text, max_claims=8)
    assert result.claims
    for claim in result.claims:
        assert claim.source_span.is_inside(text)
