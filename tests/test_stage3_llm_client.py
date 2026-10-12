"""阶段 3 测试：真实模型客户端（全部用 respx 模拟 HTTP，不产生任何费用）。

真实世界会出什么幺蛾子，这里就测什么：
    · 模型把 JSON 包在 ``` 里、前面还带一句解释；
    · 模型漏字段 / 字段名写错；
    · 401 密钥错、429 限流、超时、连接失败；
    · 供应商不支持 response_format=json_object；
    · 缓存命中、调用预算超限；
    · 模型挂了以后流水线要降级，并且**在报告里说清楚降级了**。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from verimind.core.config import Settings
from verimind.core.enums import LlmCallStatus, RoleName, RunMode
from verimind.core.errors import (
    LLMBudgetExceeded,
    LLMError,
    LLMTimeoutError,
    SchemaValidationError,
)
from verimind.core.pipeline import run_detection
from verimind.core.schemas import ClaimExtraction, FactCheckReport, LlmCallLog
from verimind.models.base import CallBudget
from verimind.models.cache import LLMCache
from verimind.models.openai_compat import OpenAICompatibleClient, extract_json_object

BASE_URL = "https://api.deepseek.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"

TEXT = "据一项发表于 2023 年的研究（DOI: 10.1038/nature12373）显示，准确率提高 40%。"

EXTRACTION_PAYLOAD = {
    "claims": [
        {
            "quote": "2023 年",
            "normalized_claim": "该文献发表于 2023 年。",
            "claim_type": "年份类",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["该文献实际发表年份是否为 2023 年？"],
            "confidence": "中",
        },
        {
            "quote": "10.1038/nature12373",
            "normalized_claim": "存在一篇 DOI 为 10.1038/nature12373 的文献。",
            "claim_type": "DOI 是否真实",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["该 DOI 是否真实存在？"],
            "confidence": "中",
        },
        {
            "quote": "40%",
            "normalized_claim": "原文给出了具体数值 40%。",
            "claim_type": "数值/统计类",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["该数值出处是什么？"],
            "confidence": "中",
        },
    ],
    "notes": [],
}

FACT_CHECK_PAYLOAD = {
    "assessments": [
        {
            "claim_id": "C001",
            "stance": "无法确定",
            "conclusion": "需要核对该文献的真实发表年份。",
            "reasoning": "原文只给出了年份，没有任何可核验凭据。",
            "suspected_problem_types": ["缺少必要证据"],
            "additional_questions": ["能否给出该文献的标题？"],
        },
        {
            "claim_id": "C002",
            "stance": "无法确定",
            "conclusion": "未发现明显疑点，但无法验证 DOI 是否存在。",
            "reasoning": "当前没有外部检索能力。",
            "suspected_problem_types": [],
            "additional_questions": [],
        },
        {
            "claim_id": "C003",
            "stance": "倾向反驳",
            "conclusion": "数值 40% 缺少出处，疑似夸大。",
            "reasoning": "句中没有任何来源说明。",
            "suspected_problem_types": ["数据或数字错误", "缺少必要证据"],
            "additional_questions": ["该数值来自哪张表或哪段结论？"],
        },
    ],
    "notes": [],
}


def completion_body(content: str, *, prompt_tokens: int = 120, completion_tokens: int = 45) -> dict:
    """构造一个 OpenAI 兼容的响应体。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_client(tmp_path: Path, **overrides) -> OpenAICompatibleClient:
    data = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": BASE_URL,
        "api_key": "sk-" + "x" * 20,
        "timeout_seconds": 10.0,
        "max_retries": 0,
        "cache": overrides.pop("cache", None),
        "budget": overrides.pop("budget", None),
        "recorder": overrides.pop("recorder", None),
    }
    data.update(overrides)
    return OpenAICompatibleClient(**data)


# ==========================================================================
# JSON 容错解析
# ==========================================================================
def test_extract_json_handles_plain_fenced_and_noisy_output() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('好的，结果如下：\n{"a": 1}\n希望有帮助。') == {"a": 1}


def test_extract_json_rejects_content_without_object() -> None:
    with pytest.raises(ValueError):
        extract_json_object("对不起，我无法回答这个问题。")
    with pytest.raises(ValueError):
        extract_json_object("")


# ==========================================================================
# 正常路径
# ==========================================================================
@respx.mock
def test_client_returns_validated_model_and_usage(tmp_path: Path) -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=completion_body(json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)))
    )
    client = make_client(tmp_path)
    extraction, usage = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    assert isinstance(extraction, ClaimExtraction)
    assert len(extraction.claims) == 3
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 45
    assert usage.simulated is False
    assert usage.cache_hit is False
    assert client.is_simulated is False


@respx.mock
def test_client_accepts_fenced_json_from_model(tmp_path: Path) -> None:
    fenced = "```json\n" + json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False) + "\n```"
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(fenced)))
    client = make_client(tmp_path)
    extraction, _ = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    assert len(extraction.claims) == 3


# ==========================================================================
# 格式修复重试
# ==========================================================================
@respx.mock
def test_client_repairs_schema_violation_once(tmp_path: Path) -> None:
    bad = json.dumps({"claims": [{"quote": "2023 年"}]}, ensure_ascii=False)  # 缺必填字段
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT)
    route.side_effect = [
        httpx.Response(200, json=completion_body(bad)),
        httpx.Response(200, json=completion_body(good)),
    ]
    client = make_client(tmp_path)
    extraction, usage = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    assert len(extraction.claims) == 3
    assert route.call_count == 2, "格式不合格时必须带着错误信息让模型修一次"
    assert usage.status is LlmCallStatus.SCHEMA_RETRY
    assert usage.prompt_tokens == 240, "两次调用的 token 都要计入成本，不能漏算"


@respx.mock
def test_client_raises_after_two_invalid_responses(tmp_path: Path) -> None:
    bad = json.dumps({"claims": [{"quote": "只有引用"}]}, ensure_ascii=False)
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(bad)))
    client = make_client(tmp_path)
    with pytest.raises(SchemaValidationError) as excinfo:
        client.complete_structured(
            role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
        )
    assert "ClaimExtraction" in str(excinfo.value)


# ==========================================================================
# 错误处理
# ==========================================================================
@respx.mock
def test_client_fails_fast_on_bad_key(tmp_path: Path) -> None:
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(401, json={"error": {"message": "Authentication Fails"}})
    )
    client = make_client(tmp_path, max_retries=2)
    with pytest.raises(LLMError) as excinfo:
        client.complete_structured(
            role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
        )
    assert "密钥" in str(excinfo.value)
    assert route.call_count == 1, "密钥错误不该重试——重试也只会继续失败"


@respx.mock
def test_client_retries_on_rate_limit_then_succeeds(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT)
    route.side_effect = [
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
        httpx.Response(200, json=completion_body(good)),
    ]
    client = make_client(tmp_path, max_retries=1)
    extraction, _ = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    assert len(extraction.claims) == 3
    assert route.call_count == 2


@respx.mock
def test_client_translates_timeout(tmp_path: Path) -> None:
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectTimeout("timed out"))
    client = make_client(tmp_path, max_retries=0)
    with pytest.raises(LLMTimeoutError) as excinfo:
        client.complete_structured(
            role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
        )
    assert "超时" in str(excinfo.value)


@respx.mock
def test_client_falls_back_when_json_mode_unsupported(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT)
    route.side_effect = [
        httpx.Response(
            400,
            json={"error": {"message": "response_format is not supported for this model"}},
        ),
        httpx.Response(200, json=completion_body(good)),
    ]
    client = make_client(tmp_path, max_retries=0)
    extraction, _ = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    assert len(extraction.claims) == 3
    assert route.call_count == 2
    assert client._json_mode_supported is False


# ==========================================================================
# 缓存与预算
# ==========================================================================
@respx.mock
def test_cache_hit_avoids_second_paid_call(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(good)))
    cache = LLMCache(tmp_path / "cache", enabled=True)
    client = make_client(tmp_path, cache=cache)

    _, first = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )
    _, second = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
    )

    assert route.call_count == 1, "第二次应当命中缓存，不再产生费用"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.status is LlmCallStatus.CACHE_HIT
    assert second.prompt_tokens == 0, "缓存命中没有消耗 token，不能虚报用量"
    assert cache.stats["hits"] == 1


@respx.mock
def test_cache_is_disabled_when_configured_off(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(good)))
    client = make_client(tmp_path, cache=LLMCache(tmp_path / "cache", enabled=False))
    client.complete_structured(role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction)
    client.complete_structured(role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction)
    assert route.call_count == 2


@respx.mock
def test_budget_exceeded_stops_calling(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(good)))
    budget = CallBudget(limit=1)
    client = make_client(tmp_path, budget=budget)
    client.complete_structured(role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction)
    with pytest.raises(LLMBudgetExceeded):
        client.complete_structured(
            role=RoleName.CLAIM_EXTRACTOR, prompt="拆解", schema=ClaimExtraction
        )
    assert route.call_count == 1
    assert budget.used == 1


# ==========================================================================
# 调用日志
# ==========================================================================
@respx.mock
def test_recorder_logs_call_without_leaking_prompt_or_key(tmp_path: Path) -> None:
    good = json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False)
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=completion_body(good)))
    recorded: list[LlmCallLog] = []
    client = make_client(tmp_path, recorder=recorded.append, run_id="run-123")

    client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR, prompt=f"绝密原文：{TEXT}", schema=ClaimExtraction
    )

    assert len(recorded) == 1
    log = recorded[0]
    assert log.run_id == "run-123"
    assert log.role is RoleName.CLAIM_EXTRACTOR
    assert log.prompt_tokens == 120
    assert len(log.prompt_hash) == 64
    dumped = log.model_dump_json()
    assert "绝密原文" not in dumped, "调用日志里不能出现用户原文"
    assert "sk-" not in dumped, "调用日志里不能出现密钥"


# ==========================================================================
# 流水线（真实客户端 + 模拟 HTTP）
# ==========================================================================
def pipeline_settings(tmp_path: Path, **overrides) -> Settings:
    data = {
        "_env_file": None,
        "data_dir": tmp_path,
        "save_runs": False,
        "llm_provider": "deepseek",
        "llm_base_url": BASE_URL,
        "llm_model": "deepseek-chat",
        "llm_api_key": "sk-" + "x" * 20,
        "llm_cache_enabled": False,
    }
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


@respx.mock
def test_pipeline_runs_with_real_model_and_records_metrics(tmp_path: Path) -> None:
    route = respx.post(ENDPOINT)
    route.side_effect = [
        httpx.Response(
            200,
            json=completion_body(json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False), prompt_tokens=200, completion_tokens=80),
        ),
        httpx.Response(
            200,
            json=completion_body(json.dumps(FACT_CHECK_PAYLOAD, ensure_ascii=False), prompt_tokens=300, completion_tokens=120),
        ),
    ]
    settings = pipeline_settings(
        tmp_path, llm_price_prompt_per_mtok="2", llm_price_completion_per_mtok="8"
    )
    report = run_detection(TEXT, settings=settings)

    assert report.run_mode is RunMode.SINGLE_MODEL_MULTI_ROLE
    assert report.claims, "模型应当拆出主张"
    assert route.call_count == 2, "一次拆解 + 一次事实审查"
    assert report.stats.total_claims >= 3

    # 指标来自真实用法
    assert report.metrics.prompt_tokens == 500
    assert report.metrics.completion_tokens == 200
    assert report.metrics.llm_calls == 2
    assert report.metrics.estimated_cost == pytest.approx(500 / 1e6 * 2 + 200 / 1e6 * 8)

    # 引擎配置如实记录
    assert report.llm_config_summary["claim_extractor"]["provider"] == "deepseek"
    assert report.llm_config_summary["claim_extractor"]["simulated"] == "否"
    assert report.llm_config_summary["rule_baseline"]["provider"] == "mock-rule-engine"

    # 交叉比对与角色结论
    assert report.cross_check_notes, "必须记录两个引擎的差异"
    assert any("规则基线" in note for note in report.cross_check_notes)
    with_positions = [r for r in report.verification_results if r.role_positions]
    assert with_positions, "事实审查员的结论应当写入报告"

    # 模型怀疑但规则层未确认的，必须作为分歧写明，而不是直接判成错误
    assert any("但规则层未确认" in note for note in report.cross_check_notes)
    assert all(not r.is_suspicious or r.refuting_evidence_ids for r in report.verification_results)


@respx.mock
def test_pipeline_degrades_loudly_when_model_fails(tmp_path: Path) -> None:
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(503, json={"error": {"message": "server overloaded"}})
    )
    settings = pipeline_settings(tmp_path, llm_max_retries=0)
    report = run_detection(TEXT, settings=settings)

    assert report.run_mode is RunMode.MOCK, "模型失败后必须如实把运行模式改回离线模拟"
    assert report.claims, "降级后仍应由规则引擎给出结果"
    assert report.metrics.llm_calls == 0
    assert any("已降级为离线规则引擎" in item for item in report.limitations)
    assert any("503" in item or "调用" in item for item in report.limitations)


@respx.mock
def test_pipeline_cost_is_none_without_configured_prices(tmp_path: Path) -> None:
    route = respx.post(ENDPOINT)
    route.side_effect = [
        httpx.Response(200, json=completion_body(json.dumps(EXTRACTION_PAYLOAD, ensure_ascii=False))),
        httpx.Response(200, json=completion_body(json.dumps(FACT_CHECK_PAYLOAD, ensure_ascii=False))),
    ]
    report = run_detection(TEXT, settings=pipeline_settings(tmp_path))
    assert report.metrics.estimated_cost is None, "没有单价就不许编一个成本数字"
    assert "VERIMIND_LLM_PRICE_PROMPT_PER_MTOK" in report.metrics.cost_note
    assert report.metrics.prompt_tokens > 0, "但 token 用量必须如实记录"


@pytest.mark.live
def test_live_model_call_is_available() -> None:
    """真实调用（默认跳过）：需要 pytest -m live 才会执行，会花钱。

    运行方式：
        .\\.venv\\Scripts\\python.exe -m pytest -m live -q
    """
    settings = Settings()
    if settings.missing_credentials or settings.llm_provider == "mock":
        pytest.skip("没有配置真实模型密钥，跳过真实调用测试")
    client = OpenAICompatibleClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=0,
    )
    extraction, usage = client.complete_structured(
        role=RoleName.CLAIM_EXTRACTOR,
        prompt='请只输出这个 JSON：{"claims":[],"notes":["连接测试"]}',
        schema=ClaimExtraction,
    )
    assert isinstance(extraction, ClaimExtraction)
    assert usage.prompt_tokens and usage.prompt_tokens > 0
