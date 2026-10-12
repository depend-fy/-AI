"""阶段 4 测试：端到端（真实检索链路 + 模拟 LLM + 模拟 HTTP）。

这里验证的是**阶段 4 的验收标准**：
    ① 虚构 DOI → 「存在矛盾 / 引用不存在」；
    ② 真实 DOI + 年份写错 → 「存在矛盾 / 时间信息错误」；
    ③ 真实 DOI + 年份一致 → 「已核实」；
    ④ 摘要反驳结论（且引用了原句）→ 「存在矛盾 / 数据错误」；
    ⑤ 摘要里没有模型引用的那句话 → 丢弃该判断，停在「证据不足」；
    ⑥ 检索服务全挂 → 「暂无法核验」（不是"证据不足"，更不是"错误"）；
    ⑦ 只有元数据 → 只核验存在性与书目信息，绝不判断内容；
    ⑧ 报告里每条外部证据都能追溯到留档的原始响应。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from verimind.core.config import Settings
from verimind.core.enums import ProblemType, RunMode, VerificationStatus
from verimind.core.pipeline import run_detection
from verimind.core.schemas import DetectionReport

CHAT = "https://api.deepseek.com/v1/chat/completions"
CROSSREF = r"api\.crossref\.org/works/"
OPENALEX = r"api\.openalex\.org/works/"

DOI = "10.1038/nature12373"
TEXT = f"某研究（DOI: {DOI}）发表于 2008 年，显示准确率提高了 40%。"
ABSTRACT = "We report a certified power conversion efficiency of 12% in perovskite solar cells."
QUOTE = "a certified power conversion efficiency of 12%"
FAKE_QUOTE = "We achieved 99% efficiency in our experiments."

EXTRACTION_PAYLOAD = {
    "claims": [
        {
            "quote": "2008 年",
            "normalized_claim": "该文献发表于 2008 年。",
            "claim_type": "年份类",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["实际发表年份是否为 2008 年？"],
            "confidence": "中",
        },
        {
            "quote": DOI,
            "normalized_claim": f"存在一篇 DOI 为 {DOI} 的文献。",
            "claim_type": "DOI 是否真实",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["该 DOI 是否真实存在？"],
            "confidence": "中",
        },
        {
            "quote": "40%",
            "normalized_claim": "原文称准确率提高了 40%。",
            "claim_type": "数值/统计类",
            "priority": "高",
            "checkability": "可核验",
            "verification_questions": ["该数值是否有出处？"],
            "confidence": "中",
        },
    ],
    "notes": [],
}


def completion_body(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "deepseek-chat",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def _evidence_review_payload(prompt: str, refute_quote: str | None) -> dict:
    """从提示词里读出**真实的** evidence_id 与摘要，据此构造角色 B 的判断。

    这样测试不用写死证据编号（编号由程序分配），也不会因为编号顺序变化而失效。
    """
    raw = prompt.split("<<<PAIRS", 1)[1].split("PAIRS>>>", 1)[0]
    pairs = json.loads(raw)
    assessments: list[dict] = []
    for pair in pairs:
        if not (pair.get("摘要") or "").strip():
            continue
        if refute_quote is not None and pair.get("主张类型") == "数值/统计类":
            assessments.append(
                {
                    "evidence_id": pair["evidence_id"],
                    "claim_id": pair["claim_id"],
                    "relation": "反驳",
                    "judged_aspect": "内容结论是否被支持",
                    "rationale": "摘要给出的效率是 12%，与原文的 40% 不符。",
                    "quoted_snippet": refute_quote,
                }
            )
        else:
            assessments.append(
                {
                    "evidence_id": pair["evidence_id"],
                    "claim_id": pair["claim_id"],
                    "relation": "无法判断",
                    "judged_aspect": "内容结论是否被支持",
                    "rationale": "摘要没有涉及该主张的具体内容。",
                    "quoted_snippet": None,
                }
            )
    return {"assessments": assessments, "notes": []}


def chat_handler(*, refute_quote: str | None = QUOTE):
    """按提示词内容分派 LLM 响应（拆解 / 证据审查 / 事实审查）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = " ".join(str(message.get("content", "")) for message in body.get("messages", []))
        if "拆解为原子事实主张" in prompt:
            payload = EXTRACTION_PAYLOAD
        elif "证据审查员" in prompt:
            payload = _evidence_review_payload(prompt, refute_quote)
        else:
            payload = {"assessments": [], "notes": []}
        return httpx.Response(200, json=completion_body(json.dumps(payload, ensure_ascii=False)))

    return handler


def crossref_payload(year: int = 2013) -> dict:
    return {
        "status": "ok",
        "message": {
            "DOI": DOI,
            "title": ["Certified perovskite solar cells"],
            "author": [{"family": "Smith", "given": "Jane"}],
            "issued": {"date-parts": [[year, 9, 11]]},
            "container-title": ["Nature"],
            "publisher": "Springer",
            "URL": f"https://doi.org/{DOI}",
        },
    }


def openalex_payload(year: int = 2013, with_abstract: bool = True) -> dict:
    words = ABSTRACT.split()
    payload = {
        "id": "https://openalex.org/W1",
        "doi": f"https://doi.org/{DOI}",
        "display_name": "Certified perovskite solar cells",
        "publication_year": year,
        "authorships": [{"author": {"display_name": "Jane Smith"}}],
        "primary_location": {
            "landing_page_url": f"https://doi.org/{DOI}",
            "source": {"display_name": "Nature", "host_organization_name": "Springer"},
        },
    }
    if with_abstract:
        payload["abstract_inverted_index"] = {word: [index] for index, word in enumerate(words)}
    return payload


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        save_runs=False,
        llm_provider="deepseek",
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-chat",
        llm_api_key="sk-" + "x" * 20,
        llm_cache_enabled=False,
        retrieve_enabled=True,
        retrieval_min_interval_seconds=0.0,
        retrieval_max_retries=0,
        retrieval_mailto="test@example.com",
    )


def run(tmp_path: Path, **kwargs) -> DetectionReport:
    return run_detection(TEXT, settings=make_settings(tmp_path), **kwargs)


def result_for(report: DetectionReport, claim_type: str):
    claim = next(item for item in report.claims if item.claim_type.value == claim_type)
    return claim, report.result_by_claim(claim.claim_id)


# ==========================================================================
# ① ② ③ ④：一次运行同时覆盖四类结论
# ==========================================================================
@respx.mock
def test_full_retrieval_produces_verified_and_contradicted(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler(refute_quote=QUOTE))
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(200, json=crossref_payload(year=2013))
    )
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(200, json=openalex_payload(year=2013))
    )

    report = run(tmp_path)

    # ② 年份写错 → 存在矛盾 + 时间信息错误
    _, year_result = result_for(report, "年份类")
    assert year_result is not None
    assert year_result.status is VerificationStatus.CONTRADICTED
    assert ProblemType.METADATA_ERROR in year_result.problem_types
    assert year_result.adjudication_rule == "E2"
    assert year_result.refuting_evidence_ids

    # ③ DOI 存在 → 已核实（但只核实"存在"这一层面）
    _, doi_result = result_for(report, "DOI 是否真实")
    assert doi_result is not None
    assert doi_result.status is VerificationStatus.VERIFIED
    assert doi_result.adjudication_rule == "E5"
    assert doi_result.supporting_evidence_ids
    assert "不代表其内容支持" in doi_result.rationale

    # ④ 摘要反驳数值 → 存在矛盾 + 数据错误
    _, number_result = result_for(report, "数值/统计类")
    assert number_result is not None
    assert number_result.status is VerificationStatus.CONTRADICTED
    assert ProblemType.NUMERIC_ERROR in number_result.problem_types
    assert number_result.adjudication_rule == "E3"

    # ⑧ 可追溯性：每条外部证据都有留档，且能点开
    external = [item for item in report.evidence if item.source.value != "原文内部"]
    assert external, "应当产生外部证据"
    for item in external:
        assert item.url and item.url.startswith("http")
        assert item.raw_ref and (tmp_path / item.raw_ref).exists()


# ==========================================================================
# ① 虚构 DOI
# ==========================================================================
@respx.mock
def test_fabricated_doi_is_contradicted(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(404, json={"status": "not found"})
    )
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )

    report = run(tmp_path)
    _, doi_result = result_for(report, "DOI 是否真实")

    assert doi_result is not None
    assert doi_result.status is VerificationStatus.CONTRADICTED
    assert ProblemType.CITATION_MISSING in doi_result.problem_types
    assert doi_result.adjudication_rule == "E1"
    assert doi_result.severity.value == "高"
    assert doi_result.refuting_evidence_ids
    for evidence_id in doi_result.refuting_evidence_ids:
        evidence = report.evidence_by_id(evidence_id)
        assert evidence is not None and "404" in (evidence.relation_rationale or "")


# ==========================================================================
# ⑤ 模型引用的原句在摘要里找不到 → 丢弃该判断
# ==========================================================================
@respx.mock
def test_fabricated_quote_is_rejected_and_stays_insufficient(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler(refute_quote=FAKE_QUOTE))
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(200, json=crossref_payload(year=2008))
    )
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(200, json=openalex_payload(year=2008))
    )

    report = run(tmp_path)
    _, number_result = result_for(report, "数值/统计类")

    assert number_result is not None
    assert number_result.status is VerificationStatus.INSUFFICIENT, "无法复核的引用不足以支撑矛盾结论"
    assert number_result.adjudication_rule == "N6"
    assert any("找不到" in note for note in report.cross_check_notes), "被丢弃的判断必须记在报告里"


# ==========================================================================
# ⑥ 检索全部失败 → 暂无法核验
# ==========================================================================
@respx.mock
def test_retrieval_failure_yields_unverifiable_not_contradicted(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    respx.get(url__regex=CROSSREF).mock(side_effect=httpx.ConnectError("down"))
    respx.get(url__regex=OPENALEX).mock(side_effect=httpx.ConnectError("down"))

    report = run(tmp_path)
    assert report.verification_results
    assert all(
        result.status is VerificationStatus.UNVERIFIABLE
        for result in report.verification_results
        if result.status is not VerificationStatus.INSUFFICIENT
    ) or any(result.adjudication_rule == "N0" for result in report.verification_results)
    assert any(result.adjudication_rule == "N0" for result in report.verification_results)
    assert not any(result.is_suspicious for result in report.verification_results), (
        "检索失败绝不能被当成内容有问题的证据"
    )


# ==========================================================================
# ⑦ 只有元数据（没有摘要）→ 只核验存在性，不判断内容
# ==========================================================================
@respx.mock
def test_metadata_only_evidence_does_not_judge_content(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(200, json=crossref_payload(year=2008))
    )
    # OpenAlex 这条记录没有摘要
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(200, json=openalex_payload(year=2008, with_abstract=False))
    )

    report = run(tmp_path)

    # 年份一致 → 已核实
    _, year_result = result_for(report, "年份类")
    assert year_result is not None and year_result.status is VerificationStatus.VERIFIED
    assert year_result.adjudication_rule == "E6"

    # 数值类：只有元数据 → 证据不足（缺少内容证据），且绝不判成错误
    _, number_result = result_for(report, "数值/统计类")
    assert number_result is not None
    assert number_result.status is VerificationStatus.INSUFFICIENT
    assert number_result.adjudication_rule == "N6"
    assert not number_result.is_suspicious
    assert any("没有拿到摘要或全文" in item for item in [number_result.rationale])

    # 报告必须把"只读到摘要层面"的限制写清楚
    assert any("摘要" in item for item in report.limitations)


# ==========================================================================
# 检索统计与运行模式
# ==========================================================================
@respx.mock
def test_metrics_and_cross_check_report_the_retrieval(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(200, json=openalex_payload())
    )

    report = run(tmp_path)

    assert report.run_mode is RunMode.SINGLE_MODEL_MULTI_ROLE
    assert report.metrics.retrieval_queries >= 1
    assert report.metrics.retrieval_requests >= 2
    assert any("检索" in note for note in report.limitations)
    assert report.stats.status_counts[VerificationStatus.VERIFIED.value] >= 1
