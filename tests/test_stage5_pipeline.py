"""阶段 5 测试：端到端（真实检索链路 + 模拟 LLM）。

验收标准：
    ① 年份写错 → 生成修正（改成证据里的年份）→ 复核通过 → **修正后全文里年份被改正**；
    ② 数值被摘要反驳 → 修正成摘要里的数值 → 复核通过 → 应用；
    ③ 修正里出现证据中没有的数字 → **确定性检查否决** → 不应用，并在报告里列为"被拒绝"；
    ④ 复核环节整个失败 → 一律不应用（未复核的改动不许进正文）；
    ⑤ 离线规则引擎模式 → 不生成修正，并说明原因。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
import respx

from verimind.core.config import Settings
from verimind.core.enums import ChangeType, ReviewOverall, VerificationStatus
from verimind.core.pipeline import run_detection
from verimind.core.schemas import DetectionReport

CHAT = "https://api.deepseek.com/v1/chat/completions"
CROSSREF = r"api\.crossref\.org/works/"
OPENALEX = r"api\.openalex\.org/works/"

DOI = "10.1038/nature12373"
TEXT = f"某研究（DOI: {DOI}）发表于 2008 年，显示准确率提高了 40%。"
ABSTRACT = "We report a certified power conversion efficiency of 12% in perovskite solar cells."
QUOTE = "a certified power conversion efficiency of 12%"

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


def _pairs(prompt: str) -> list[dict]:
    return json.loads(prompt.split("<<<PAIRS", 1)[1].split("PAIRS>>>", 1)[0])


def _dossiers(prompt: str) -> list[dict]:
    return json.loads(prompt.split("<<<DOSSIERS", 1)[1].split("DOSSIERS>>>", 1)[0])


def _items(prompt: str) -> list[dict]:
    return json.loads(prompt.split("<<<ITEMS", 1)[1].split("ITEMS>>>", 1)[0])


def _correction_payload(prompt: str, *, number_override: str | None = None) -> dict:
    """从材料里读出真实的 claim_id / evidence_id / 片段，动态构造修正建议。"""
    corrections: list[dict] = []
    for dossier in _dossiers(prompt):
        quote = dossier["主张原文片段"]
        refuting = [item for item in dossier["可用证据"] if item.get("关系") == "反驳"]
        if not refuting:
            continue
        if "年" in quote:
            for item in refuting:
                match = re.search(r"(20\d{2})\s*年", item.get("判断理由") or "")
                if match:
                    corrections.append(
                        {
                            "claim_id": dossier["claim_id"],
                            "target_quote": quote,
                            "proposed_text": f"{match.group(1)} 年",
                            "change_type": "更正数字",
                            "rationale": f"{item['evidence_id']} 显示该文献发表于 {match.group(1)} 年。",
                            "evidence_ids": [item["evidence_id"]],
                        }
                    )
                    break
        elif "%" in quote:
            value = number_override
            evidence_id = refuting[0]["evidence_id"]
            if value is None:
                for item in refuting:
                    match = re.search(r"(\d+(?:\.\d+)?)\s*%", item.get("片段") or "")
                    if match:
                        value = match.group(1)
                        evidence_id = item["evidence_id"]
                        break
            if value:
                corrections.append(
                    {
                        "claim_id": dossier["claim_id"],
                        "target_quote": quote,
                        "proposed_text": f"{value}%",
                        "change_type": "更正数字",
                        "rationale": f"{evidence_id} 中的数值是 {value}%。",
                        "evidence_ids": [evidence_id],
                    }
                )
    return {"corrections": corrections, "notes": []}


def _review_payload(prompt: str, *, issue_resolved: bool = True) -> dict:
    return {
        "reviews": [
            {
                "target_correction_id": item["correction_id"],
                "issue_resolved": issue_resolved,
                "issue_resolved_detail": "修改后的内容与证据一致。" if issue_resolved else "问题没有真正解决。",
                "no_new_overreach": True,
                "no_new_overreach_detail": "没有引入新的论断。",
                "unresolved": [],
            }
            for item in _items(prompt)
        ],
        "notes": [],
    }


def chat_handler(
    *, number_override: str | None = None, review_fails: bool = False, review_errors: bool = False
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = " ".join(str(message.get("content", "")) for message in body.get("messages", []))
        if "拆解为原子事实主张" in prompt:
            payload = EXTRACTION_PAYLOAD
        elif "证据审查员" in prompt:
            payload = {
                "assessments": [
                    {
                        "evidence_id": pair["evidence_id"],
                        "claim_id": pair["claim_id"],
                        "relation": "反驳",
                        "judged_aspect": "内容结论是否被支持",
                        "rationale": "摘要给出的效率是 12%，与原文的 40% 不符。",
                        "quoted_snippet": QUOTE,
                    }
                    for pair in _pairs(prompt)
                ],
                "notes": [],
            }
        # 注意顺序：复核提示词里也会出现"修正员"（"你不是修正员…"），
        # 所以必须先判复核，否则会把修正员的响应喂给复核请求。
        elif "独立复核员" in prompt:
            if review_errors:
                return httpx.Response(500, json={"error": {"message": "review unavailable"}})
            payload = _review_payload(prompt, issue_resolved=not review_fails)
        elif "修正员" in prompt:
            payload = _correction_payload(prompt, number_override=number_override)
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


def openalex_payload(year: int = 2013) -> dict:
    words = ABSTRACT.split()
    return {
        "id": "https://openalex.org/W1",
        "doi": f"https://doi.org/{DOI}",
        "display_name": "Certified perovskite solar cells",
        "publication_year": year,
        "authorships": [{"author": {"display_name": "Jane Smith"}}],
        "primary_location": {
            "landing_page_url": f"https://doi.org/{DOI}",
            "source": {"display_name": "Nature", "host_organization_name": "Springer"},
        },
        "abstract_inverted_index": {word: [index] for index, word in enumerate(words)},
    }


def make_settings(tmp_path: Path, *, offline: bool = False, **overrides) -> Settings:
    data = {
        "_env_file": None,
        "data_dir": tmp_path,
        "save_runs": False,
        "llm_provider": "mock" if offline else "deepseek",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-chat",
        "llm_api_key": "" if offline else "sk-" + "x" * 20,
        "llm_cache_enabled": False,
        "llm_max_retries": 0,
        "retrieve_enabled": True,
        "retrieval_min_interval_seconds": 0.0,
        "retrieval_max_retries": 0,
        "retrieval_mailto": "test@example.com",
    }
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


def mock_sources(year: int = 2013) -> None:
    respx.get(url__regex=CROSSREF).mock(
        return_value=httpx.Response(200, json=crossref_payload(year=year))
    )
    respx.get(url__regex=OPENALEX).mock(
        return_value=httpx.Response(200, json=openalex_payload(year=year))
    )


# ==========================================================================
# ① ② 年份与数值都被改正
# ==========================================================================
@respx.mock
def test_corrections_fix_year_and_number_and_are_applied(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    mock_sources(year=2013)

    report = run_detection(TEXT, settings=make_settings(tmp_path))

    assert report.corrections, "存在矛盾的主张应当生成修正建议"
    assert report.corrected_text is not None
    assert "2013 年" in report.corrected_text, "年份应当被改成证据里的 2013 年"
    assert "2008" not in report.corrected_text
    assert "12%" in report.corrected_text, "数值应当被改成摘要里的 12%"
    assert "40%" not in report.corrected_text

    assert report.applied_correction_ids, "复核通过的修正应当被应用"
    assert report.review_results
    assert all(
        review.overall in (ReviewOverall.PASSED, ReviewOverall.PASSED_WITH_RISKS)
        for review in report.review_results
    )

    # 每条修正都能追溯到证据
    for correction in report.corrections:
        assert correction.evidence_ids, "写入新事实的修正必须挂证据"
        assert correction.change_type is ChangeType.FIX_NUMBER


# ==========================================================================
# ③ 编造的数字被确定性检查拦住
# ==========================================================================
@respx.mock
def test_invented_number_is_rejected_by_deterministic_checks(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler(number_override="25"))
    mock_sources(year=2013)

    report = run_detection(TEXT, settings=make_settings(tmp_path))

    number_corrections = [
        item for item in report.corrections if "25%" in item.proposed_text
    ]
    assert number_corrections, "模型确实提出了一个编造的数值"
    assert number_corrections[0].correction_id in report.rejected_correction_ids, (
        "证据里没有 25%，必须被拒绝"
    )
    assert report.corrected_text is not None
    assert "25%" not in report.corrected_text, "被拒绝的修正绝不能出现在修正后全文里"

    failed = [
        review
        for review in report.review_results
        if review.target_correction_id == number_corrections[0].correction_id
    ]
    assert failed and failed[0].overall is ReviewOverall.FAILED
    assert any(
        "25%" in check.detail
        for check in failed[0].checks
        if not check.passed
    )
    # 年份那条修正仍然应当被正常采用
    assert "2013 年" in report.corrected_text


# ==========================================================================
# ④ 复核整个失败 → 一律不应用
# ==========================================================================
@respx.mock
def test_corrections_are_not_applied_when_review_fails(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler(review_errors=True))
    mock_sources(year=2013)

    report = run_detection(TEXT, settings=make_settings(tmp_path))

    assert report.corrections, "修正仍然会被生成"
    assert not report.applied_correction_ids, "未复核的修正不许写进正文"
    assert report.rejected_correction_ids == [
        item.correction_id for item in report.corrections
    ]
    assert report.corrected_text == TEXT, "一条都没应用时，修正后全文应当与原文一致"
    assert any("独立复核未能完成" in item for item in report.cross_check_notes)


@respx.mock
def test_corrections_are_not_applied_when_reviewer_rejects(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler(review_fails=True))
    mock_sources(year=2013)

    report = run_detection(TEXT, settings=make_settings(tmp_path))

    assert report.corrections
    assert not report.applied_correction_ids
    assert all(review.overall is ReviewOverall.FAILED for review in report.review_results)
    assert report.corrected_text == TEXT


# ==========================================================================
# ⑤ 离线模式
# ==========================================================================
def test_offline_mode_produces_no_corrections(tmp_path: Path) -> None:
    report = run_detection(TEXT, settings=make_settings(tmp_path, offline=True))

    assert report.corrections == []
    assert report.review_results == []
    assert report.corrected_text is None
    assert any("离线规则引擎模式" in note and "修正" in note for note in report.cross_check_notes)


# ==========================================================================
# 统计与自洽
# ==========================================================================
@respx.mock
def test_report_stats_reflect_corrections_and_reviews(tmp_path: Path) -> None:
    respx.post(CHAT).mock(side_effect=chat_handler())
    mock_sources(year=2013)

    report = run_detection(TEXT, settings=make_settings(tmp_path))

    assert report.stats.corrections_count == len(report.corrections)
    assert report.stats.reviews_passed + report.stats.reviews_failed == len(
        report.review_results
    )
    # 被拒的修正对应的主张要进"未解决问题"清单
    rejected_claims = {
        item.claim_id
        for item in report.corrections
        if item.correction_id in report.rejected_correction_ids
    }
    assert rejected_claims <= set(report.unresolved_claim_ids)
    assert report.stats.status_counts[VerificationStatus.CONTRADICTED.value] >= 2
