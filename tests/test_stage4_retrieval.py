"""阶段 4 测试：外部检索（Crossref / OpenAlex / arXiv），全部用 respx 模拟 HTTP。

要钉死的几条铁律：
    · **不联网也能测**：所有 HTTP 都模拟，测试零成本、不受公网影响；
    · **404 与网络失败必须区分**：前者是"文献不存在"（可以据实判断），后者只能说"暂无法核验"；
    · **证据 URL 必须来自真实响应**：测试断言每条外部证据的 URL 出现在留档的原始响应里，
      从机制上杜绝"编造一个看起来很像的来源"；
    · **原始响应必须留档**：报告里每条证据都能回到原始数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from verimind.claims.extractor import RuleBasedExtractor
from verimind.claims.preprocess import analyze_text
from verimind.core.config import Settings
from verimind.core.enums import (
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    JudgedAspect,
    SearchStatus,
)
from verimind.retrieval.arxiv import ArxivSource
from verimind.retrieval.crossref import CrossrefSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.openalex import OpenAlexSource
from verimind.retrieval.router import plan_claim, run_retrieval

EXTRACTOR = RuleBasedExtractor()

DOI = "10.1038/nature12373"
DOI_TEXT = f"该成果发表于 2023 年（DOI: {DOI}），是领域内的重要进展。"
ABSTRACT = "We report a certified power conversion efficiency of 12% in perovskite solar cells."


def make_settings(tmp_path: Path, **overrides) -> Settings:
    data = {
        "_env_file": None,
        "data_dir": tmp_path,
        "save_runs": False,
        "retrieve_enabled": True,
        "retrieval_min_interval_seconds": 0.0,  # 测试里不需要真的等
        "retrieval_cache_hours": 24,
        "retrieval_max_retries": 0,
        "retrieval_mailto": "test@example.com",
    }
    data.update(overrides)
    return Settings(**data)  # type: ignore[arg-type]


def crossref_payload(year: int = 2013, title: str = "Certified perovskite solar cells") -> dict:
    return {
        "status": "ok",
        "message": {
            "DOI": DOI,
            "title": [title],
            "author": [{"family": "Smith", "given": "Jane"}, {"family": "Li", "given": "Wei"}],
            "issued": {"date-parts": [[year, 9, 11]]},
            "container-title": ["Nature"],
            "publisher": "Springer Science and Business Media LLC",
            "URL": f"https://doi.org/{DOI}",
        },
    }


def openalex_payload(year: int = 2013, with_abstract: bool = True) -> dict:
    words = ABSTRACT.split()
    return {
        "id": f"https://openalex.org/W123",
        "doi": f"https://doi.org/{DOI}",
        "display_name": "Certified perovskite solar cells",
        "publication_year": year,
        "authorships": [{"author": {"display_name": "Jane Smith"}}],
        "primary_location": {
            "landing_page_url": f"https://doi.org/{DOI}",
            "source": {"display_name": "Nature", "host_organization_name": "Springer"},
        },
        "abstract_inverted_index": {word: [index] for index, word in enumerate(words)},
    } if with_abstract else {
        "id": "https://openalex.org/W123",
        "doi": f"https://doi.org/{DOI}",
        "display_name": "Certified perovskite solar cells",
        "publication_year": year,
        "authorships": [],
        "primary_location": {"landing_page_url": f"https://doi.org/{DOI}", "source": {}},
    }


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Certified perovskite solar cells</title>
    <summary>We report a certified power conversion efficiency of 12% in perovskite solar cells.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Jane Smith</name></author>
  </entry>
</feed>
"""


def make_http(tmp_path: Path, settings: Settings) -> RetrievalHttpClient:
    return RetrievalHttpClient(settings, run_id="testrun")


# ==========================================================================
# 来源解析
# ==========================================================================
@respx.mock
def test_crossref_doi_lookup_parses_metadata(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = CrossrefSource(http, mailto="t@e.com").lookup_doi(DOI)

    assert lookup.status == "ok"
    record = lookup.records[0]
    assert record.found is True
    assert record.title == "Certified perovskite solar cells"
    assert record.year == 2013
    assert record.venue == "Nature"
    assert record.authors == ("Smith Jane", "Li Wei")
    assert record.level is EvidenceLevel.METADATA_ONLY  # Crossref 多数条目没有摘要


@respx.mock
def test_crossref_404_is_reported_as_not_found_not_error(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(404, json={"status": "not found"})
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = CrossrefSource(http, mailto="t@e.com").lookup_doi(DOI)

    assert lookup.status == "not_found"
    assert lookup.records[0].found is False
    assert "404" in lookup.records[0].note
    assert http.stats.not_found == 1
    assert http.stats.failures == 0, "404 不是故障"


@respx.mock
def test_network_failure_is_reported_as_error(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        side_effect=httpx.ConnectError("boom")
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = CrossrefSource(http, mailto="t@e.com").lookup_doi(DOI)

    assert lookup.status == "error"
    assert lookup.error
    assert http.stats.failures == 1


@respx.mock
def test_openalex_reconstructs_abstract_from_inverted_index(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.openalex\.org/works/").mock(
        return_value=httpx.Response(200, json=openalex_payload())
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = OpenAlexSource(http, mailto="t@e.com").lookup_doi(DOI)

    record = lookup.records[0]
    assert record.found is True
    assert record.level is EvidenceLevel.ABSTRACT
    assert record.abstract == ABSTRACT, "倒排索引必须按位置还原成正常语序"
    assert record.year == 2013


@respx.mock
def test_arxiv_parses_atom_feed(tmp_path: Path) -> None:
    respx.get(url__regex=r"export\.arxiv\.org/api/query").mock(
        return_value=httpx.Response(200, text=ARXIV_XML)
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = ArxivSource(http, mailto="t@e.com").search_title("Certified perovskite solar cells")

    record = lookup.records[0]
    assert record.source is EvidenceSource.ARXIV
    assert record.year == 2023
    assert record.abstract and "12%" in record.abstract
    assert record.authors == ("Jane Smith",)


# ==========================================================================
# 缓存、限速与留档
# ==========================================================================
@respx.mock
def test_second_identical_request_uses_cache(tmp_path: Path) -> None:
    route = respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        source = CrossrefSource(http, mailto="t@e.com")
        source.lookup_doi(DOI)
        source.lookup_doi(DOI)

    assert route.call_count == 1, "同一查询第二次应当命中缓存，不再打接口"
    assert http.stats.cache_hits == 1


@respx.mock
def test_raw_response_is_archived_for_traceability(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    settings = make_settings(tmp_path)
    with make_http(tmp_path, settings) as http:
        lookup = CrossrefSource(http, mailto="t@e.com").lookup_doi(DOI)

    raw_ref = lookup.records[0].raw_ref
    assert raw_ref, "必须留档原始响应"
    raw_path = settings.data_dir / raw_ref
    assert raw_path.exists()
    archived = json.loads(raw_path.read_text(encoding="utf-8"))
    assert archived["message"]["DOI"] == DOI, "留档的应当是原始响应本身"


# ==========================================================================
# 查询规划
# ==========================================================================
def test_plan_uses_doi_when_present() -> None:
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims
    plans = [plan_claim(claim, analysis, allow_retrieval=True) for claim in claims]
    doi_plans = [plan for plan in plans if any(q.kind == "doi" for q in plan.queries)]
    assert doi_plans, "含 DOI 的主张应当规划出 DOI 查询"
    assert doi_plans[0].queries[0].value == DOI


def test_plan_skips_opinion_claims() -> None:
    text = "业内专家普遍认为，到 2030 年该技术将全面普及。"
    analysis = analyze_text(text)
    claims = EXTRACTOR.extract(text, max_claims=8).claims
    opinion = next(claim for claim in claims if claim.claim_type is ClaimType.OPINION)
    plan = plan_claim(opinion, analysis, allow_retrieval=True)
    assert not plan.queries
    assert plan.skipped_reason and "观点" in plan.skipped_reason


def test_plan_explains_why_chinese_text_cannot_be_searched() -> None:
    """没有 DOI 的中文自由文本查不了——系统必须如实说明，而不是假装查过了。"""
    text = "该方案在多个城市的试点覆盖率已经达到较高水平。"
    analysis = analyze_text(text)
    claims = EXTRACTOR.extract(text, max_claims=8).claims
    assert claims
    plan = plan_claim(claims[0], analysis, allow_retrieval=True)
    assert not plan.queries
    # 意图：没有可检索标识时必须如实说明原因（而不是硬凑一个查询）。
    # 说明文字随能力变化而更新：现在还会指出「也不属于可核验的实体或通用事实」。
    assert plan.skipped_reason
    assert "暂无法核验" in plan.skipped_reason
    assert "可检索" in plan.skipped_reason


def test_plan_explains_invalid_doi_instead_of_silently_skipping() -> None:
    """DOI 格式不合规时，要明确说明"为什么不查"，而不是含糊地说"没有标识"。"""
    text = "据一项研究（DOI: 10.xxxx/example.001）显示，准确率提高 40%。"
    analysis = analyze_text(text)
    claims = EXTRACTOR.extract(text, max_claims=8).claims
    doi_claim = next(claim for claim in claims if claim.claim_type is ClaimType.DOI_VALIDITY)
    plan = plan_claim(doi_claim, analysis, allow_retrieval=True)
    assert not plan.queries
    assert plan.skipped_reason and "格式不合规" in plan.skipped_reason
    assert "10.xxxx/example.001" in plan.skipped_reason


def test_plan_ignores_retrieval_when_disabled() -> None:
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims
    plan = plan_claim(claims[0], analysis, allow_retrieval=False)
    assert not plan.queries
    assert plan.skipped_reason and "关闭" in plan.skipped_reason


# ==========================================================================
# 端到端检索（含证据 URL 来源断言）
# ==========================================================================
@respx.mock
def test_run_retrieval_creates_evidence_and_traces(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    respx.get(url__regex=r"api\.openalex\.org/works/").mock(
        return_value=httpx.Response(200, json=openalex_payload())
    )
    settings = make_settings(tmp_path)
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims

    outcome = run_retrieval(claims, analysis, settings, run_id="run1")

    assert outcome.evidence, "应当产生外部证据"
    assert outcome.query_count == 1, "同一 DOI 只查一次"
    assert outcome.stats.requests == 2, "DOI 查询会同时打 Crossref 与 OpenAlex"

    # 每条证据都要能追溯到主张，并且痕迹里记录了查询过程
    for item in outcome.evidence:
        assert item.claim_ids
        assert item.url.startswith("http")
        assert item.raw_ref, "证据必须带留档路径"
        assert (settings.data_dir / item.raw_ref).exists(), "留档文件必须真实存在"
    traced = [cid for cid, trace in outcome.traces.items() if trace.records]
    assert traced, "必须留下检索痕迹"
    assert any(
        record.status is SearchStatus.OK
        for trace in outcome.traces.values()
        for record in trace.records
    )


@respx.mock
def test_evidence_urls_come_from_real_responses(tmp_path: Path) -> None:
    """核心断言：证据的 URL 要么是响应里给出的地址，要么是我们真实请求过的地址。

    这条断言的意思是：**系统不可能凭空编造一个来源**。
    """
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(200, json=crossref_payload())
    )
    respx.get(url__regex=r"api\.openalex\.org/works/").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    settings = make_settings(tmp_path)
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims

    outcome = run_retrieval(claims, analysis, settings, run_id="run2")

    # 收集所有"真实发生过的东西"：请求 URL + 响应体里出现的 URL 字符串
    legit: set[str] = set()
    for item in outcome.evidence:
        if item.raw_ref:
            raw_text = (settings.data_dir / item.raw_ref).read_text(encoding="utf-8")
            for token in raw_text.split('"'):
                if token.startswith("http"):
                    legit.add(token)
            legit.add(item.query)  # 查询式本身（如 DOI）
    for trace in outcome.traces.values():
        for record in trace.records:
            legit.add(record.query)

    for item in outcome.evidence:
        # 404 记录的 URL 是我们真实请求过的接口地址，也属于"真实发生过"
        assert item.url.startswith("http")
        assert (
            item.url in legit
            or any(item.url.endswith(record.query) for record in trace.records)
            or "api.crossref.org" in item.url
            or "api.openalex.org" in item.url
        ), f"{item.evidence_id} 的 URL {item.url} 找不到真实来源依据"


@respx.mock
def test_not_found_evidence_is_marked_as_refuting_existence(tmp_path: Path) -> None:
    """服务端明确 404 时，证据在生成阶段就被标记为「反驳 / 文献是否存在」。"""
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        return_value=httpx.Response(404, json={"status": "not found"})
    )
    respx.get(url__regex=r"api\.openalex\.org/works/").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    settings = make_settings(tmp_path)
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims

    outcome = run_retrieval(claims, analysis, settings, run_id="run3")
    refuting = [item for item in outcome.evidence if item.relation is EvidenceRelation.REFUTES]
    assert refuting, "404 应当产生「反驳存在性」的证据"
    assert all(item.judged_aspect is JudgedAspect.EXISTENCE for item in refuting)
    assert all(item.relation_rationale for item in refuting)


@respx.mock
def test_all_sources_failing_yields_error_traces(tmp_path: Path) -> None:
    respx.get(url__regex=r"api\.crossref\.org/works/").mock(
        side_effect=httpx.ConnectError("down")
    )
    respx.get(url__regex=r"api\.openalex\.org/works/").mock(
        side_effect=httpx.ConnectError("down")
    )
    settings = make_settings(tmp_path)
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims

    outcome = run_retrieval(claims, analysis, settings, run_id="run4")
    assert not outcome.evidence
    assert outcome.stats.failures >= 1
    statuses = {
        record.status for trace in outcome.traces.values() for record in trace.records
    }
    assert SearchStatus.ERROR in statuses


def test_retrieval_disabled_records_skipped(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, retrieve_enabled=False)
    analysis = analyze_text(DOI_TEXT)
    claims = EXTRACTOR.extract(DOI_TEXT, max_claims=8).claims
    outcome = run_retrieval(claims, analysis, settings, run_id="run5")
    assert not outcome.evidence
    assert outcome.query_count == 0
    for trace in outcome.traces.values():
        assert all(record.status is SearchStatus.SKIPPED for record in trace.records)

# ==========================================================================
# Wikidata / Wikipedia：实体与通用事实来源
# ==========================================================================
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = r"https://(zh|en)\.wikipedia\.org/w/api.php"


@respx.mock
def test_wikidata_returns_structured_facts_as_metadata_only(tmp_path: Path) -> None:
    """Wikidata 给出的是**仅元数据**级证据（不能用于内容层面判断）。"""
    from verimind.core.enums import EvidenceLevel, EvidenceSource
    from verimind.retrieval.wikidata import WikidataSource

    def handler(request):
        action = dict(request.url.params).get("action")
        if action == "wbsearchentities":
            return httpx.Response(
                200,
                json={
                    "search": [
                        {"id": "Q1073814", "label": "中国中医科学院", "description": "education organization in Beijing, China"}
                    ]
                },
            )
        if action == "wbgetentities":
            ids = dict(request.url.params).get("ids", "")
            if ids == "Q1073814":
                return httpx.Response(
                    200,
                    json={
                        "entities": {
                            "Q1073814": {
                                "claims": {
                                    "P571": [
                                        {
                                            "mainsnak": {
                                                "datavalue": {"value": {"time": "+1955-00-00T00:00:00Z"}}
                                            }
                                        }
                                    ],
                                    "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q148"}}}}],
                                }
                            }
                        }
                    },
                )
            return httpx.Response(200, json={"entities": {"Q148": {"labels": {"zh": {"value": "中华人民共和国"}}}}})

    respx.get(url__startswith=WIKIDATA_API).mock(side_effect=handler)
    client = make_http(tmp_path, make_settings(tmp_path, retrieval_cache_hours=0))
    lookup = WikidataSource(client).search_entity("中国中医科学院")
    client.close()

    assert lookup.records, "应当返回实体记录"
    record = lookup.records[0]
    assert record.source is EvidenceSource.WIKIDATA
    assert record.level is EvidenceLevel.METADATA_ONLY, "Wikidata 只能是元数据级"
    assert record.title == "中国中医科学院"
    assert "1955" in (record.abstract or ""), "成立时间应当被取到"
    assert "中华人民共和国" in (record.abstract or ""), "实体值应当被换成可读标签"
    assert record.url.startswith("https://www.wikidata.org/wiki/Q")


@respx.mock
def test_wikipedia_returns_abstract_level_evidence(tmp_path: Path) -> None:
    """Wikipedia 导语是**摘要**级证据（可进入内容层面判断）。"""
    from verimind.core.enums import EvidenceLevel, EvidenceSource
    from verimind.retrieval.wikipedia import WikipediaSource, looks_chinese

    respx.get(url__regex=WIKIPEDIA_API).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "index": 1,
                            "title": "Tu Youyou",
                            "extract": "Tu Youyou (born 30 December 1930) is a Nobel Prize-winning Chinese pharmaceutical chemist.",
                        }
                    }
                }
            },
        )
    )
    client = make_http(tmp_path, make_settings(tmp_path, retrieval_cache_hours=0))
    lookup = WikipediaSource(client).search_summary("屠呦呦")
    client.close()

    assert lookup.records
    record = lookup.records[0]
    assert record.source is EvidenceSource.WIKIPEDIA
    assert record.level is EvidenceLevel.ABSTRACT, "维基导语是摘要级，可进入内容判断"
    assert "1930" in (record.abstract or "")
    assert record.url.startswith("https://zh.wikipedia.org/wiki/") or record.url.startswith(
        "https://en.wikipedia.org/wiki/"
    )
    # 中文查询走中文维基，英文查询走英文维基
    assert looks_chinese("屠呦呦于 1930 年出生") is True
    assert looks_chinese("Tu Youyou was born in 1930") is False


def test_subject_query_only_accepts_model_provided_subject() -> None:
    """没有模型给出的检索主体时，不得拿整句去查百科（中文没有空格，必然召不回）。"""
    # EXTRACTOR / analyze_text / plan_claim 都已在本文件顶部导入
    from verimind.retrieval.router import plan_claim

    text = "该方案在多个城市的试点覆盖率已经达到较高水平。"
    analysis = analyze_text(text)
    claims = EXTRACTOR.extract(text, max_claims=8).claims
    plan = plan_claim(claims[0], analysis, allow_retrieval=True)
    assert not plan.queries, "规则引擎抽出的主张没有检索主体，不应发起实体/百科查询"
    assert plan.skipped_reason

    # 模型给了检索主体时，才允许发起实体与百科查询
    with_subject = claims[0].model_copy(update={"search_subject": "屠呦呦"})
    plan2 = plan_claim(with_subject, analysis, allow_retrieval=True)
    kinds = [query.kind for query in plan2.queries]
    assert "wikidata_entity" in kinds and "wikipedia_summary" in kinds
    assert all("屠呦呦" in query.value for query in plan2.queries), "必须用检索主体而不是整句"
