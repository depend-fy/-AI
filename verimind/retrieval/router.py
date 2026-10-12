"""检索路由：决定"哪条主张该去查什么、用什么查询式"，并统一生成证据。

设计要点
--------
1. **按主张类型决定查不查、查什么**：
   - 有 DOI 的主张 → Crossref（权威注册信息）+ OpenAlex（常常带摘要）；
   - 只有标题/英文片段的主张 → Crossref 书目检索 + OpenAlex 检索 + arXiv 标题检索；
   - 观点/预测类 → 不检索（它本来就不是可核验的事实主张）；
   - 中文自由文本、又没有 DOI → 记下"查过了但没结果"，结论只能是「暂无法核验」。
     这不是系统偷懒：Crossref/OpenAlex 收录的是学术元数据，对中文自由文本几乎无能为力，
     如实承认这一点比硬凑一个结论重要得多。
2. **同一 DOI 只查一次**：多个主张共用一次查询结果，证据对象共用、claim_ids 累加。
3. **失败与"查不到"严格区分**：网络失败 → `SearchStatus.ERROR`（结论只能是「暂无法核验」）；
   服务端明确 404 → `SearchStatus.NO_HIT`（可以据实判断"该 DOI 未注册"）。
4. **全部留痕**：每次查询都写进 SearchTrace，包括查询式、命中数、原始响应留档路径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verimind.claims.preprocess import DocumentAnalysis
from verimind.core.config import Settings
from verimind.core.enums import (
    ClaimType,
    Checkability,
    EvidenceRelation,
    EvidenceSource,
    JudgedAspect,
    MatchBasis,
    SearchStatus,
)
from verimind.core.schemas import Claim, Evidence, SearchRecord, SearchTrace
from verimind.retrieval.arxiv import ArxivSource
from verimind.retrieval.crossref import CrossrefSource
from verimind.retrieval.http import RetrievalHttpClient, RetrievalStats
from verimind.retrieval.openalex import OpenAlexSource
from verimind.retrieval.parsing import SourceLookup, SourceRecord

#: 从中文书名号里取候选标题；以及一段连续拉丁文（≥2 个词）当作英文标题候选
_TITLE_IN_BRACKETS_RE = re.compile(r"《([^》]{4,80})》")
_LATIN_PHRASE_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-' ]{7,80}")


@dataclass(frozen=True)
class PlannedQuery:
    """一条计划中的查询。"""

    kind: str  # "doi" | "bibliographic" | "arxiv_title"
    value: str
    reason: str


@dataclass(frozen=True)
class ClaimPlan:
    """某条主张的检索计划。"""

    claim_id: str
    queries: tuple[PlannedQuery, ...] = ()
    skipped_reason: str | None = None


@dataclass
class RetrievalOutcome:
    """一次运行的检索总产出。"""

    evidence: list[Evidence] = field(default_factory=list)
    traces: dict[str, SearchTrace] = field(default_factory=dict)
    evidence_ids_by_claim: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    stats: RetrievalStats = field(default_factory=RetrievalStats)
    query_count: int = 0

    @property
    def hit_claims(self) -> int:
        return sum(1 for ids in self.evidence_ids_by_claim.values() if ids)


# ==========================================================================
# 查询规划
# ==========================================================================
def _candidate_title(signals_text: str) -> str | None:
    """从一句话里找可检索的标题候选（英文片段或书名号内的内容）。"""
    brackets = _TITLE_IN_BRACKETS_RE.search(signals_text)
    if brackets:
        candidate = brackets.group(1).strip()
        if len(candidate) >= 4:
            return candidate
    latin = _LATIN_PHRASE_RE.search(signals_text)
    if latin:
        candidate = latin.group(0).strip(" -'")
        if len(candidate.split()) >= 2:
            return candidate
    return None


def plan_claim(claim: Claim, analysis: DocumentAnalysis, *, allow_retrieval: bool) -> ClaimPlan:
    """为一条主张决定检索策略。"""
    if not allow_retrieval:
        return ClaimPlan(claim.claim_id, skipped_reason="本次配置已关闭外部检索。")

    if claim.checkability is Checkability.NOT_CHECKABLE or claim.claim_type is ClaimType.OPINION:
        return ClaimPlan(
            claim.claim_id, skipped_reason="该主张属于观点/预测，不做外部检索（它本来就不是可核验的事实主张）。"
        )

    sentence = claim.parent_sentence_span or claim.source_span
    signals = analysis.for_span(sentence)

    # 优先用主张片段里的 DOI；没有就用整句里的 DOI
    dois = [
        match for match in signals.dois if match.span.start < claim.source_span.end and claim.source_span.start < match.span.end
    ] or list(signals.dois)

    if dois:
        doi = dois[0].text
        queries = [
            PlannedQuery("doi", doi, f"按 DOI 精确查询注册信息（{doi}）"),
        ]
        return ClaimPlan(claim.claim_id, queries=tuple(queries))

    title = _candidate_title(signals.text)
    if title:
        return ClaimPlan(
            claim.claim_id,
            queries=(
                PlannedQuery("bibliographic", title, f"按标题/书目信息检索：{title}"),
                PlannedQuery("arxiv_title", title, f"在 arXiv 按标题检索预印本：{title}"),
            ),
        )

    # 句子里有"像 DOI 但格式不合规"的串：连查都不用查——注册机构不接受这种写法。
    # 这件事由规则 N5 负责指出，检索层要做的只是如实说明"为什么不查"。
    invalid_dois = [
        match
        for match in signals.invalid_dois
        if match.span.start < claim.source_span.end and claim.source_span.start < match.span.end
    ] or list(signals.invalid_dois)
    if invalid_dois:
        return ClaimPlan(
            claim.claim_id,
            skipped_reason=(
                f"原文引用的「{invalid_dois[0].text}」格式不合规（DOI 前缀应为 4~9 位数字），"
                "无法作为检索标识提交给 DOI 注册机构。这类格式问题由规则 N5 指出。"
            ),
        )

    return ClaimPlan(
        claim.claim_id,
        skipped_reason=(
            "该主张没有可检索的标识（DOI、标题或作者），"
            "而原文是中文自由文本——Crossref/OpenAlex 主要收录学术元数据，"
            "无法据此有效检索。系统选择如实标记「暂无法核验」，而不是硬凑结论。"
        ),
    )


# ==========================================================================
# 执行检索
# ==========================================================================
def _to_evidence(record: SourceRecord, evidence_id: str, claim_id: str) -> Evidence:
    """把来源记录变成证据对象。

    注意两个刻意的设计：
    1. **证据是按"主张"生成的**（而不是全局共用一条）。同一篇文献对 A 主张支撑"存在"、
       对 B 主张可能反驳"内容"，如果把关系写在一个共用对象上必然打架。
       HTTP 层面依然只查一次（见 lookup_cache），重复的只是报告里的呈现。
    2. **"服务端明确 404"在解析层就表达为「反驳 / 文献是否存在」**：
       这是注册机构的权威答复，属于可以据实判断的结论；
       而"网络失败"绝不这样处理（它什么也证明不了）。
    """
    snippet: str | None = None
    if record.abstract:
        snippet = record.abstract[:600]
    elif record.note:
        snippet = record.note[:600]

    if record.found:
        relation = EvidenceRelation.UNKNOWN
        judged_aspect = None
        rationale = ""
    else:
        relation = EvidenceRelation.REFUTES
        judged_aspect = JudgedAspect.EXISTENCE
        rationale = record.note or f"{record.source.value} 中未找到该记录。"

    return Evidence(
        evidence_id=evidence_id,
        claim_ids=[claim_id],
        source=record.source,
        level=record.level,
        url=record.url,
        title=record.title,
        doi=record.doi,
        authors=list(record.authors),
        year=record.year,
        venue=record.venue,
        publisher=record.publisher,
        query=record.query,
        raw_ref=record.raw_ref,
        snippet=snippet,
        relation=relation,
        relation_rationale=rationale,
        match_basis=MatchBasis.METADATA if not record.found else MatchBasis.NONE,
        judged_aspect=judged_aspect,
    )


def _record_key(record: SourceRecord) -> tuple[str, str]:
    if record.doi:
        return (str(record.source.value), f"doi:{record.doi.lower()}")
    return (str(record.source.value), f"title:{record.title.lower()[:120]}")


def _status_of(lookup: SourceLookup) -> tuple[SearchStatus, str]:
    if lookup.status == "error":
        return SearchStatus.ERROR, lookup.error or "检索失败"
    if lookup.status == "not_found":
        return SearchStatus.NO_HIT, "服务端明确返回 404：该记录不存在"
    if not lookup.records:
        return SearchStatus.NO_HIT, "检索完成但没有命中任何记录"
    if all(not record.found for record in lookup.records):
        return SearchStatus.NO_HIT, "服务端明确返回 404：该记录不存在"
    return SearchStatus.OK, ""


def run_retrieval(
    claims: list[Claim],
    analysis: DocumentAnalysis,
    settings: Settings,
    *,
    run_id: str,
    start_index: int = 0,
    http: RetrievalHttpClient | None = None,
) -> RetrievalOutcome:
    """对所有主张执行检索，返回证据、检索痕迹与统计。"""
    outcome = RetrievalOutcome()
    plans = [
        plan_claim(claim, analysis, allow_retrieval=settings.retrieve_enabled) for claim in claims
    ]

    if not settings.retrieve_enabled:
        for claim, plan in zip(claims, plans):
            outcome.traces[claim.claim_id] = SearchTrace(
                records=[
                    SearchRecord(
                        source=None,
                        query=claim.normalized_claim[:80],
                        status=SearchStatus.SKIPPED,
                        hit_count=0,
                        detail=plan.skipped_reason or "未检索",
                    )
                ]
            )
        return outcome

    client = http or RetrievalHttpClient(settings, run_id=run_id)
    owns_client = http is None
    crossref = CrossrefSource(client, mailto=settings.retrieval_mailto)
    openalex = OpenAlexSource(client, mailto=settings.retrieval_mailto)
    arxiv = ArxivSource(client, mailto=settings.retrieval_mailto)

    lookup_cache: dict[tuple[str, str], list[SourceLookup]] = {}
    counter = start_index
    budget = settings.retrieval_max_queries_per_run

    try:
        for claim, plan in zip(claims, plans):
            records_for_claim: list[SourceRecord] = []
            trace_records: list[SearchRecord] = []

            if plan.skipped_reason and not plan.queries:
                trace_records.append(
                    SearchRecord(
                        source=None,
                        query=claim.normalized_claim[:80],
                        status=SearchStatus.SKIPPED,
                        hit_count=0,
                        detail=plan.skipped_reason,
                    )
                )
                outcome.traces[claim.claim_id] = SearchTrace(records=trace_records)
                continue

            for query in plan.queries:
                cache_key = (query.kind, query.value.lower())
                if cache_key not in lookup_cache:
                    if outcome.query_count >= budget:
                        trace_records.append(
                            SearchRecord(
                                source=None,
                                query=query.value[:80],
                                status=SearchStatus.SKIPPED,
                                hit_count=0,
                                detail=(
                                    f"已达本次运行的检索次数上限（{budget} 次），该查询未执行。"
                                    "可在 .env 调整 VERIMIND_RETRIEVAL_MAX_QUERIES_PER_RUN。"
                                ),
                            )
                        )
                        continue
                    outcome.query_count += 1
                    lookup_cache[cache_key] = _execute(
                        kind=query.kind,
                        value=query.value,
                        crossref=crossref,
                        openalex=openalex,
                        arxiv=arxiv,
                    )

                for lookup in lookup_cache[cache_key]:
                    status, detail = _status_of(lookup)
                    trace_records.append(
                        SearchRecord(
                            source=lookup.source,
                            query=query.value[:120],
                            status=status,
                            hit_count=len([record for record in lookup.records if record.found]),
                            error=lookup.error,
                            detail=f"{query.reason}｜{detail}" if detail else query.reason,
                        )
                    )
                    records_for_claim.extend(lookup.records)

            outcome.traces[claim.claim_id] = SearchTrace(records=trace_records)

            # 同一条主张内按 (来源, DOI/标题) 去重；不同主张各自持有自己的证据对象
            seen_keys: set[tuple[str, str]] = set()
            for record in records_for_claim:
                key = _record_key(record)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                counter += 1
                evidence_id = f"E{counter:03d}"
                outcome.evidence.append(_to_evidence(record, evidence_id, claim.claim_id))
                outcome.evidence_ids_by_claim.setdefault(claim.claim_id, []).append(evidence_id)

        outcome.stats = client.stats
    finally:
        if owns_client:
            client.close()

    if outcome.query_count == 0:
        outcome.notes.append("本次没有任何主张具备可检索的标识，因此未发起外部查询。")
    return outcome


def _execute(
    *,
    kind: str,
    value: str,
    crossref: CrossrefSource,
    openalex: OpenAlexSource,
    arxiv: ArxivSource,
) -> list[SourceLookup]:
    """执行一个查询。DOI 查询会同时打 Crossref 与 OpenAlex（互为交叉验证）。"""
    if kind == "doi":
        return [crossref.lookup_doi(value), openalex.lookup_doi(value)]
    if kind == "bibliographic":
        return [crossref.search_bibliographic(value), openalex.search(value)]
    if kind == "arxiv_title":
        return [arxiv.search_title(value)]
    return []


__all__ = [
    "ClaimPlan",
    "PlannedQuery",
    "RetrievalOutcome",
    "plan_claim",
    "run_retrieval",
]
