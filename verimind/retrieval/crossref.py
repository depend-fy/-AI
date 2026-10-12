"""Crossref：DOI 注册信息的权威来源（免密钥）。

用法上最关键的一点：**它既能证明"存在"，也能证明"不存在"。**
    · 查到 → 该 DOI 已注册，拿到权威的标题/作者/年份/期刊；
    · 404 → 该 DOI 未在 DOI 注册体系中登记（这是一个可以据实判断的结论）；
    · 网络失败 → 什么都说明不了，只能说"暂无法核验"。

Crossref 多数条目没有摘要，所以它通常只能支撑"文献存在 + 书目信息"层面的判断。
"""

from __future__ import annotations

from urllib.parse import quote

from verimind.core.enums import EvidenceLevel, EvidenceSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.parsing import (
    SourceLookup,
    SourceRecord,
    safe_year,
    strip_markup,
)

CROSSREF_API = "https://api.crossref.org"


class CrossrefSource:
    name = EvidenceSource.CROSSREF
    base_url = CROSSREF_API

    def __init__(self, http: RetrievalHttpClient, *, mailto: str = "") -> None:
        self.http = http
        self.mailto = mailto

    # ------------------------------------------------------------------
    def lookup_doi(self, doi: str) -> SourceLookup:
        """按 DOI 精确查询注册信息。"""
        doi = (doi or "").strip()
        lookup = SourceLookup(source=self.name, query=doi)
        if not doi:
            lookup.status = "error"
            lookup.error = "空 DOI"
            return lookup

        url = f"{self.base_url}/works/{quote(doi, safe='/')}"
        result = self.http.fetch(url, source=self.name, params={"mailto": self.mailto})
        if result.raw_ref:
            lookup.raw_refs.append(result.raw_ref)

        if result.failed:
            lookup.status = "error"
            lookup.error = result.error or f"HTTP {result.status_code}"
            return lookup

        if result.not_found:
            lookup.status = "not_found"
            lookup.records.append(
                SourceRecord(
                    source=self.name,
                    url=result.url,
                    level=EvidenceLevel.METADATA_ONLY,
                    query=doi,
                    raw_ref=result.raw_ref,
                    found=False,
                    note="Crossref 返回 404：该 DOI 未在 DOI 注册机构中登记。",
                )
            )
            return lookup

        payload = result.json() or {}
        message = payload.get("message") or {}
        if not message:
            lookup.status = "no_hit"
            return lookup
        lookup.records.append(self._record_from_message(message, doi, result.raw_ref, result.url))
        return lookup

    # ------------------------------------------------------------------
    def search_bibliographic(self, query: str, *, rows: int = 3) -> SourceLookup:
        """按书目信息（标题/作者等）检索。注意：对中文查询的召回很有限。"""
        query = (query or "").strip()
        lookup = SourceLookup(source=self.name, query=query)
        if not query:
            lookup.status = "error"
            lookup.error = "空查询"
            return lookup

        result = self.http.fetch(
            f"{self.base_url}/works",
            source=self.name,
            params={
                "query.bibliographic": query,
                "rows": rows,
                "mailto": self.mailto,
            },
        )
        if result.raw_ref:
            lookup.raw_refs.append(result.raw_ref)

        if result.failed:
            lookup.status = "error"
            lookup.error = result.error or f"HTTP {result.status_code}"
            return lookup
        if result.not_found:
            lookup.status = "not_found"
            return lookup

        payload = result.json() or {}
        items = ((payload.get("message") or {}).get("items")) or []
        for item in items:
            doi = item.get("DOI")
            lookup.records.append(
                self._record_from_message(item, doi or query, result.raw_ref, result.url)
            )
        if not lookup.records:
            lookup.status = "no_hit"
        return lookup

    # ------------------------------------------------------------------
    @staticmethod
    def _record_from_message(
        message: dict, doi: str, raw_ref: str | None, request_url: str
    ) -> SourceRecord:
        titles = message.get("title") or []
        title = strip_markup(titles[0]) if titles else ""

        authors: list[str] = []
        for author in message.get("author") or []:
            family = (author.get("family") or "").strip()
            given = (author.get("given") or "").strip()
            name = " ".join(part for part in (family, given) if part) or author.get("name") or ""
            if name:
                authors.append(name)

        year = None
        issued = message.get("issued") or {}
        parts = issued.get("date-parts") or []
        if parts and parts[0]:
            year = safe_year(parts[0][0])
        if year is None:
            year = safe_year(message.get("published-print") or message.get("published"))

        containers = message.get("container-title") or []
        abstract = strip_markup(message.get("abstract")) or None

        return SourceRecord(
            source=EvidenceSource.CROSSREF,
            url=message.get("URL") or f"https://doi.org/{doi}",
            level=EvidenceLevel.ABSTRACT if abstract else EvidenceLevel.METADATA_ONLY,
            title=title,
            doi=(message.get("DOI") or doi) or None,
            authors=tuple(authors),
            year=year,
            venue=containers[0] if containers else None,
            publisher=message.get("publisher"),
            abstract=abstract,
            query=doi,
            raw_ref=raw_ref,
            note=f"Crossref 注册记录（请求 {request_url}）",
        )
