"""OpenAlex：覆盖广、免密钥，而且**常常带摘要**（倒排索引形式）。

它在项目里的分工：
    · 与 Crossref 互为交叉验证（"来源冲突"需要两个以上来源）；
    · 提供摘要 → 让"内容结论是否被支持"这一层面的判断成为可能；
    · 按标题/关键词检索（对英文条目的召回明显好于中文）。
"""

from __future__ import annotations

from urllib.parse import quote

from verimind.core.enums import EvidenceLevel, EvidenceSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.parsing import (
    SourceLookup,
    SourceRecord,
    normalize_whitespace,
    reconstruct_abstract,
    safe_year,
)

OPENALEX_API = "https://api.openalex.org"


class OpenAlexSource:
    name = EvidenceSource.OPENALEX
    base_url = OPENALEX_API

    def __init__(self, http: RetrievalHttpClient, *, mailto: str = "") -> None:
        self.http = http
        self.mailto = mailto

    # ------------------------------------------------------------------
    def lookup_doi(self, doi: str) -> SourceLookup:
        """按 DOI 精确查询（OpenAlex 接受把 DOI 网址直接拼在 /works/ 后面）。"""
        doi = (doi or "").strip()
        lookup = SourceLookup(source=self.name, query=doi)
        if not doi:
            lookup.status = "error"
            lookup.error = "空 DOI"
            return lookup

        path = f"/works/{quote(f'https://doi.org/{doi}', safe=':/')}"
        result = self.http.fetch(f"{self.base_url}{path}", source=self.name, params={"mailto": self.mailto})
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
                    note="OpenAlex 返回 404：未收录该 DOI 对应的文献。",
                )
            )
            return lookup

        payload = result.json() or {}
        record = self._record_from_work(payload, doi, result.raw_ref)
        if record is None:
            lookup.status = "no_hit"
            return lookup
        lookup.records.append(record)
        return lookup

    # ------------------------------------------------------------------
    def search(self, query: str, *, per_page: int = 3) -> SourceLookup:
        """按关键词检索（中文查询的召回通常很差，这是数据源本身的限制）。"""
        query = normalize_whitespace(query)
        lookup = SourceLookup(source=self.name, query=query)
        if not query:
            lookup.status = "error"
            lookup.error = "空查询"
            return lookup

        result = self.http.fetch(
            f"{self.base_url}/works",
            source=self.name,
            params={"search": query, "per-page": per_page, "mailto": self.mailto},
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
        for work in payload.get("results") or []:
            record = self._record_from_work(work, query, result.raw_ref)
            if record is not None:
                lookup.records.append(record)
        if not lookup.records:
            lookup.status = "no_hit"
        return lookup

    # ------------------------------------------------------------------
    @staticmethod
    def _record_from_work(work: dict, query: str, raw_ref: str | None) -> SourceRecord | None:
        if not work:
            return None
        title = normalize_whitespace(work.get("display_name") or work.get("title"))
        doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        authors: list[str] = []
        for authorship in work.get("authorships") or []:
            name = ((authorship.get("author") or {}).get("display_name")) or ""
            if name:
                authors.append(name)

        location = work.get("primary_location") or {}
        source_info = location.get("source") or {}
        url = (
            location.get("landing_page_url")
            or (f"https://doi.org/{doi}" if doi else None)
            or work.get("id")
            or ""
        )

        return SourceRecord(
            source=EvidenceSource.OPENALEX,
            url=str(url),
            level=EvidenceLevel.ABSTRACT if abstract else EvidenceLevel.METADATA_ONLY,
            title=title,
            doi=doi,
            authors=tuple(authors),
            year=safe_year(work.get("publication_year")),
            venue=source_info.get("display_name"),
            publisher=source_info.get("host_organization_name"),
            abstract=abstract,
            query=query,
            raw_ref=raw_ref,
            note="OpenAlex 收录记录",
        )
