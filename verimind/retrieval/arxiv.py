"""arXiv：预印本检索（免密钥，Atom XML）。

它在项目里的分工：
    · 当原文只提到"某篇论文/某项研究"而**没有 DOI** 时，用标题去碰一碰；
    · 命中时能拿到摘要 → 可以进入"内容是否被支持"这一层面的判断。

注意：arXiv 只收录预印本，命中不等于"正式发表版本"；也查不到中文文献。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from verimind.core.enums import EvidenceLevel, EvidenceSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.parsing import (
    SourceLookup,
    SourceRecord,
    normalize_whitespace,
    safe_year,
)

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivSource:
    name = EvidenceSource.ARXIV
    base_url = ARXIV_API

    def __init__(self, http: RetrievalHttpClient, *, mailto: str = "") -> None:
        self.http = http
        self.mailto = mailto

    # ------------------------------------------------------------------
    def search_title(self, title: str, *, max_results: int = 3) -> SourceLookup:
        query = self._clean_query(title)
        lookup = SourceLookup(source=self.name, query=query)
        if not query:
            lookup.status = "error"
            lookup.error = "空查询"
            return lookup

        result = self.http.fetch(
            self.base_url,
            source=self.name,
            params={
                "search_query": f'ti:"{query}"',
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
            suffix=".xml",
        )
        if result.raw_ref:
            lookup.raw_refs.append(result.raw_ref)
        if result.failed:
            lookup.status = "error"
            lookup.error = result.error or f"HTTP {result.status_code}"
            return lookup
        if not result.ok:
            lookup.status = "not_found" if result.not_found else "error"
            return lookup

        try:
            root = ET.fromstring(result.text)
        except ET.ParseError as exc:
            lookup.status = "error"
            lookup.error = f"arXiv 返回的 XML 无法解析：{exc}"
            return lookup

        for entry in root.findall(f"{_ATOM}entry"):
            record = self._record_from_entry(entry, query, result.raw_ref)
            if record is not None:
                lookup.records.append(record)
        if not lookup.records:
            lookup.status = "no_hit"
        return lookup

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_query(text: str) -> str:
        """arXiv 的查询语法对引号等字符敏感，这里做保守清洗。"""
        cleaned = normalize_whitespace(text)
        for char in ('"', ":", "(", ")", "[", "]", "{", "}"):
            cleaned = cleaned.replace(char, " ")
        return normalize_whitespace(cleaned)[:200]

    @staticmethod
    def _record_from_entry(entry: ET.Element, query: str, raw_ref: str | None) -> SourceRecord | None:
        title = normalize_whitespace((entry.findtext(f"{_ATOM}title") or ""))
        if not title:
            return None
        abstract = normalize_whitespace(entry.findtext(f"{_ATOM}summary") or "") or None
        published = entry.findtext(f"{_ATOM}published") or entry.findtext(f"{_ATOM}updated") or ""
        url = normalize_whitespace(entry.findtext(f"{_ATOM}id") or "")

        authors = tuple(
            normalize_whitespace(author.findtext(f"{_ATOM}name") or "")
            for author in entry.findall(f"{_ATOM}author")
            if normalize_whitespace(author.findtext(f"{_ATOM}name") or "")
        )

        doi_element = entry.find("{http://arxiv.org/schemas/atom}doi")
        doi = normalize_whitespace(doi_element.text) if doi_element is not None else None

        return SourceRecord(
            source=EvidenceSource.ARXIV,
            url=url or f"https://arxiv.org/abs/{query}",
            level=EvidenceLevel.ABSTRACT if abstract else EvidenceLevel.METADATA_ONLY,
            title=title,
            doi=doi or None,
            authors=authors,
            year=safe_year(published),
            venue="arXiv（预印本）",
            publisher="arXiv",
            abstract=abstract,
            query=query,
            raw_ref=raw_ref,
            note="arXiv 预印本记录（未经同行评审的版本，不等同于正式发表版）",
        )
