"""Wikipedia（中文 / 英文）：非学术事实的**可引用文本**来源。

它在项目里的分工：
    · Wikidata 给"结构化事实"（谁、哪年、任职于哪）；
    · Wikipedia 给"可引用的原文段落"（导语摘要），因此**可以**支撑内容层面的判断；
    · 两者互补：Wikidata 适合核对字段，Wikipedia 适合核对表述。

为什么用导语（intro extract）而不是全文：
    一次请求即可取回，长度可控，而且导语本身就是条目对主题的概括性陈述——
    正好对应"这条主张有没有公开资料支持"这个问题。

边界（写进报告与界面）：
    · Wikipedia 是**百科**，不是一次文献；它的结论不能当作学术证据；
    · 因此本来源只用于「一般事实类 / 归属类」主张，不用于论文内容层面的判断。
"""

from __future__ import annotations

import re

from verimind.core.enums import EvidenceLevel, EvidenceSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.parsing import SourceLookup, SourceRecord, normalize_whitespace

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def looks_chinese(text: str) -> bool:
    """粗略判断一段文本是否以中文为主（决定查中文还是英文维基）。"""
    if not text:
        return False
    cjk = len(_CJK_RE.findall(text))
    return cjk >= max(4, len(text) * 0.15)


class WikipediaSource:
    name = EvidenceSource.WIKIPEDIA
    #: 语言子域由查询语言决定
    base_template = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(self, http: RetrievalHttpClient, *, mailto: str = "") -> None:
        self.http = http
        self.mailto = mailto

    def search_summary(self, query: str, *, lang: str | None = None, limit: int = 2) -> SourceLookup:
        """按关键词检索条目，并取回导语摘要（一次请求）。"""
        query = normalize_whitespace(query)
        lookup = SourceLookup(source=self.name, query=query)
        if not query:
            lookup.status = "error"
            lookup.error = "空查询"
            return lookup

        language = lang or ("zh" if looks_chinese(query) else "en")
        result = self.http.fetch(
            self.base_template.format(lang=language),
            source=self.name,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
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
        pages = ((payload.get("query") or {}).get("pages")) or {}
        # API 用 pageid 作键；按 index 排序保持检索相关度顺序
        ordered = sorted(
            pages.values(),
            key=lambda item: item.get("index", 999),
        )
        for page in ordered[:limit]:
            title = normalize_whitespace(page.get("title"))
            extract = normalize_whitespace(page.get("extract"))
            if not title or not extract:
                continue
            page_id = page.get("pageid")
            lookup.records.append(
                SourceRecord(
                    source=self.name,
                    url=f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    level=EvidenceLevel.ABSTRACT,
                    title=title,
                    abstract=extract[:900],
                    query=query,
                    raw_ref=result.raw_ref,
                    note=f"{language}.wikipedia.org 条目（pageid {page_id}）",
                )
            )
        if not lookup.records:
            lookup.status = "no_hit"
        return lookup


__all__ = ["WikipediaSource", "looks_chinese"]
