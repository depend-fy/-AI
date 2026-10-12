"""检索结果的公共解析工具，以及"未变成证据之前"的中间结构。

为什么要有个 `SourceRecord` 中间层？
    各个来源返回的字段名五花八门（Crossref 叫 container-title，OpenAlex 叫
    primary_location.source.display_name，arXiv 是 Atom 的 <summary>）。
    解析层只负责把它们**统一成一种结构**，编号、关联主张、生成 Evidence 由路由层统一做。
    这样"编号连续唯一"和"证据必须带 URL"这类约束只在一个地方实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from verimind.core.enums import EvidenceLevel, EvidenceSource

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")


def normalize_whitespace(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").replace("\n", " ")).strip()


def strip_markup(text: str | None) -> str:
    """去掉 JATS/HTML 标签（Crossref 的摘要是 JATS XML）。"""
    return normalize_whitespace(_TAG_RE.sub(" ", text or ""))


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex 的摘要是"倒排索引"，这里还原成正常顺序的文本。"""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        for index in indexes or []:
            positions.append((int(index), word))
    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    return normalize_whitespace(" ".join(word for _, word in positions))


def similarity(first: str | None, second: str | None) -> float:
    """两个标题的相似度（0~1），用于判断"查到的这篇是不是原文说的那篇"。"""
    left = normalize_whitespace(first).lower()
    right = normalize_whitespace(second).lower()
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def safe_year(value: object) -> int | None:
    """尽力从各种形态里取出年份。"""
    if isinstance(value, int):
        return value if 1000 <= value <= 2999 else None
    if isinstance(value, str):
        match = re.search(r"(1[5-9]\d{2}|20\d{2})", value)
        if match:
            return int(match.group(1))
    return None


@dataclass(frozen=True)
class SourceRecord:
    """从某个来源解析出来的一条文献记录（尚未编号、尚未关联主张）。"""

    source: EvidenceSource
    url: str
    level: EvidenceLevel = EvidenceLevel.METADATA_ONLY
    title: str = ""
    doi: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    abstract: str | None = None
    query: str = ""
    raw_ref: str | None = None
    found: bool = True
    note: str = ""

    @property
    def display(self) -> str:
        if not self.found:
            return f"未找到：{self.query}"
        if self.title:
            return f"{self.title}（{self.year}）" if self.year else self.title
        return self.doi or self.url

    @property
    def authors_text(self) -> str:
        return "、".join(self.authors)


@dataclass
class SourceLookup:
    """一个来源的一次查询结果。"""

    source: EvidenceSource
    query: str
    records: list[SourceRecord] = field(default_factory=list)
    status: str = "ok"  # ok / no_hit / not_found / error
    error: str | None = None
    raw_refs: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return bool(self.records) and all(record.found for record in self.records)
