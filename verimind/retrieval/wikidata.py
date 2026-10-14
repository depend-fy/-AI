"""Wikidata：人物、机构、时间等**结构化事实**的免密钥来源。

它在项目里的分工（与 Crossref/OpenAlex 互补，不重叠）：
    · Crossref / OpenAlex 管"学术文献"：DOI 是否存在、年份期刊、论文摘要；
    · Wikidata 管"实体事实"：某人是谁、生于何年、任职于哪个机构、获得过什么奖项。
      —— 这正是原文里「时间、人物或机构信息错误」这一类主张此前查不到的地方。

证据级别：**仅元数据**。Wikidata 给出的是结构化声明（P569 出生日期等），
不是可引用的原文段落，因此**不允许**据此判断"内容结论是否被支持"（见 JudgedAspect 约束）。
每条声明都带来源链接（https://www.wikidata.org/wiki/Q...），可逐条点开核对。
"""

from __future__ import annotations

from verimind.core.enums import EvidenceLevel, EvidenceSource
from verimind.retrieval.http import RetrievalHttpClient
from verimind.retrieval.parsing import (
    SourceLookup,
    SourceRecord,
    normalize_whitespace,
    safe_year,
)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

#: 只取这几类"可核验的实体事实"，避免把整个条目搬进来（也避免越界判断）
_PROPERTY_LABELS: dict[str, str] = {
    "P31": "所属类别",
    "P569": "出生日期",
    "P570": "逝世日期",
    "P106": "职业",
    "P108": "任职机构",
    "P166": "获得奖项",
    "P39": "担任职务",
    "P17": "所属国家",
    "P571": "成立时间",
}

#: 值是"另一个实体"的属性（需要再查一次标签才能读）
_ENTITY_VALUED = {"P31", "P106", "P108", "P166", "P39", "P17"}


class WikidataSource:
    name = EvidenceSource.WIKIDATA
    base_url = WIKIDATA_API

    def __init__(self, http: RetrievalHttpClient, *, mailto: str = "") -> None:
        self.http = http
        self.mailto = mailto

    # ------------------------------------------------------------------
    def search_entity(self, query: str, *, language: str = "zh", limit: int = 3) -> SourceLookup:
        """按关键词找实体，并取出该实体的关键结构化事实。"""
        query = normalize_whitespace(query)
        lookup = SourceLookup(source=self.name, query=query)
        if not query:
            lookup.status = "error"
            lookup.error = "空查询"
            return lookup

        search = self.http.fetch(
            self.base_url,
            source=self.name,
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": language,
                "uselang": language,
                "format": "json",
                "limit": limit,
            },
        )
        if search.raw_ref:
            lookup.raw_refs.append(search.raw_ref)
        if search.failed:
            lookup.status = "error"
            lookup.error = search.error or f"HTTP {search.status_code}"
            return lookup
        if search.not_found:
            lookup.status = "not_found"
            return lookup

        payload = search.json() or {}
        candidates = payload.get("search") or []
        if not candidates:
            lookup.status = "no_hit"
            return lookup

        for candidate in candidates[:limit]:
            entity_id = str(candidate.get("id") or "")
            if not entity_id:
                continue
            facts, label_map, raw_ref = self._entity_facts(entity_id, language=language)
            if raw_ref:
                lookup.raw_refs.append(raw_ref)
            label = normalize_whitespace(candidate.get("label")) or entity_id
            description = normalize_whitespace(candidate.get("description"))
            lookup.records.append(
                SourceRecord(
                    source=self.name,
                    url=f"https://www.wikidata.org/wiki/{entity_id}",
                    level=EvidenceLevel.METADATA_ONLY,
                    title=label,
                    abstract=self._render_facts(label, description, facts, label_map),
                    query=query,
                    raw_ref=raw_ref,
                    note=f"Wikidata 实体 {entity_id}",
                )
            )
        if not lookup.records:
            lookup.status = "no_hit"
        return lookup

    # ------------------------------------------------------------------
    def _entity_facts(self, entity_id: str, *, language: str) -> tuple[dict[str, list[str]], dict[str, str], str | None]:
        """取实体的关键声明；对"实体值"的属性再批量查一次标签。"""
        result = self.http.fetch(
            self.base_url,
            source=self.name,
            params={
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "claims|labels",
                "languages": f"{language}|en",
                "format": "json",
            },
        )
        payload = result.json() or {}
        entity = (payload.get("entities") or {}).get(entity_id) or {}
        claims = entity.get("claims") or {}

        facts: dict[str, list[str]] = {}
        referenced: list[str] = []
        for prop, label in _PROPERTY_LABELS.items():
            values: list[str] = []
            for statement in claims.get(prop) or []:
                snak = statement.get("mainsnak") or {}
                value = ((snak.get("datavalue") or {}).get("value"))
                if value is None:
                    continue
                if prop in _ENTITY_VALUED and isinstance(value, dict):
                    referenced_id = str(value.get("id") or "")
                    if referenced_id:
                        values.append(referenced_id)
                        referenced.append(referenced_id)
                elif isinstance(value, dict) and "time" in value:
                    year = safe_year(str(value.get("time")))
                    if year:
                        values.append(f"{year} 年")
                elif isinstance(value, str):
                    values.append(normalize_whitespace(value))
            if values:
                facts[label] = values[:4]

        label_map: dict[str, str] = {}
        if referenced:
            unique = list(dict.fromkeys(referenced))[:8]
            label_result = self.http.fetch(
                self.base_url,
                source=self.name,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(unique),
                    "props": "labels",
                    "languages": f"{language}|en",
                    "format": "json",
                },
            )
            label_payload = label_result.json() or {}
            for referenced_id, item in (label_payload.get("entities") or {}).items():
                labels = item.get("labels") or {}
                text = (
                    (labels.get(language) or {}).get("value")
                    or (labels.get("en") or {}).get("value")
                    or referenced_id
                )
                label_map[referenced_id] = normalize_whitespace(text)
        return facts, label_map, result.raw_ref

    # ------------------------------------------------------------------
    @staticmethod
    def _render_facts(
        label: str, description: str, facts: dict[str, list[str]], label_map: dict[str, str]
    ) -> str:
        """把结构化事实拼成一句可读、可核对的文本（作为报告里的证据片段）。"""
        parts: list[str] = []
        if description:
            parts.append(f"简介：{description}")
        for fact_label, values in facts.items():
            rendered = [label_map.get(value, value) for value in values]
            parts.append(f"{fact_label}：{'、'.join(rendered)}")
        if not parts:
            return f"{label}：Wikidata 收录该实体，但未取到可核验的关键声明。"
        return f"{label}｜" + "；".join(parts)


__all__ = ["WikidataSource", "WIKIDATA_API"]
