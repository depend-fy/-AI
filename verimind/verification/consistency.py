"""文内一致性检查：不需要任何外部证据就能发现的矛盾。

这是阶段 2 唯一能给出「来源冲突」结论的来源——因为冲突的双方都在用户自己的原文里，
证据是**原文本身**（EvidenceSource.INTERNAL），既不需要联网，也不需要编造来源。

阶段 2 实现了两类最常见的文内矛盾：
    · 同一篇文献（同一 DOI）在原文中被写成两个不同的年份；
    · 同一个指标（如"准确率"）在原文中出现了两个不同的数值。

局限（必须写清楚）：
    这里用的是"同一句话里同时出现"的启发式匹配。如果指标词和数值跨句出现，
    或者一句话里混了多个指标，可能漏检或误配。真正的语义级一致性检查要等阶段 3 的大模型复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verimind.claims.preprocess import DocumentAnalysis, TextMatch, analyze_text
from verimind.core.enums import ClaimType
from verimind.core.schemas import Claim, Span

#: 指标词与数值之间最多隔多少字符还算同一个论断
_METRIC_VALUE_MAX_GAP = 24

#: 各类冲突只挂到"确实在谈这件事"的主张类型上。
#: 为什么必须这样限制：大模型给出的引用片段往往更宽（可能把整句都引上），
#: 如果只按"区间重叠"归属，就会出现"期刊类主张被标成年份冲突"这种过度归因——
#: 那等于给用户安了一个它没说过的错。
_CONFLICT_CLAIM_TYPES: dict[str, frozenset[ClaimType]] = {
    "doi_year": frozenset({ClaimType.METADATA_YEAR}),
    "metric_value": frozenset({ClaimType.QUANTITATIVE}),
}


@dataclass(frozen=True)
class InternalConflict:
    """一处原文内部的矛盾。"""

    kind: str  # "doi_year"（同一文献不同年份）/ "metric_value"（同一指标不同数值）
    label: str  # 冲突对象，例如 "DOI 10.1234/abc" 或 "准确率"
    fragments: list[Span]  # 互相矛盾的原文片段（≥2 个）
    detail: str  # 面向用户的说明
    claim_ids: list[str] = field(default_factory=list)

    @property
    def rule_id(self) -> str:
        return "I1" if self.kind == "doi_year" else "I2"


def _claims_touching(
    claims: list[Claim], spans: list[Span], allowed_types: frozenset[ClaimType] | None = None
) -> list[str]:
    """找出与给定片段有重叠、且类型匹配的主张 ID（用于把冲突挂到具体主张上）。"""
    ids: list[str] = []
    for claim in claims:
        if allowed_types is not None and claim.claim_type not in allowed_types:
            continue
        span = claim.source_span
        for other in spans:
            if span.start < other.end and other.start < span.end:
                ids.append(claim.claim_id)
                break
    return ids


def _doi_year_conflicts(analysis: DocumentAnalysis, claims: list[Claim]) -> list[InternalConflict]:
    """同一 DOI 被描述为不同年份。"""
    # doi -> {年份 -> (年份片段, DOI 片段)}
    seen: dict[str, dict[int, tuple[Span, Span]]] = {}
    for sentence in analysis.sentences:
        signals = analysis.for_span(sentence)
        if not signals.dois or not signals.years:
            continue
        for doi in signals.dois:
            for year in signals.years:
                if not isinstance(year.value, int):
                    continue
                seen.setdefault(doi.text, {}).setdefault(int(year.value), (year.span, doi.span))

    conflicts: list[InternalConflict] = []
    for doi, years in seen.items():
        if len(years) < 2:
            continue
        ordered = sorted(years.items())  # 按年份升序，保证结果稳定
        year_a, (year_span_a, _doi_span) = ordered[0]
        year_b, (year_span_b, _) = ordered[1]
        fragments = [year_span_a, year_span_b]
        conflicts.append(
            InternalConflict(
                kind="doi_year",
                label=f"DOI {doi}",
                fragments=fragments,
                detail=(
                    f"原文把同一个 DOI（{doi}）描述的年份写成了两个不同的值："
                    f"{year_a} 年 与 {year_b} 年。同一篇文献不可能有两个发表年份，"
                    "至少有一处是错的（具体哪个正确需要核对文献本身）。"
                ),
                # 只把"年份类主张"挂到这个冲突上（详见 _CONFLICT_CLAIM_TYPES 的说明）。
                claim_ids=_claims_touching(
                    claims, fragments, _CONFLICT_CLAIM_TYPES["doi_year"]
                ),
            )
        )
    return conflicts


def _pair_values_with_metrics(signals) -> dict[str, list[TextMatch]]:
    """把每个数值配到"离它最近的那个指标词"上。

    为什么不能简单地"一句话里所有指标 × 所有数值"两两组合？
    因为那样 "准确率提高到 95%，误差降低到 3%" 会被判成
    「准确率出现了两个数值 95 与 3」——一个彻头彻尾的**误报**。
    误报是本项目最不能接受的错误类型，所以必须按位置就近配对。
    """
    metrics = sorted(signals.metrics, key=lambda m: m.span.start)
    values = sorted(
        [*signals.percents, *signals.strong_quantities], key=lambda v: v.span.start
    )
    pairing: dict[str, list[TextMatch]] = {}
    for value in values:
        if value.value is None:
            continue
        # 优先取"数值之前最近的那个指标"；如果指标写在数值后面（如"95% 的准确率"），再取后面最近的
        preceding = [metric for metric in metrics if metric.span.end <= value.span.start]
        following = [metric for metric in metrics if metric.span.start >= value.span.end]
        candidates = preceding[-1:] or following[:1]
        if not candidates:
            continue
        metric = candidates[0]
        gap = (
            value.span.start - metric.span.end
            if metric.span.end <= value.span.start
            else metric.span.start - value.span.end
        )
        if gap > _METRIC_VALUE_MAX_GAP:
            continue
        pairing.setdefault(metric.text, []).append(value)
    return pairing


def _metric_value_conflicts(
    analysis: DocumentAnalysis, claims: list[Claim]
) -> list[InternalConflict]:
    """同一指标出现两个不同数值。"""
    # 指标名 -> {数值 -> 数值片段}
    seen: dict[str, dict[float, Span]] = {}
    for sentence in analysis.sentences:
        signals = analysis.for_span(sentence)
        if not signals.metrics:
            continue
        for metric, values in _pair_values_with_metrics(signals).items():
            for value in values:
                if value.value is None:
                    continue
                seen.setdefault(metric, {}).setdefault(float(value.value), value.span)

    conflicts: list[InternalConflict] = []
    for metric, values in seen.items():
        if len(values) < 2:
            continue
        ordered = sorted(values.items())
        value_a, span_a = ordered[0]
        value_b, span_b = ordered[1]
        fragments = [span_a, span_b]
        conflicts.append(
            InternalConflict(
                kind="metric_value",
                label=f"指标「{metric}」",
                fragments=fragments,
                detail=(
                    f"原文对同一个指标「{metric}」给出了两个不同的数值："
                    f"{_fmt(value_a)} 与 {_fmt(value_b)}。同一指标在同一段文字里应当一致，"
                    "需要确认哪个数值是正确的。"
                ),
                claim_ids=_claims_touching(
                    claims, fragments, _CONFLICT_CLAIM_TYPES["metric_value"]
                ),
            )
        )
    return conflicts


def _fmt(value: float) -> str:
    """数值展示：整数不带小数点。"""
    return str(int(value)) if float(value).is_integer() else str(value)


def find_internal_conflicts(
    text: str,
    claims: list[Claim],
    *,
    analysis: DocumentAnalysis | None = None,
) -> list[InternalConflict]:
    """找出原文内部的全部矛盾（当前覆盖 DOI-年份 与 指标-数值 两类）。"""
    analysis = analysis or analyze_text(text)
    conflicts: list[InternalConflict] = [
        *_doi_year_conflicts(analysis, claims),
        *_metric_value_conflicts(analysis, claims),
    ]
    # 稳定排序，保证同一份输入每次得到同样的报告
    conflicts.sort(key=lambda c: (c.kind, c.fragments[0].start if c.fragments else 0))
    return conflicts


def conflicts_by_claim(conflicts: list[InternalConflict]) -> dict[str, list[InternalConflict]]:
    """把冲突按主张 ID 归组，方便逐条裁决时快速查到。"""
    mapping: dict[str, list[InternalConflict]] = {}
    for conflict in conflicts:
        for claim_id in conflict.claim_ids:
            mapping.setdefault(claim_id, []).append(conflict)
    return mapping


__all__ = [
    "InternalConflict",
    "conflicts_by_claim",
    "find_internal_conflicts",
]
