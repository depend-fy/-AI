"""把修正应用到原文，得到"修正后全文"。

这是纯函数、零成本、完全可测的一步——正因为如此，它承担了一个重要的职责：
**只有复核通过的修正才会被应用**；复核没通过的一条都不改，
否则"修正"本身就成了新的错误来源（这正是本项目想防的事）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verimind.core.schemas import Correction


@dataclass(frozen=True)
class ApplyOutcome:
    """应用结果。"""

    corrected_text: str
    applied_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def apply_corrections(
    text: str,
    corrections: list[Correction],
    *,
    only: set[str] | None = None,
) -> ApplyOutcome:
    """按区间从后往前替换文本。

    安全措施（每一条都会写进 notes，不会静默处理）：
        · 区间与原文对不上 → 跳过（防止把改动落在错误的位置）；
        · 两条修正的区间重叠 → 只应用靠后的那条，另一条跳过并说明；
        · 越界区间 → 跳过。
    """
    candidates = [
        item for item in corrections if only is None or item.correction_id in only
    ]
    notes: list[str] = []
    skipped: list[str] = []

    valid: list[Correction] = []
    for item in candidates:
        span = item.original_span
        if span.end > len(text) or span.start < 0 or span.start >= span.end:
            notes.append(f"{item.correction_id} 的原文区间越界，未应用。")
            skipped.append(item.correction_id)
            continue
        if text[span.start : span.end] != span.text:
            notes.append(
                f"{item.correction_id} 的原文区间与当前文本不一致（可能被其它修正影响），未应用。"
            )
            skipped.append(item.correction_id)
            continue
        valid.append(item)

    valid.sort(key=lambda item: item.original_span.start, reverse=True)

    applied: list[str] = []
    boundary = len(text) + 1  # 上一个（更靠后的）已应用修正的起点
    result = text
    for item in valid:
        if item.original_span.end > boundary:
            notes.append(
                f"{item.correction_id} 与另一条已应用的修正范围重叠，未应用（避免相互覆盖）。"
            )
            skipped.append(item.correction_id)
            continue
        result = (
            result[: item.original_span.start]
            + item.proposed_text
            + result[item.original_span.end :]
        )
        applied.append(item.correction_id)
        boundary = item.original_span.start

    return ApplyOutcome(
        corrected_text=result,
        applied_ids=list(reversed(applied)),  # 按原文顺序返回，便于阅读
        skipped_ids=skipped,
        notes=notes,
    )
