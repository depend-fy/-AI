"""指标计算（纯函数，可单元测试）。

两条自我约束：
    1. **分母为 0 时一律返回 None**，界面显示「—」。绝不用 0 或者 1.0 去凑一个好看的数。
    2. 每个指标都写清定义。比赛答辩时最怕的就是"这个数字怎么算的"答不上来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseOutcome:
    """一条用例的评测结果（预测 vs 标注）。"""

    case_id: str
    category: str
    gold_status: str
    predicted_status: str | None
    matched: bool
    gold_problem_types: tuple[str, ...] = ()
    predicted_problem_types: tuple[str, ...] = ()
    evidence_count: int = 0
    traceable_evidence_count: int = 0
    corrections_total: int = 0
    corrections_applied: int = 0
    correction_contains_ok: bool = False
    correction_excludes_ok: bool = False
    new_fact_tokens: tuple[str, ...] = ()
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    retrieval_queries: int = 0

    @property
    def gold_is_problem(self) -> bool:
        return self.gold_status in ("存在矛盾", "部分支持", "来源冲突")

    @property
    def predicted_is_problem(self) -> bool:
        return (self.predicted_status or "") in ("存在矛盾", "部分支持", "来源冲突")

    @property
    def status_correct(self) -> bool:
        return self.matched and self.gold_status == self.predicted_status


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def compute_metrics(outcomes: list[CaseOutcome], needs_correction_ids: set[str]) -> dict[str, Any]:
    """计算全部指标。`needs_correction_ids` 是"标注里期望被修正"的用例编号集合。"""
    matched = [item for item in outcomes if item.matched]
    tp = sum(1 for item in matched if item.gold_is_problem and item.predicted_is_problem)
    fp = sum(1 for item in matched if not item.gold_is_problem and item.predicted_is_problem)
    fn = sum(1 for item in matched if item.gold_is_problem and not item.predicted_is_problem)
    tn = sum(1 for item in matched if not item.gold_is_problem and not item.predicted_is_problem)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else None
    )

    correction_cases = [item for item in matched if item.case_id in needs_correction_ids]
    success = [
        item for item in correction_cases if item.correction_contains_ok and item.correction_excludes_ok
    ]
    residual = [item for item in correction_cases if not item.correction_excludes_ok]
    new_error = [item for item in matched if item.new_fact_tokens]

    by_category: dict[str, dict[str, int]] = {}
    for item in outcomes:
        bucket = by_category.setdefault(item.category, {"cases": 0, "matched": 0, "correct": 0})
        bucket["cases"] += 1
        if item.matched:
            bucket["matched"] += 1
        if item.status_correct:
            bucket["correct"] += 1

    return {
        # 样本规模先说清楚
        "case_count": len(outcomes),
        "matched_count": len(matched),
        "unmatched_count": len(outcomes) - len(matched),
        "by_category": by_category,
        # 主张级检测（正类 = 标注为"确实有问题"）
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
        # 六状态判定完全一致的占比
        "status_accuracy": _ratio(sum(1 for item in matched if item.status_correct), len(matched)),
        # 证据可追溯率：至少有一条可追溯证据（外部链接或原文内部定位）的主张占比
        "evidence_traceability_rate": _ratio(
            sum(1 for item in matched if item.traceable_evidence_count > 0), len(matched)
        ),
        "avg_evidence_per_case": (
            round(sum(item.evidence_count for item in matched) / len(matched), 2) if matched else None
        ),
        # 修正质量
        "correction_expected_cases": len(correction_cases),
        "correction_success_rate": _ratio(len(success), len(correction_cases)),
        "residual_error_rate": _ratio(len(residual), len(correction_cases)),
        "new_error_rate": _ratio(len(new_error), len(matched)),
        "corrections_applied_total": sum(item.corrections_applied for item in matched),
        # 成本与性能
        "avg_latency_ms": (
            round(sum(item.latency_ms for item in matched) / len(matched)) if matched else None
        ),
        "total_llm_calls": sum(item.llm_calls for item in matched),
        "total_prompt_tokens": sum(item.prompt_tokens for item in matched),
        "total_completion_tokens": sum(item.completion_tokens for item in matched),
        "total_retrieval_queries": sum(item.retrieval_queries for item in matched),
        "avg_llm_calls_per_case": (
            round(sum(item.llm_calls for item in matched) / len(matched), 2) if matched else None
        ),
    }


def summarize_failures(outcomes: list[CaseOutcome], limit: int = 50) -> list[dict[str, Any]]:
    """列出失败案例（错误分类或未命中）——**如实列出，不隐藏**。"""
    failures: list[dict[str, Any]] = []
    for item in outcomes:
        if not item.matched:
            failures.append(
                {
                    "case_id": item.case_id,
                    "kind": "未命中主张",
                    "detail": "预测的主张里找不到与标注片段对齐的那一条（拆解环节漏检或定位失败）",
                    "gold": item.gold_status,
                    "predicted": item.predicted_status,
                }
            )
        elif not item.status_correct:
            failures.append(
                {
                    "case_id": item.case_id,
                    "kind": "状态不一致",
                    "detail": f"标注={item.gold_status}，预测={item.predicted_status}",
                    "gold": item.gold_status,
                    "predicted": item.predicted_status,
                }
            )
        if item.new_fact_tokens:
            failures.append(
                {
                    "case_id": item.case_id,
                    "kind": "修正引入新事实",
                    "detail": "修正后出现了原文与证据里都没有的内容：" + "、".join(item.new_fact_tokens),
                    "gold": item.gold_status,
                    "predicted": item.predicted_status,
                }
            )
    return failures[:limit]


__all__ = ["CaseOutcome", "compute_metrics", "summarize_failures"]
