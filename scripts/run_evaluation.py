"""命令行评测入口：在人工标注测试集上跑一遍并打印真实指标。

用法（在项目根目录）：
    # 下界基线：离线规则引擎 + 不检索（完全免费，先跑这个）
    .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --baseline b1 --all

    # 完整流程（会调用模型与外部检索；结果会缓存，重复跑不再付费）
    .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --baseline b2 --ids G01,G04,G07,G13,G16,G19
    .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --baseline b2 --categories 来源冲突 --limit 3

    # 把评测报告存成 JSON（可归档、可复现）
    .\\.venv\\Scripts\\python.exe scripts\\run_evaluation.py --baseline b2 --all --json out-eval.json

设计原则：**跑的就是生产线**（调用同一个 `run_detection`），
并且把样本量、标注者身份、失败案例一起打印出来，不做任何粉饰。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from verimind.core.config import Settings  # noqa: E402
from verimind.core.errors import VeriMindError  # noqa: E402
from verimind.evaluation.dataset import load_golden_set  # noqa: E402
from verimind.evaluation.runner import (  # noqa: E402
    BASELINE_B0,
    BASELINE_B1,
    BASELINE_RULES,
    EvaluationReport,
    offline_settings,
    run_evaluation,
)

LINE = "=" * 78


def _show(value: object, *, percent: bool = True) -> str:
    if value is None:
        return "—（分母为 0，不做估计）"
    if percent and isinstance(value, float):
        return f"{value * 100:.1f}%"
    return str(value)


def print_report(report: EvaluationReport) -> None:
    metrics = report.metrics
    print(LINE)
    print(f"评测基线：{report.baseline}")
    print(f"引擎    ：{report.engine_summary}")
    print(f"用例数  ：{report.case_count}（对齐成功 {metrics.get('matched_count')} 条"
          f"，未命中 {metrics.get('unmatched_count')} 条）")
    print(LINE)

    print("\n【主张级检测指标】（正类 = 标注为「确实有问题」）")
    print(f"  精确率 Precision        : {_show(metrics.get('detection_precision'))}")
    print(f"  召回率 Recall           : {_show(metrics.get('detection_recall'))}")
    print(f"  F1                      : {_show(metrics.get('detection_f1'))}")
    print(f"  误报率 FPR              : {_show(metrics.get('false_positive_rate'))}")
    print(f"  混淆矩阵 TP/FP/FN/TN    : {metrics.get('true_positive')}/{metrics.get('false_positive')}"
          f"/{metrics.get('false_negative')}/{metrics.get('true_negative')}")
    print(f"  六状态完全一致率        : {_show(metrics.get('status_accuracy'))}")
    print(f"  证据可追溯率            : {_show(metrics.get('evidence_traceability_rate'))}")

    print("\n【修正质量】")
    print(f"  期望修正的用例          : {metrics.get('correction_expected_cases')}")
    print(f"  修正成功率              : {_show(metrics.get('correction_success_rate'))}")
    print(f"  剩余错误率              : {_show(metrics.get('residual_error_rate'))}")
    print(f"  新增错误率              : {_show(metrics.get('new_error_rate'))}")
    print(f"  实际采用的修正总数      : {metrics.get('corrections_applied_total')}")

    print("\n【性能与成本】")
    print(f"  平均耗时                : {_show(metrics.get('avg_latency_ms'), percent=False)} ms")
    print(f"  平均模型调用次数/用例   : {_show(metrics.get('avg_llm_calls_per_case'), percent=False)}")
    print(f"  累计 token              : prompt {metrics.get('total_prompt_tokens')} + "
          f"completion {metrics.get('total_completion_tokens')}")
    print(f"  累计检索查询            : {metrics.get('total_retrieval_queries')}")

    print("\n【分类别】")
    for name, item in (metrics.get("by_category") or {}).items():
        rate = "—" if not item["matched"] else f"{item['correct'] / item['matched'] * 100:.0f}%"
        print(f"  {name:<16} 用例 {item['cases']}｜对齐 {item['matched']}｜状态一致 {item['correct']}（{rate}）")

    print("\n【逐条结果】")
    for row in report.rows:
        mark = "OK  " if row.gold_status == row.predicted_status else "DIFF"
        print(f"  [{mark}] {row.case_id} {row.category:<12} 标注={row.gold_status:<8} "
              f"预测={row.predicted_status or '未命中':<8} "
              f"证据 {len(row.evidence_ids)}｜修正 {row.corrections}/{row.corrections_applied}")

    if report.failures:
        print(f"\n【失败案例】共 {len(report.failures)} 条（如实列出，不隐藏）")
        for item in report.failures:
            print(f"  - {item['case_id']}（{item['kind']}）：{item['detail']}")
    else:
        print("\n【失败案例】本次没有")

    print("\n【实验说明与局限】")
    for caveat in report.caveats:
        print(f"  · {caveat}")
    print(LINE)


def main() -> int:
    parser = argparse.ArgumentParser(description="在人工标注测试集上运行评测")
    parser.add_argument(
        "--baseline",
        choices=["b0", "b1", "rules"],
        default="b1",
        help="b0=直接问模型（无外部证据）；b1=本项目方法（完整流程）；rules=离线规则引擎（免费下界）",
    )
    parser.add_argument("--all", action="store_true", help="跑全部用例")
    parser.add_argument("--ids", default="", help="指定用例编号，逗号分隔（如 G01,G04）")
    parser.add_argument("--categories", default="", help="按类别筛选，逗号分隔")
    parser.add_argument("--limit", type=int, default=None, help="最多跑多少条")
    parser.add_argument("--json", default="", help="把评测报告存为 JSON")
    args = parser.parse_args()

    cases = load_golden_set(PROJECT_ROOT / "examples" / "golden_set.jsonl")
    settings = Settings()
    if args.baseline == "b1":
        active, baseline = settings, BASELINE_B1
    elif args.baseline == "rules":
        active, baseline = offline_settings(settings), BASELINE_RULES
    else:
        active, baseline = settings, BASELINE_B0

    only_ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    only_categories = [item.strip() for item in args.categories.split(",") if item.strip()]

    print(f"\n>>> 基线：{baseline}")
    print(f">>> 引擎：{active.llm_provider}/{active.llm_model}"
          f"｜检索：{'开' if active.retrieve_enabled else '关'}"
          f"｜复核：{active.llm_reviewer_provider or '与主模型相同'}")
    if args.limit is None and not args.all and not only_ids and not only_categories:
        print(">>> 未指定范围：默认只跑前 3 条（避免误花钱）。要跑全部请加 --all。")
        limit = 3
    else:
        limit = args.limit

    def on_case(index: int, total: int, case) -> None:  # noqa: ANN001
        print(f"  [{index}/{total}] {case.case_id}（{case.category}）：{case.text[:42]}…")

    try:
        report = run_evaluation(
            settings=active,
            cases=cases,
            baseline=baseline,
            limit=limit,
            only_categories=only_categories or None,
            only_case_ids=only_ids or None,
            on_case=on_case,
        )
    except VeriMindError as exc:
        print(f"评测失败：{exc}")
        return 1

    print()
    print_report(report)

    if args.json:
        target = Path(args.json)
        target.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"评测报告已写入：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
