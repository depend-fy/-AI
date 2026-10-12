"""命令行演示入口：不开网页也能完整跑一次检测。

用途：
    · 开发时快速验证流水线（比开 Streamlit 快得多）；
    · 阶段 6 的现场演示兜底（断网也能跑，因为阶段 2 完全离线）；
    · 生成报告 JSON 供评测评测或人工核对。

用法（在项目根目录）：
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py                      # 跑第一条示例
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py --list               # 列出所有示例
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py --id internal_conflict
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py --text "待核验文本"
    .\\.venv\\Scripts\\python.exe scripts\\run_demo.py --id fake_doi --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 保证中文在 Windows 终端里不乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - 极少数环境不支持
    pass

from verimind.core.config import Settings  # noqa: E402
from verimind.core.errors import VeriMindError  # noqa: E402
from verimind.core.pipeline import run_detection  # noqa: E402
from verimind.core.schemas import DetectionReport  # noqa: E402

SAMPLES_PATH = PROJECT_ROOT / "examples" / "sample_inputs.jsonl"


def load_samples() -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    if not SAMPLES_PATH.exists():
        return samples
    for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def print_report(report: DetectionReport) -> None:
    line = "=" * 72
    print(line)
    print(f"运行 ID：{report.run_id}")
    print(f"运行模式：{report.run_mode.value}")
    print(
        f"主张总数：{report.stats.total_claims}    "
        f"已判定有问题：{report.stats.suspicious_claims}    "
        f"需关注（含疑点/证据缺口）：{report.stats.flagged_claims}"
    )
    print("状态分布：" + "  ".join(
        f"{name}={count}" for name, count in report.stats.status_counts.items() if count
    ))
    print(line)

    for claim in report.claims:
        result = report.result_by_claim(claim.claim_id)
        print(f"\n[{claim.claim_id}] {claim.claim_type.value}  优先级={claim.priority.value}  "
              f"可核验性={claim.checkability.value}")
        print(f"  原文片段：{claim.source_span.text}")
        print(f"  规范化：  {claim.normalized_claim}")
        if claim.verification_questions:
            print("  核验问题：" + " / ".join(claim.verification_questions))
        if result is None:
            print("  （无核验结果）")
            continue
        print(f"  结论：    {result.status.value}  严重度={result.severity.value}  规则={result.adjudication_rule}")
        print(f"  理由：    {result.rationale}")
        if result.problem_types:
            print("  问题类型：" + "、".join(p.value for p in result.problem_types))
        if result.refuting_evidence_ids:
            print("  相关证据：")
            for evidence_id in result.refuting_evidence_ids:
                evidence = report.evidence_by_id(evidence_id)
                if evidence:
                    print(f"    - [{evidence.evidence_id}] {evidence.source.value}"
                          f"／{evidence.level.value}：{evidence.snippet}")
        if result.missing_evidence:
            print("  还缺什么：" + " ".join(result.missing_evidence))

    if report.limitations:
        print("\n" + line)
        print("系统局限（报告中会如实展示）：")
        for item in report.limitations:
            print(f"  · {item}")

    print("\n" + line)
    print(
        f"耗时：{report.metrics.latency_ms} ms    "
        f"大模型调用：{report.metrics.llm_calls} 次（离线规则引擎不计为大模型调用）"
    )
    print(f"成本说明：{report.metrics.cost_note}")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="智鉴 AI 命令行演示")
    parser.add_argument("--list", action="store_true", help="列出所有示例文本")
    parser.add_argument("--id", default="", help="按示例 ID 运行")
    parser.add_argument("--text", default="", help="直接指定待核验文本")
    parser.add_argument("--json", default="", help="把完整报告另存为 JSON 文件")
    args = parser.parse_args()

    samples = load_samples()

    if args.list:
        print(f"共 {len(samples)} 条示例：")
        for sample in samples:
            print(f"  - {sample['id']:<24} {sample['title']}")
        return 0

    if args.text:
        text = args.text
        label = "（命令行输入）"
    else:
        if not samples:
            print(f"找不到示例文件：{SAMPLES_PATH}")
            return 1
        sample_id = args.id or samples[0]["id"]
        match = next((s for s in samples if s["id"] == sample_id), None)
        if match is None:
            print(f"没有 ID 为 {sample_id!r} 的示例。可用：{[s['id'] for s in samples]}")
            return 1
        text = match["text"]
        label = f"{match['title']}（{match['id']}）"

    settings = Settings()
    print(f"\n>>> 示例：{label}")
    print(f">>> 引擎：{settings.llm_provider}    运行模式：{settings.run_mode.value}")
    print(f">>> 输入（{len(text)} 字）：{text}\n")

    try:
        report = run_detection(text, settings=settings)
    except VeriMindError as exc:
        print(f"检测失败：{exc}")
        return 1

    print_report(report)

    if args.json:
        target = Path(args.json)
        target.write_text(report.to_json(), encoding="utf-8")
        print(f"完整报告已写入：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
