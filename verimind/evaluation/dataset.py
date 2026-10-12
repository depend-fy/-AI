"""人工标注测试集：加载、校验与对齐。

**诚实性要求（写进代码里）**：
    · 每条标注都必须带 `source_of_truth`（真值是怎么确认的）与 `labeler`（谁标的）；
    · 标注者含 AI 时必须在字段里写明，评测页会原样显示，不允许含糊成"人工标注"；
    · 真值尽量锚定在**可复现的实测**上（例如 Crossref 实际返回的年份），
      而不是"我觉得应该是这样"。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from verimind.core.enums import ProblemType, VerificationStatus
from verimind.core.schemas import Claim, DetectionReport, VerificationResult, VeriMindModel


class GoldenCase(VeriMindModel):
    """一条标注用例。"""

    case_id: str
    category: str
    text: str = Field(min_length=1)
    gold_claim_quote: str = Field(min_length=1, description="用于把预测结果与标注对齐的原文片段")
    gold_status: VerificationStatus
    gold_problem_types: list[ProblemType] = Field(default_factory=list)
    gold_evidence_doi: str | None = None
    expect_correction_contains: list[str] = Field(
        default_factory=list, description="修正后全文应当包含的片段（用于衡量修正成功率）"
    )
    expect_correction_excludes: list[str] = Field(
        default_factory=list, description="修正后全文不应当包含的片段（用于衡量剩余错误率）"
    )
    notes: str = ""
    source_of_truth: str = Field(min_length=1, description="真值是怎么确认的")
    labeler: str = Field(min_length=1, description="标注者（含 AI 时必须写明）")
    label_date: str = ""
    human_confirmed: bool = Field(
        default=False, description="是否已由人工逐条确认（未确认的不能被表述为「人工标注结果」）"
    )
    human_comment: str = Field(default="", description="人工确认时的意见")

    @property
    def awaiting_confirmation(self) -> bool:
        """还没经人工确认——评测页必须把这一部分单独标出来。"""
        return not self.human_confirmed

    @property
    def gold_is_problem(self) -> bool:
        return self.gold_status in (
            VerificationStatus.CONTRADICTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
            VerificationStatus.SOURCE_CONFLICT,
        )

    @property
    def needs_correction(self) -> bool:
        return bool(self.expect_correction_contains or self.expect_correction_excludes)

    @property
    def labeler_is_ai(self) -> bool:
        return "AI" in self.labeler.upper() or "起草" in self.labeler


DEFAULT_GOLDEN_SET = Path("examples/golden_set.jsonl")


def load_golden_set(path: Path | None = None) -> list[GoldenCase]:
    """读取标注集（JSONL，一行一条）。"""
    target = path or DEFAULT_GOLDEN_SET
    if not target.exists():
        raise FileNotFoundError(f"找不到标注集：{target}")
    cases: list[GoldenCase] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(GoldenCase.model_validate(json.loads(line)))
    return cases


def match_claim(report: DetectionReport, case: GoldenCase) -> tuple[Claim | None, VerificationResult | None]:
    """把标注里指定的原文片段对齐到报告中的某条主张。

    优先"主张片段包含标注片段"，其次"标注片段包含主张片段"，
    最后退回"两者都在同一句里"——都找不到就返回 None（算作未命中，不计入指标分母前会单独报告）。
    """
    quote = case.gold_claim_quote
    for claim in report.claims:
        if quote in claim.source_span.text:
            return claim, report.result_by_claim(claim.claim_id)
    for claim in report.claims:
        if claim.source_span.text in quote:
            return claim, report.result_by_claim(claim.claim_id)
    for claim in report.claims:
        sentence = claim.parent_sentence_span
        if sentence is not None and quote in sentence.text:
            return claim, report.result_by_claim(claim.claim_id)
    return None, None


def categories(cases: list[GoldenCase]) -> list[str]:
    seen: list[str] = []
    for case in cases:
        if case.category not in seen:
            seen.append(case.category)
    return seen
