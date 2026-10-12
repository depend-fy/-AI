"""智鉴 AI 核心数据结构（全项目的数据契约，唯一事实来源）。

设计原则
--------
1. **禁止在模块之间传裸 dict**。所有跨模块流转的数据都必须是这里的模型，
   这样"某个字段拼错了"会在边界处立刻报错，而不是等到界面上显示成空白。
2. **没有证据，就不允许说"存在矛盾/部分支持"**。见 VerificationResult 的校验器：
   判定有问题的状态必须挂上真实的 evidence_id，并有判断理由。
3. **不允许假装读过全文**。证据必须声明自己的内容层级（EvidenceLevel 的
   "仅元数据/摘要/全文"），反驳类判断还必须引用证据片段（snippet）。
4. **报告必须自洽**。DetectionReport 会校验所有 claim_id / evidence_id /
   correction_id 的引用完整性，"报告里指向了一条不存在的证据"这种错误无法通过校验。
5. 给大模型的结构化输出 schema 就来自这些模型（`compact_json_schema`），
   模型输出与内部数据结构因此天然一致。

注意：Pydantic 的 `extra="forbid"` 意味着模型多输出一个字段都会被拒绝——
这是我们对"结构化输出"的强约束，配合阶段 3 的"格式修复重试"使用。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verimind.core.enums import (
    EXPECTED_REVIEW_CHECK_COUNT,
    FACT_CHANGING_TYPES,
    SUSPICIOUS_STATUSES,
    ChangeType,
    Checkability,
    ClaimPriority,
    ClaimType,
    EvidenceLevel,
    EvidenceRelation,
    EvidenceSource,
    ExtractionConfidence,
    LlmCallStatus,
    MatchBasis,
    ProblemType,
    ReviewCheckType,
    ReviewOverall,
    RoleName,
    RoleStance,
    RunMode,
    SearchStatus,
    Severity,
    VerificationStatus,
)
from verimind.core.enums import STATUS_DISPLAY_ORDER

DEFAULT_DISCLAIMER = (
    "本报告是系统基于可检索到的公开证据所作出的辅助判断，不是学术结论、法律意见或事实认定的最终依据。"
    "系统无法确知其他模型内部真实的出错原因；报告中的问题类型是依据可观察证据作出的归因。"
    "当证据不足或无法核验时，系统会明确标注，而不是给出猜测性结论。"
)


def utc_now() -> datetime:
    """统一使用 UTC 时间，避免跨时区比较出错。"""
    return datetime.now(timezone.utc)


class VeriMindModel(BaseModel):
    """全项目数据模型基类。"""

    model_config = ConfigDict(
        extra="forbid",  # 多余字段直接报错（对 LLM 输出尤其重要）
        str_strip_whitespace=True,
        validate_assignment=False,
    )


# ==========================================================================
# 基础部件
# ==========================================================================
class Span(VeriMindModel):
    """原文片段定位。

    `start` / `end` 是**原始文本中的字符下标**（左闭右开），
    `text` 必须与原文切片完全一致 —— 这一条保证界面上能精确高亮，
    也保证"系统引用原文"这件事没有被篡改。
    """

    start: int = Field(ge=0, description="起始字符下标（含）")
    end: int = Field(ge=0, description="结束字符下标（不含）")
    text: str = Field(description="原文切片，必须与 full_text[start:end] 完全一致")

    @model_validator(mode="after")
    def _check_consistency(self) -> "Span":
        if self.end < self.start:
            raise ValueError(f"end({self.end}) 不能小于 start({self.start})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"text 长度({len(self.text)}) 与 end-start({self.end - self.start}) 不一致，"
                "原文定位会错位，请用 Span.from_text() 构造"
            )
        return self

    @classmethod
    def from_text(cls, full_text: str, start: int, end: int) -> "Span":
        """从完整文本 + 下标构造，自动填充 text（推荐入口）。"""
        if start < 0 or end > len(full_text) or end < start:
            raise ValueError(f"非法区间 [{start}, {end})，原文长度 {len(full_text)}")
        return cls(start=start, end=end, text=full_text[start:end])

    def is_inside(self, full_text: str) -> bool:
        """校验该片段在给定原文中确实位于同一位置（用于报告自检）。"""
        return full_text[self.start : self.end] == self.text


class SearchRecord(VeriMindModel):
    """一次检索动作的记录：查了哪个源、用了什么检索式、结果如何。

    没有命中也必须记录 —— "我查过了但没找到"是"暂无法核验"的正当依据，
    同时也是向用户证明"系统真的去查了"。
    """

    source: EvidenceSource
    query: str = Field(min_length=1)
    status: SearchStatus = SearchStatus.OK
    hit_count: int = Field(default=0, ge=0)
    error: str | None = None
    detail: str = ""


class SearchTrace(VeriMindModel):
    """一条主张的完整检索痕迹。"""

    records: list[SearchRecord] = Field(default_factory=list)
    searched_at: datetime = Field(default_factory=utc_now)

    @property
    def total_hits(self) -> int:
        return sum(r.hit_count for r in self.records)

    @property
    def all_failed(self) -> bool:
        """所有检索都失败了（此时结论必须是"暂无法核验"而不是"证据不足"）。"""
        return bool(self.records) and all(r.status is SearchStatus.ERROR for r in self.records)

    @property
    def searched_sources(self) -> list[str]:
        return sorted({str(r.source.value) for r in self.records})


class RolePosition(VeriMindModel):
    """某个角色对某条主张的判断（用于在报告里展示角色的分歧）。"""

    role: RoleName
    provider: str = ""
    model: str = ""
    stance: RoleStance = RoleStance.UNCERTAIN
    conclusion: str = Field(min_length=1, description="角色结论要点")
    reasoning: str = Field(default="", description="角色给出的推理/依据")


# ==========================================================================
# 一、事实主张
# ==========================================================================
class Claim(VeriMindModel):
    """一条可单独核验的原子事实主张。"""

    claim_id: str = Field(pattern=r"^C\d{3,}$", description="形如 C001，报告内唯一")
    source_span: Span = Field(description="该主张在原文中的位置")
    normalized_claim: str = Field(min_length=1, description="规范化后的主张表述")
    claim_type: ClaimType
    priority: ClaimPriority = ClaimPriority.P1
    checkability: Checkability = Checkability.CHECKABLE
    verification_questions: list[str] = Field(
        default_factory=list,
        description="要核验这条主张，必须回答的具体问题（至少 1 条）",
    )
    initial_status: VerificationStatus | None = Field(
        default=None, description="初步核验状态；None 表示尚未核验"
    )
    extracted_by: str = Field(default="", description="由哪个角色/模型拆解出来（可追溯）")
    extraction_confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM

    @model_validator(mode="after")
    def _require_questions(self) -> "Claim":
        if self.checkability is not Checkability.NOT_CHECKABLE and not self.verification_questions:
            raise ValueError(
                f"{self.claim_id} 标记为可核验，但没有给出任何核验问题。"
                "没有明确的核验问题，就无法检索证据，也无法判断。"
            )
        return self


# ==========================================================================
# 二、外部证据
# ==========================================================================
class Evidence(VeriMindModel):
    """一条来自外部来源的证据。

    关键约束：
        - `url` 必须是真实请求返回的地址，不允许模型自行编造（阶段 4 会用测试断言）。
        - `level` 必须如实标注内容层级；只有 `仅元数据` 时不得声称能判断结论是否被支持。
        - 判定为"支持/部分支持/反驳"时必须有 `relation_rationale`；
          判定为"反驳"时还必须给出 `snippet`（证据原文片段）。
    """

    evidence_id: str = Field(pattern=r"^E\d{3,}$")
    claim_ids: list[str] = Field(default_factory=list, description="关联的主张 ID")
    source: EvidenceSource
    level: EvidenceLevel
    url: str = Field(min_length=1, description="真实可点击的来源地址")
    title: str = ""
    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=2999)
    venue: str | None = Field(default=None, description="期刊/会议名称")
    publisher: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    query: str = Field(default="", description="触发这条证据的检索式（可复现）")
    raw_ref: str | None = Field(default=None, description="本地原始响应留档路径")
    snippet: str | None = Field(default=None, description="证据原文片段（摘要/正文节选）")
    relation: EvidenceRelation = EvidenceRelation.UNKNOWN
    relation_rationale: str = ""
    match_basis: MatchBasis = MatchBasis.NONE

    @model_validator(mode="after")
    def _check_consistency(self) -> "Evidence":
        if not self.url.lower().startswith(("http://", "https://")):
            raise ValueError(f"{self.evidence_id} 的 url 不是可点击的 http(s) 地址：{self.url!r}")

        judged = self.relation in (
            EvidenceRelation.SUPPORTS,
            EvidenceRelation.PARTIALLY_SUPPORTS,
            EvidenceRelation.REFUTES,
        )
        if judged and not self.relation_rationale.strip():
            raise ValueError(
                f"{self.evidence_id} 判定为「{self.relation.value}」，但没给出判断理由（relation_rationale）。"
            )
        if self.relation is EvidenceRelation.REFUTES and not (self.snippet or "").strip():
            raise ValueError(
                f"{self.evidence_id} 判定为「反驳」，但没有引用证据原文片段（snippet）——"
                "反驳必须能指出原文到底说了什么。"
            )
        if self.level is EvidenceLevel.METADATA_ONLY and judged:
            raise ValueError(
                f"{self.evidence_id} 只有元数据（无摘要/全文），却给出了"
                f"「{self.relation.value}」的关系判断。仅凭元数据只能核验文献是否存在与元数据是否一致。"
            )
        return self

    @property
    def display_name(self) -> str:
        if self.title:
            year = f"（{self.year}）" if self.year else ""
            return f"{self.title}{year}"
        return self.doi or self.url


# ==========================================================================
# 三、核验结果
# ==========================================================================
class VerificationResult(VeriMindModel):
    """一条主张的最终核验结论。由规则裁决层产出（不是模型直接说了算）。"""

    claim_id: str
    status: VerificationStatus
    problem_types: list[ProblemType] = Field(default_factory=list)
    severity: Severity = Severity.NONE
    rationale: str = Field(min_length=1, description="面向用户的判断理由")
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    related_evidence_ids: list[str] = Field(default_factory=list)
    search_trace: SearchTrace = Field(default_factory=SearchTrace)
    missing_evidence: list[str] = Field(
        default_factory=list, description="还缺什么才能作出判断（对用户最有用的信息之一）"
    )
    role_positions: list[RolePosition] = Field(default_factory=list)
    adjudication_rule: str = Field(default="", description="命中的规则编号与说明（可解释性）")

    @model_validator(mode="after")
    def _enforce_evidence_contract(self) -> "VerificationResult":
        evidence_refs = (
            set(self.supporting_evidence_ids)
            | set(self.refuting_evidence_ids)
            | set(self.related_evidence_ids)
        )
        bad = self.status in SUSPICIOUS_STATUSES

        if bad and not self.problem_types:
            raise ValueError(
                f"{self.claim_id} 的状态为「{self.status.value}」，必须指明至少一个问题类型。"
            )
        if bad and not evidence_refs:
            raise ValueError(
                f"{self.claim_id} 的状态为「{self.status.value}」，但没有任何证据 ID。"
                "没有证据就不允许判定主张有问题。"
            )
        if self.status is VerificationStatus.VERIFIED and not self.supporting_evidence_ids:
            raise ValueError(
                f"{self.claim_id} 判定为「已核实」，必须给出至少一条支持性证据。"
            )
        if self.status is VerificationStatus.SOURCE_CONFLICT and len(evidence_refs) < 2:
            raise ValueError(
                f"{self.claim_id} 判定为「来源冲突」，至少需要两条来源不一致的证据。"
            )
        if self.status in (VerificationStatus.INSUFFICIENT, VerificationStatus.UNVERIFIABLE):
            if not self.missing_evidence:
                raise ValueError(
                    f"{self.claim_id} 的状态为「{self.status.value}」，必须写明还缺什么证据（missing_evidence），"
                    "否则用户不知道下一步该怎么办。"
                )
            if not self.search_trace.records:
                raise ValueError(
                    f"{self.claim_id} 的状态为「{self.status.value}」，但没有任何检索记录。"
                    "没有检索记录就说'查不到'是不成立的。"
                )
        return self

    @property
    def is_suspicious(self) -> bool:
        return self.status in SUSPICIOUS_STATUSES


# ==========================================================================
# 四、修正与复核
# ==========================================================================
class Correction(VeriMindModel):
    """一条基于证据的修正建议。"""

    correction_id: str = Field(pattern=r"^K\d{3,}$")
    claim_id: str
    original_span: Span = Field(description="被修改的原文片段")
    proposed_text: str = Field(min_length=1, description="建议修改后的文本")
    change_type: ChangeType
    rationale: str = Field(min_length=1, description="为什么要这样改")
    evidence_ids: list[str] = Field(default_factory=list, description="支撑本次修改的证据")
    risk_note: str | None = Field(default=None, description="该修改自身可能带来的风险")

    @model_validator(mode="after")
    def _require_evidence_for_fact_change(self) -> "Correction":
        if self.change_type in FACT_CHANGING_TYPES and not self.evidence_ids:
            raise ValueError(
                f"{self.correction_id} 属于「{self.change_type.value}」——"
                "这类修改会写入新的事实内容，必须有证据支撑（evidence_ids 不能为空）。"
                "如果拿不到证据，请改用『弱化表述 / 标注不确定性 / 删除』。"
            )
        return self


class ReviewCheck(VeriMindModel):
    """独立复核中的单项检查。"""

    check: ReviewCheckType
    passed: bool
    detail: str = Field(min_length=1, description="无论通过与否，都要说明依据")


class ReviewResult(VeriMindModel):
    """修正后的独立复核结论。"""

    review_id: str = Field(pattern=r"^V\d{3,}$")
    target_correction_id: str
    checks: list[ReviewCheck] = Field(default_factory=list)
    overall: ReviewOverall
    unresolved: list[str] = Field(default_factory=list, description="复核后仍未解决的问题")
    reviewer: RolePosition | None = None

    @model_validator(mode="after")
    def _require_all_checks(self) -> "ReviewResult":
        provided = {c.check for c in self.checks}
        missing = [c.value for c in ReviewCheckType if c not in provided]
        if missing:
            raise ValueError(
                f"{self.review_id} 缺少必要的复核项：{'、'.join(missing)}"
                f"（必须完成全部 {EXPECTED_REVIEW_CHECK_COUNT} 项检查）"
            )
        if self.overall is ReviewOverall.PASSED and not all(c.passed for c in self.checks):
            raise ValueError(f"{self.review_id} 存在未通过的检查项，结论不能是「通过」。")
        if self.overall is not ReviewOverall.PASSED and not self.unresolved:
            raise ValueError(
                f"{self.review_id} 的结论为「{self.overall.value}」，必须列出仍未解决的问题（unresolved）。"
            )
        return self


# ==========================================================================
# 五、调用日志与统计
# ==========================================================================
class LlmCallLog(VeriMindModel):
    """一次大模型调用的记录（用于成本核算与问题排查）。

    默认只记录提示词哈希，不记录完整提示词 —— 避免把用户文本散落在日志里。
    """

    call_id: str
    run_id: str | None = None
    role: RoleName | None = None
    provider: str = ""
    model: str = ""
    prompt_hash: str = ""
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    status: LlmCallStatus = LlmCallStatus.OK
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class ReportStats(VeriMindModel):
    """报告顶部统计卡片的数字来源。必须由实际结果算出，不允许手工填写。"""

    total_claims: int = Field(default=0, ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    problem_counts: dict[str, int] = Field(default_factory=dict)
    suspicious_claims: int = Field(default=0, ge=0)
    corrections_count: int = Field(default=0, ge=0)
    reviews_passed: int = Field(default=0, ge=0)
    reviews_failed: int = Field(default=0, ge=0)


class RunMetrics(VeriMindModel):
    """运行指标。成本没有真实单价来源时保持 None，不猜数字。"""

    latency_ms: int | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated_cost: float | None = Field(
        default=None, description="只有在配置了明确单价后才计算；否则保持 None"
    )
    cost_note: str = ""


# ==========================================================================
# 六、完整检测报告
# ==========================================================================
class DetectionReport(VeriMindModel):
    """一次完整检测的全部产物，也是落盘与界面展示的根对象。"""

    run_id: str = Field(min_length=8)
    created_at: datetime = Field(default_factory=utc_now)
    app_version: str = ""
    pipeline_version: str = ""

    run_mode: RunMode = Field(description="本次运行的真是模式，必须在界面如实展示")
    input_text: str = Field(min_length=1)
    input_sha256: str = ""

    llm_config_summary: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="角色→模型映射（不含密钥）"
    )

    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    review_results: list[ReviewResult] = Field(default_factory=list)

    stats: ReportStats = Field(default_factory=ReportStats)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    unresolved_claim_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list, description="系统自己声明的局限（如：未获得全文，仅依据摘要判断）"
    )
    disclaimer: str = DEFAULT_DISCLAIMER

    # ---------------- 校验：报告必须自洽 ----------------
    @model_validator(mode="after")
    def _check_reference_integrity(self) -> "DetectionReport":
        problems: list[str] = []

        claim_ids = [c.claim_id for c in self.claims]
        evidence_ids = [e.evidence_id for e in self.evidence]
        correction_ids = [k.correction_id for k in self.corrections]

        if len(set(claim_ids)) != len(claim_ids):
            problems.append("claim_id 存在重复")
        if len(set(evidence_ids)) != len(evidence_ids):
            problems.append("evidence_id 存在重复")
        if len(set(correction_ids)) != len(correction_ids):
            problems.append("correction_id 存在重复")

        known_claims, known_evidence = set(claim_ids), set(evidence_ids)

        for e in self.evidence:
            for cid in e.claim_ids:
                if cid not in known_claims:
                    problems.append(f"{e.evidence_id} 关联了不存在的主张 {cid}")

        for vr in self.verification_results:
            if vr.claim_id not in known_claims:
                problems.append(f"核验结果关联了不存在的主张 {vr.claim_id}")
            for eid in (
                vr.supporting_evidence_ids + vr.refuting_evidence_ids + vr.related_evidence_ids
            ):
                if eid not in known_evidence:
                    problems.append(f"{vr.claim_id} 的核验结果引用了不存在的证据 {eid}")

        for k in self.corrections:
            if k.claim_id not in known_claims:
                problems.append(f"{k.correction_id} 关联了不存在的主张 {k.claim_id}")
            for eid in k.evidence_ids:
                if eid not in known_evidence:
                    problems.append(f"{k.correction_id} 引用了不存在的证据 {eid}")

        for r in self.review_results:
            if r.target_correction_id not in set(correction_ids):
                problems.append(f"{r.review_id} 复核了不存在的修正 {r.target_correction_id}")

        for cid in self.unresolved_claim_ids:
            if cid not in known_claims:
                problems.append(f"未解决问题列表里有不存在的主张 {cid}")

        if problems:
            raise ValueError("报告引用完整性校验失败：" + "；".join(problems))
        return self

    # ---------------- 便捷方法 ----------------
    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        input_text: str,
        run_mode: RunMode,
        app_version: str = "",
        pipeline_version: str = "",
        llm_config_summary: dict[str, dict[str, str]] | None = None,
    ) -> "DetectionReport":
        return cls(
            run_id=run_id,
            input_text=input_text,
            run_mode=run_mode,
            app_version=app_version,
            pipeline_version=pipeline_version,
            input_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
            llm_config_summary=llm_config_summary or {},
        )

    def claim_by_id(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)

    def result_by_claim(self, claim_id: str) -> VerificationResult | None:
        return next((r for r in self.verification_results if r.claim_id == claim_id), None)

    def evidence_by_id(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidence if e.evidence_id == evidence_id), None)

    def suspicious_claims(self) -> list[Claim]:
        """按严重度排序的可疑主张（报告页"可疑主张列表"用）。"""
        order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.NONE: 3}
        results = [r for r in self.verification_results if r.is_suspicious]
        results.sort(key=lambda r: order.get(r.severity, 9))
        out: list[Claim] = []
        for r in results:
            claim = self.claim_by_id(r.claim_id)
            if claim is not None:
                out.append(claim)
        return out

    def recompute(self) -> "DetectionReport":
        """依据当前数据重算统计与"未解决问题"清单（不允许手工填统计数字）。"""
        status_counts = {s.value: 0 for s in STATUS_DISPLAY_ORDER}
        problem_counts = {p.value: 0 for p in ProblemType}
        for vr in self.verification_results:
            status_counts[vr.status.value] = status_counts.get(vr.status.value, 0) + 1
            for p in vr.problem_types:
                problem_counts[p.value] = problem_counts.get(p.value, 0) + 1

        failed_corrections = {
            r.target_correction_id for r in self.review_results if r.overall is ReviewOverall.FAILED
        }
        unresolved: list[str] = []
        for vr in self.verification_results:
            if vr.status in (VerificationStatus.INSUFFICIENT, VerificationStatus.UNVERIFIABLE):
                unresolved.append(vr.claim_id)
        for k in self.corrections:
            if k.correction_id in failed_corrections and k.claim_id not in unresolved:
                unresolved.append(k.claim_id)
        # 说明：复核结论里的 unresolved 是自由文本（如"仍缺少全文证据"），
        # 它按条展示在报告的复核区域，不混进这里的 claim_id 清单。

        self.stats = ReportStats(
            total_claims=len(self.claims),
            status_counts=status_counts,
            problem_counts={k: v for k, v in problem_counts.items() if v},
            suspicious_claims=sum(1 for r in self.verification_results if r.is_suspicious),
            corrections_count=len(self.corrections),
            reviews_passed=sum(1 for r in self.review_results if r.overall is ReviewOverall.PASSED),
            reviews_failed=sum(1 for r in self.review_results if r.overall is ReviewOverall.FAILED),
        )
        self.unresolved_claim_ids = unresolved
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """序列化为 JSON 文本（中文不转义，便于直接阅读与作为答辩材料）。"""
        return self.model_dump_json(indent=indent, exclude_none=False)

    @classmethod
    def from_json(cls, text: str) -> "DetectionReport":
        return cls.model_validate_json(text)


# ==========================================================================
# 工具函数
# ==========================================================================
def compact_json_schema(model_cls: Type[BaseModel]) -> str:
    """把数据模型压成紧凑 JSON Schema，用于写进给大模型的提示词（阶段 3 使用）。

    这样"给模型的输出格式"和"我们内部校验用的格式"永远是同一份定义，
    不会出现"提示词里说要 X 字段，代码里却按 Y 解析"的经典事故。
    """
    schema: dict[str, Any] = model_cls.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def find_span(full_text: str, snippet: str, *, start_hint: int = 0) -> Span | None:
    """在原文中定位一个片段，返回精确的 Span（定位不到返回 None，绝不瞎猜）。

    阶段 2 会用它把模型给出的原文引用换算成可高亮的字符区间；
    如果模型改写了原文、导致找不到，我们宁可放弃定位，也不硬凑一个错误区间。
    """
    if not snippet:
        return None
    idx = full_text.find(snippet, start_hint)
    if idx < 0:
        idx = full_text.find(snippet)
    if idx < 0:
        return None
    return Span.from_text(full_text, idx, idx + len(snippet))
