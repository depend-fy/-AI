"""原子事实主张拆解。

阶段 2 用的是**规则引擎**（正则 + 关键词 + 位置定位），不是大模型。
这在报告和界面上都会如实标注 —— 它不是"模型在思考"，而是"代码在按规则找线索"。

它的价值不止是"离线兜底"：
    1. 完全确定性，可以被穷举单元测试（零成本、可复现）；
    2. 阶段 3 接入大模型后，两者输出同一种格式（ClaimDraft），
       可以做**交叉比对**："模型说这里引用了文献，规则说这句话里根本没有 DOI"。

拆分原则（本项目的一条核心设计）：
    拆解方只负责**引用原文**，不负责计算字符位置。
    位置由代码用 `find_span()` 换算；换算不出来就丢弃该条并在报告里说明，
    绝不硬凑一个错误的高亮区间。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verimind.claims.preprocess import DocumentAnalysis, SentenceSignals, TextMatch, analyze_text
from verimind.core.enums import (
    Checkability,
    ClaimPriority,
    ClaimType,
    ExtractionConfidence,
)
from verimind.core.schemas import Claim, ClaimDraft, ClaimExtraction, Span, find_span

#: 优先级排序权重（数字越小越先核验）
PRIORITY_RANK: dict[ClaimPriority, int] = {
    ClaimPriority.P0: 0,
    ClaimPriority.P1: 1,
    ClaimPriority.P2: 2,
}

#: 判断一句话"像事实陈述"的动词（用于兜底生成一般事实类主张）
_FACTUAL_VERB_RE = re.compile(r"是|为|达|达到|超过|不足|属于|位于|占|拥有|成立于")


@dataclass(frozen=True)
class ExtractionResult:
    """主张拆解的最终结果（已带精确位置）。"""

    claims: list[Claim] = field(default_factory=list)
    total_detected: int = 0
    truncated: bool = False
    dropped_unlocatable: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.claims)


# ==========================================================================
# 草稿生成（按"一句话"为单位）
# ==========================================================================
def _drafts_for_sentence(
    signals: SentenceSignals, sentence: Span
) -> list[tuple[ClaimDraft, Span]]:
    """针对一句话，产出候选主张草稿及其在原文中的位置。

    返回顺序就是报告中的展示顺序：DOI → 年份 → 数值 → 因果 → 归属 → 模糊引用 → 一般事实 → 观点。
    """
    out: list[tuple[ClaimDraft, Span]] = []
    seen: set[tuple[str, str]] = set()

    def add(draft: ClaimDraft, span: Span) -> None:
        key = (str(draft.claim_type), span.text)
        if key in seen:
            return
        seen.add(key)
        out.append((draft, span))

    # 「大量研究表明……」这类用模糊表述代替具体来源的句子，单独成一条主张就够了。
    # 否则同一句话会同时产出"文献是否存在""因果推断""模糊引用"三条，
    # 报告看起来像在凑数量——那是评测里最忌讳的事。
    vague_without_source = bool(signals.vague_words) and not signals.has_concrete_source

    # ---- ① DOI：真实性与元数据一致性（最高优先级）----
    for doi in signals.dois:
        add(
            ClaimDraft(
                quote=doi.text,
                normalized_claim=f"原文引用了一篇 DOI 为 {doi.text} 的文献。",
                claim_type=ClaimType.DOI_VALIDITY,
                priority=ClaimPriority.P0,
                verification_questions=[
                    "该 DOI 是否在 DOI 注册机构中真实存在？",
                    "该 DOI 对应文献的标题、作者、年份是否与原文描述一致？",
                ],
            ),
            doi.span,
        )

    for bad in signals.invalid_dois:
        add(
            ClaimDraft(
                quote=bad.text,
                normalized_claim=(
                    f"原文引用的 DOI「{bad.text}」在格式上不是合法 DOI（前缀应为 4~9 位数字）。"
                ),
                claim_type=ClaimType.DOI_VALIDITY,
                priority=ClaimPriority.P0,
                verification_questions=[
                    "该 DOI 是否符合 DOI 语法规范？",
                    "是否存在与之一一对应的真实文献？",
                ],
                confidence=ExtractionConfidence.HIGH,
            ),
            bad.span,
        )

    # ---- ② 文献是否存在（提到了研究，但没有给出可核验的标识）----
    if signals.has_study_word and not signals.dois and not signals.invalid_dois and not vague_without_source:
        add(
            ClaimDraft(
                quote=sentence.text,
                normalized_claim="原文提到了一项研究/报告，但未给出可核验的标识（DOI、标题或期刊）。",
                claim_type=ClaimType.WORK_EXISTENCE,
                priority=ClaimPriority.P1,
                verification_questions=[
                    "原文提到的这篇研究是否真实存在？",
                    "能否通过标题、作者或期刊检索到对应文献？",
                ],
            ),
            sentence,
        )

    # ---- ③ 年份类元数据 ----
    if signals.years and (signals.has_study_word or signals.dois or signals.invalid_dois):
        for year in signals.years:
            year_value = int(year.value) if isinstance(year.value, int) else None
            if year_value is None:
                continue
            add(
                ClaimDraft(
                    quote=year.text,
                    normalized_claim=f"原文所述文献的发表年份为 {year_value} 年。",
                    claim_type=ClaimType.METADATA_YEAR,
                    priority=ClaimPriority.P0,
                    verification_questions=[
                        f"该文献的实际发表年份是否确为 {year_value} 年？",
                        "该年份是否与 DOI / 标题所对应的文献一致？",
                    ],
                ),
                year.span,
            )

    # ---- ④ 数值类主张（最容易被夸大的一类）----
    strong_values: list[TextMatch] = [*signals.percents, *signals.strong_quantities]
    for value in strong_values:
        add(
            ClaimDraft(
                quote=value.span.text,
                normalized_claim=f"原文给出了一个具体数值：{value.span.text}。",
                claim_type=ClaimType.QUANTITATIVE,
                priority=ClaimPriority.P0,
                verification_questions=[
                    "该数值是否有可追溯的原始出处？",
                    "该数值是否被所引用的研究/数据支持？",
                ],
            ),
            value.span,
        )
    if signals.has_study_word:
        for value in signals.weak_quantities:
            add(
                ClaimDraft(
                    quote=value.span.text,
                    normalized_claim=f"原文给出了一个规模性数据：{value.span.text}。",
                    claim_type=ClaimType.QUANTITATIVE,
                    priority=ClaimPriority.P1,
                    verification_questions=["该数据的来源是什么？是否可核验？"],
                ),
                value.span,
            )

    # ---- ⑤ 因果 / 结论性表述（"证明""意味着"最容易过度推断）----
    if signals.causal_words and not vague_without_source:
        add(
            ClaimDraft(
                quote=sentence.text,
                normalized_claim="原文声称某项研究或数据支持了一个结论性/因果性判断。",
                claim_type=ClaimType.CAUSAL,
                priority=ClaimPriority.P1,
                verification_questions=[
                    "所引用的来源是否真的支持该结论？",
                    "原文给出的是相关性证据还是因果性证据？",
                ],
            ),
            sentence,
        )

    # ---- ⑥ 归属类（把说法归于某机构）----
    if signals.orgs and (signals.causal_words or signals.has_study_word):
        for org in signals.orgs:
            add(
                ClaimDraft(
                    quote=org.text,
                    normalized_claim=f"原文将某个说法归属于「{org.text}」。",
                    claim_type=ClaimType.ATTRIBUTION,
                    priority=ClaimPriority.P2,
                    verification_questions=[
                        f"「{org.text}」是否确实发布过相关内容？",
                        "该归属是否与原始来源一致？",
                    ],
                ),
                org.span,
            )

    # ---- ⑦ 模糊权威（"大量研究表明"但没有具体来源）----
    # 注意：如果这句话本身是观点/预测（例如"业内专家普遍认为……"），
    # 那么"缺证据"就不是主要问题——主要问题是它把观点当成了事实。
    # 那种情况交给下面的 ⑧ 处理，这里不再重复出主张，避免同一句话产出两条同义主张。
    if vague_without_source and not signals.is_opinion_like:
        vague_word = signals.vague_words[0].text
        # 如果同时用了因果措辞（"多项研究表明……不可逆转"），按因果类处理，
        # 这样规则层会同时给出「逻辑推断过度」与「缺少必要证据」两个问题类型。
        if signals.causal_words:
            add(
                ClaimDraft(
                    quote=sentence.text,
                    normalized_claim=(
                        f"原文用「{vague_word}」这类模糊表述支撑一个结论性判断，"
                        "既没有给出具体来源，也没有说明证据强度。"
                    ),
                    claim_type=ClaimType.CAUSAL,
                    priority=ClaimPriority.P1,
                    verification_questions=[
                        f"「{vague_word}」具体指哪些研究？",
                        "这些研究是否真的支持该结论？",
                    ],
                ),
                sentence,
            )
        else:
            add(
                ClaimDraft(
                    quote=sentence.text,
                    normalized_claim=(
                        f"原文用「{vague_word}」这类模糊表述支撑结论，"
                        "但没有给出任何具体来源。"
                    ),
                    claim_type=ClaimType.GENERAL_FACT,
                    priority=ClaimPriority.P1,
                    verification_questions=[
                        f"「{vague_word}」具体指哪些研究？",
                        "能否给出可核验的文献标识（DOI、标题或期刊）？",
                    ],
                ),
                sentence,
            )

    # ---- ⑧ 观点 / 预测类（明确标注为不可核验，避免被当成事实错误）----
    if signals.is_opinion_like:
        vague_hint = (
            f"（其中使用了「{signals.vague_words[0].text}」这类模糊表述）"
            if signals.vague_words
            else ""
        )
        add(
            ClaimDraft(
                quote=sentence.text,
                normalized_claim=(
                    f"原文包含预测或观点性表述{vague_hint}，"
                    "不属于可以用外部证据核验的事实主张。"
                ),
                claim_type=ClaimType.OPINION,
                priority=ClaimPriority.P2,
                checkability=Checkability.NOT_CHECKABLE,
                verification_questions=[],
                confidence=ExtractionConfidence.HIGH,
            ),
            sentence,
        )

    # ---- ⑨ 兜底：像事实陈述、但前面什么特征都没命中 ----
    if not out and not signals.is_opinion_like and len(sentence.text) >= 8:
        if _FACTUAL_VERB_RE.search(sentence.text):
            add(
                ClaimDraft(
                    quote=sentence.text,
                    normalized_claim="原文包含一个可以用外部来源核验的事实性陈述。",
                    claim_type=ClaimType.GENERAL_FACT,
                    priority=ClaimPriority.P2,
                    verification_questions=["该陈述是否有可核验的公开来源？"],
                    confidence=ExtractionConfidence.LOW,
                ),
                sentence,
            )

    return out


# ==========================================================================
# 草稿 → Claim（带位置、编号、去重、截断）
# ==========================================================================
def drafts_to_claims(
    text: str,
    extraction: ClaimExtraction,
    *,
    max_claims: int,
    analysis: DocumentAnalysis | None = None,
    span_hints: list[Span | None] | None = None,
    extracted_by: str = "",
) -> ExtractionResult:
    """把拆解结果落地成 `Claim` 列表。

    - `span_hints`：规则引擎已经知道精确位置时可直接给出，避免重复查找导致的错位；
      大模型拆解时传 None，由代码用 `find_span()` 按引用文本定位。
    - 定位失败 → 丢弃该条并记录，**不硬凑位置**。
    - 按优先级排序、去重、按 `max_claims` 截断。
    """
    analysis = analysis or analyze_text(text)
    hints = span_hints or [None] * len(extraction.claims)

    resolved: list[tuple[int, int, ClaimDraft, Span, Span]] = []
    dropped = 0
    for index, draft in enumerate(extraction.claims):
        hint = hints[index] if index < len(hints) else None
        span = hint or find_span(text, draft.quote)
        if span is None or not span.is_inside(text):
            dropped += 1
            continue
        parent = analysis.nearest_sentence(span) or span
        resolved.append((PRIORITY_RANK.get(draft.priority, 9), span.start, draft, span, parent))

    resolved.sort(key=lambda item: (item[0], item[1]))

    unique: list[tuple[int, int, ClaimDraft, Span, Span]] = []
    seen: set[tuple[str, str]] = set()
    for item in resolved:
        _, _, draft, span, _ = item
        key = (str(draft.claim_type), span.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    total_detected = len(unique)
    truncated = total_detected > max_claims
    selected = unique[:max_claims]

    claims = [
        Claim(
            claim_id=f"C{position:03d}",
            source_span=span,
            parent_sentence_span=parent,
            normalized_claim=draft.normalized_claim,
            claim_type=draft.claim_type,
            priority=draft.priority,
            checkability=draft.checkability,
            verification_questions=list(draft.verification_questions),
            extracted_by=extracted_by,
            extraction_confidence=draft.confidence,
        )
        for position, (_, _, draft, span, parent) in enumerate(selected, start=1)
    ]

    notes = list(extraction.notes)
    if dropped:
        notes.append(f"{dropped} 条候选主张因为无法在原文中精确定位而被丢弃。")

    return ExtractionResult(
        claims=claims,
        total_detected=total_detected,
        truncated=truncated,
        dropped_unlocatable=dropped,
        notes=notes,
    )


def collect_drafts(
    text: str, *, analysis: DocumentAnalysis | None = None
) -> tuple[list[ClaimDraft], list[Span], DocumentAnalysis]:
    """扫描全文，产出候选主张草稿序列与它们在原文中的精确位置。

    返回的 `drafts` 与 `spans` 一一对应（下标对齐），
    这样既能满足"只输出引用文本"的大模型契约，又能保留规则引擎的定位精度。
    """
    analysis = analysis or analyze_text(text)
    drafts: list[ClaimDraft] = []
    hints: list[Span] = []
    for sentence in analysis.sentences:
        signals = analysis.for_span(sentence)
        for draft, span in _drafts_for_sentence(signals, sentence):
            drafts.append(draft)
            hints.append(span)
    return drafts, hints, analysis


# ==========================================================================
# 规则拆解器（实现与大模型客户端相同的输出格式）
# ==========================================================================
class RuleBasedExtractor:
    """基于正则与关键词的主张拆解器（确定性、离线、零成本）。"""

    name = "rule-based-extractor"
    version = "0.2.1"

    @property
    def engine_id(self) -> str:
        return f"{self.name}-{self.version}"

    def extract(self, text: str, *, max_claims: int, extracted_by: str = "") -> ExtractionResult:
        drafts, hints, analysis = collect_drafts(text)
        extraction = ClaimExtraction(
            claims=drafts,
            notes=(
                []
                if drafts
                else ["规则引擎未能从文本中识别出可核验的事实主张（可能整段都是观点或过渡性描述）。"]
            ),
        )
        return drafts_to_claims(
            text,
            extraction,
            max_claims=max_claims,
            analysis=analysis,
            span_hints=list(hints),
            extracted_by=extracted_by or self.engine_id,
        )


#: 默认实例（无状态，可复用）
DEFAULT_EXTRACTOR = RuleBasedExtractor()
