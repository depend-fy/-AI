"""文本预处理：句子切分 + "文本信号"提取。

为什么要有这一层？
    主张拆解和规则裁决都需要知道"这句话里有没有 DOI、年份、数值、引用标记、因果词"。
    把这些判断集中成**纯函数**，好处是：
      · 可以穷举单元测试，完全离线、零成本；
      · 阶段 3 接入大模型后，这些信号仍然是极有价值的交叉验证依据
        （模型说"这里有引用"，但规则说这句话里根本没有 DOI —— 这就是一个矛盾点）。

所有 Span 都指向**原始文本的字符下标**，保证报告里能精确高亮。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verimind.core.schemas import Span

# ==========================================================================
# 正则表（集中在此，方便逐条测试和调整）
# ==========================================================================
#: 合法的 DOI 前缀：10. + 4~9 位数字
_DOI_VALID_RE = re.compile(r"10\.\d{4,9}/[^\s，。；、）)】」》]+")
#: 任何"看起来像 DOI"的串（含非法前缀，例如占位符 10.xxxx/example）
_DOI_ANY_RE = re.compile(r"10\.[^\s/，。；、）)】」》]{1,20}/[^\s，。；、）)】」》]+")
#: DOI 前缀合法性（用于判定"格式不合法"）
_DOI_PREFIX_OK_RE = re.compile(r"^10\.\d{4,9}/")

_YEAR_CN_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*年")
_YEAR_BARE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_YEAR_CONTEXT_RE = re.compile(r"发表|发布|出版|上线|成立|提出|发表于")
#: 裸年份必须落在语境词前后多少字符以内才算数
_YEAR_CONTEXT_WINDOW = 15

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")
_STRONG_QUANTITY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(倍|个百分点)")
_WEAK_QUANTITY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|亿|人|例|名|份|项|篇|组)")

_STUDY_RE = re.compile(r"研究|论文|文献|期刊|报告|调查|统计|综述|白皮书|预印本|实验")
_VAGUE_AUTHORITY_RE = re.compile(
    r"大量研究|多项研究|诸多研究|研究表明|研究显示|相关研究|有研究|据统计|数据显示|"
    r"业内专家|专家表示|专家指出|普遍认为|众所周知|业界公认"
)
_CAUSAL_RE = re.compile(r"证明|证实|表明了?|说明|意味着|导致|由此可见|充分说明|足以说明")
_OPINION_RE = re.compile(
    r"认为|觉得|预计|预测|有望|可能会|或许|未来|将要|应该会|展望|大概会|看好|前景|据推测"
)
_ORG_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z]{2,14}(?:大学|学院|研究所|研究院|实验室|科学院|医院|"
    r"统计局|委员会|协会|基金会|集团|公司|疾控中心)"
)
_VENUE_RE = re.compile(r"《[^》]{2,40}》|(?:Nature|Science|Cell|Lancet|NEJM|JAMA|PNAS|arXiv)")
_METRIC_WORD_RE = re.compile(
    r"准确率|准确度|精度|召回率|F1|f1|误差|效率|性能|成功率|覆盖率|渗透率|增长率|"
    r"市场份额|收益率|转化率|通过率|满意度|得分|指标"
)

#: 句子结束符（中文标点与换行）
_SENTENCE_BOUNDARY = "。！？!?；;\n"


# ==========================================================================
# 数据结构
# ==========================================================================
@dataclass(frozen=True)
class TextMatch:
    """原文中的一处匹配。"""

    text: str
    span: Span
    value: float | int | None = None

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return self.text


@dataclass(frozen=True)
class SentenceSignals:
    """**某一句话**里检测到的全部信号。规则裁决最终看的就是它。"""

    span: Span
    text: str
    dois: list[TextMatch] = field(default_factory=list)
    invalid_dois: list[TextMatch] = field(default_factory=list)
    years: list[TextMatch] = field(default_factory=list)
    percents: list[TextMatch] = field(default_factory=list)
    strong_quantities: list[TextMatch] = field(default_factory=list)
    weak_quantities: list[TextMatch] = field(default_factory=list)
    orgs: list[TextMatch] = field(default_factory=list)
    causal_words: list[TextMatch] = field(default_factory=list)
    vague_words: list[TextMatch] = field(default_factory=list)
    opinion_words: list[TextMatch] = field(default_factory=list)
    venues: list[TextMatch] = field(default_factory=list)
    metrics: list[TextMatch] = field(default_factory=list)

    # ---------------- 布尔判断（规则表直接用这些） ----------------
    @property
    def has_study_word(self) -> bool:
        """是否提到"研究/论文/报告/统计"这类出处性词汇。"""
        return bool(_STUDY_RE.search(self.text))

    @property
    def has_concrete_source(self) -> bool:
        """是否给出了**可核验的具体出处**：DOI、文献标题、期刊名、链接。"""
        return bool(
            self.dois
            or self.venues
            or re.search(r"https?://\S+", self.text)
            or re.search(r"arXiv:\s*\d{4}\.\d{4,5}", self.text, re.IGNORECASE)
        )

    @property
    def has_number_like(self) -> bool:
        return bool(
            self.percents
            or self.strong_quantities
            or (self.weak_quantities and self.has_study_word)
        )

    @property
    def is_opinion_like(self) -> bool:
        return bool(self.opinion_words)


@dataclass
class DocumentAnalysis:
    """整段文本的分析结果，可按需取出任意一句话的信号。"""

    text: str
    sentences: list[Span]
    _dois: list[TextMatch] = field(default_factory=list)
    _invalid_dois: list[TextMatch] = field(default_factory=list)
    _years: list[TextMatch] = field(default_factory=list)
    _percents: list[TextMatch] = field(default_factory=list)
    _strong_quantities: list[TextMatch] = field(default_factory=list)
    _weak_quantities: list[TextMatch] = field(default_factory=list)
    _orgs: list[TextMatch] = field(default_factory=list)
    _causal_words: list[TextMatch] = field(default_factory=list)
    _vague_words: list[TextMatch] = field(default_factory=list)
    _opinion_words: list[TextMatch] = field(default_factory=list)
    _venues: list[TextMatch] = field(default_factory=list)
    _metrics: list[TextMatch] = field(default_factory=list)

    def for_span(self, span: Span) -> SentenceSignals:
        """取出落在给定区间内的全部信号（区间通常是"一句话"）。"""

        def inside(items: list[TextMatch]) -> list[TextMatch]:
            return [m for m in items if span.start <= m.span.start and m.span.end <= span.end]

        return SentenceSignals(
            span=span,
            text=self.text[span.start : span.end],
            dois=inside(self._dois),
            invalid_dois=inside(self._invalid_dois),
            years=inside(self._years),
            percents=inside(self._percents),
            strong_quantities=inside(self._strong_quantities),
            weak_quantities=inside(self._weak_quantities),
            orgs=inside(self._orgs),
            causal_words=inside(self._causal_words),
            vague_words=inside(self._vague_words),
            opinion_words=inside(self._opinion_words),
            venues=inside(self._venues),
            metrics=inside(self._metrics),
        )

    def sentence_containing(self, span: Span) -> Span | None:
        """找出包含给定区间的那句话（报告里用来显示上下文）。"""
        for sentence in self.sentences:
            if sentence.start <= span.start and span.end <= sentence.end:
                return sentence
        return None

    def nearest_sentence(self, span: Span) -> Span | None:
        """找不到完全包含的句子时，退而求其次找起点所在的句子。"""
        contained = self.sentence_containing(span)
        if contained is not None:
            return contained
        for sentence in self.sentences:
            if sentence.start <= span.start < sentence.end:
                return sentence
        return None


# ==========================================================================
# 主要函数
# ==========================================================================
def looks_like_doi(value: str) -> bool:
    """判断一个字符串是否"长得像 DOI"（含非法前缀的占位符也算）。"""
    return bool(_DOI_ANY_RE.match(value.strip()))


def is_valid_doi_prefix(value: str) -> bool:
    """判断 DOI 前缀是否合规（必须形如 10. + 4~9 位数字）。

    注意：这只检查**语法**，不代表该 DOI 真实存在——真实存在必须查 Crossref。
    """
    return bool(_DOI_PREFIX_OK_RE.match(value.strip()))


def split_sentences(text: str) -> list[Span]:
    """按中文标点与换行切句，返回**带精确偏移**的句子片段。

    说明：片段不含句末标点本身（高亮时更干净）。
    """
    spans: list[Span] = []
    for match in re.finditer(f"[^{re.escape(_SENTENCE_BOUNDARY)}]+", text):
        start, end = match.start(), match.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append(Span.from_text(text, start, end))
    return spans


def _find_all(
    pattern: re.Pattern[str],
    text: str,
    *,
    value_of: str | None = None,
    strip_trailing: str = "",
) -> list[TextMatch]:
    """通用匹配函数：把正则结果转成带 Span 的 TextMatch 列表。"""
    out: list[TextMatch] = []
    for match in pattern.finditer(text):
        raw = match.group(0)
        cleaned = raw.rstrip(strip_trailing) if strip_trailing else raw
        if not cleaned:
            continue
        start, end = match.start(), match.start() + len(cleaned)
        value: float | int | None = None
        if value_of is not None:
            try:
                matched_value = match.group(1)
                value = float(matched_value) if "." in matched_value else int(matched_value)
            except (IndexError, ValueError, TypeError):
                value = None
        out.append(TextMatch(text=cleaned, span=Span.from_text(text, start, end), value=value))
    return out


def analyze_text(text: str) -> DocumentAnalysis:
    """对全文做一次扫描，产出句子与全部文本信号。"""
    analysis = DocumentAnalysis(text=text, sentences=split_sentences(text))

    analysis._dois = _find_all(_DOI_VALID_RE, text, strip_trailing=".,;:)]}")
    # 非法 DOI = 像 DOI 但前缀不合法（例如 10.xxxx/example 这种占位符）
    valid_texts = {m.text for m in analysis._dois}
    analysis._invalid_dois = [
        m
        for m in _find_all(_DOI_ANY_RE, text, strip_trailing=".,;:)]}")
        if m.text not in valid_texts and not _DOI_PREFIX_OK_RE.match(m.text)
    ]

    years = _find_all(_YEAR_CN_RE, text, value_of="value")
    # 没有"年"字的年份（例如"发表于 2023"）：只在"发表/发布/出版"等语境词附近才认。
    # 必须排除两种误判，否则报告里会出现重复或荒谬的"年份主张"：
    #   ① 已被"YYYY 年"覆盖的位置（否则同一年份算两次）；
    #   ② DOI 内部的数字（例如 10.xxxx/verimind.2023.001 里的 2023 并不是年份）。
    if _YEAR_CONTEXT_RE.search(text):
        context_spans = [m.span() for m in _YEAR_CONTEXT_RE.finditer(text)]
        doi_spans = [d.span for d in analysis._dois] + [d.span for d in analysis._invalid_dois]
        for match in _find_all(_YEAR_BARE_RE, text, value_of="value"):
            covered_by_year = any(
                year.span.start <= match.span.start and match.span.end <= year.span.end
                for year in years
            )
            inside_doi = any(
                doi.start <= match.span.start and match.span.end <= doi.end for doi in doi_spans
            )
            near_context = any(
                (start - _YEAR_CONTEXT_WINDOW) <= match.span.start
                and match.span.end <= (end + _YEAR_CONTEXT_WINDOW)
                for start, end in context_spans
            )
            if not covered_by_year and not inside_doi and near_context:
                years.append(match)
    analysis._years = sorted(years, key=lambda m: m.span.start)

    analysis._percents = _find_all(_PERCENT_RE, text, value_of="value")
    analysis._strong_quantities = _find_all(_STRONG_QUANTITY_RE, text, value_of="value")
    analysis._weak_quantities = _find_all(_WEAK_QUANTITY_RE, text, value_of="value")
    analysis._orgs = _find_all(_ORG_RE, text)
    analysis._causal_words = _find_all(_CAUSAL_RE, text)
    analysis._vague_words = _find_all(_VAGUE_AUTHORITY_RE, text)
    analysis._opinion_words = _find_all(_OPINION_RE, text)
    analysis._venues = _find_all(_VENUE_RE, text)
    analysis._metrics = _find_all(_METRIC_WORD_RE, text)
    return analysis
