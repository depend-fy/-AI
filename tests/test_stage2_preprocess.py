"""阶段 2 测试：文本预处理（句子切分与文本信号）。

这里的每一条断言，都对应一个真实踩过的坑：
    · 同一个年份被识别两次 → 报告里出现两条重复主张；
    · DOI 里的数字被当成年份 → 报告里出现"文献发表于 2023 年"这种莫名其妙的主张；
    · 信号抽取错了，后面所有规则都会跟着错，所以必须逐条钉死。
"""

from __future__ import annotations

from verimind.claims.preprocess import (
    analyze_text,
    is_valid_doi_prefix,
    looks_like_doi,
    split_sentences,
)


# ==========================================================================
# 句子切分
# ==========================================================================
def test_split_sentences_keeps_exact_offsets() -> None:
    text = "第一句。第二句！第三句？"
    spans = split_sentences(text)
    assert [span.text for span in spans] == ["第一句", "第二句", "第三句"]
    for span in spans:
        assert span.is_inside(text), "句子片段的偏移必须能精确定位原文"


def test_split_sentences_handles_newlines_and_semicolons() -> None:
    text = "第一句；\n第二句\n\n第三句"
    assert [span.text for span in split_sentences(text)] == ["第一句", "第二句", "第三句"]


def test_split_sentences_ignores_empty_fragments() -> None:
    assert split_sentences("。。。") == []
    assert split_sentences("") == []


# ==========================================================================
# 年份识别（回归：重复年份 / DOI 内部数字）
# ==========================================================================
def test_year_with_suffix_is_detected_once() -> None:
    analysis = analyze_text("据一项发表于 2023 年的研究显示，结论成立。")
    assert [int(year.value) for year in analysis._years] == [2023]


def test_year_is_not_detected_twice_when_written_as_digits_only() -> None:
    """回归用例：文本同时出现"发表于 2023 年"时，不能既算"2023 年"又算"2023"。"""
    analysis = analyze_text("该成果发表于 2023 年，随后被多次引用。")
    assert len(analysis._years) == 1
    assert analysis._years[0].text.strip() == "2023 年"


def test_year_inside_doi_is_not_treated_as_a_year() -> None:
    """回归用例：10.xxxx/verimind.2023.001 里的 2023 不是年份。"""
    analysis = analyze_text("相关论文（DOI: 10.xxxx/verimind.2023.001）已经发表。")
    assert analysis._years == []


def test_bare_year_requires_publishing_context() -> None:
    text_with_context = "这篇论文发表于 2023，结论可靠。"
    text_without_context = "订单编号 2023，金额 100 元。"
    assert analyze_text(text_with_context)._years, "有发表语境时应当识别出年份"
    assert analyze_text(text_without_context)._years == [], "没有发表语境时不应识别成文献年份"


def test_year_signal_is_scoped_to_its_sentence() -> None:
    text = "第一批结果发表于 2019 年。第二批结果发表于 2021 年。"
    analysis = analyze_text(text)
    first_sentence = analysis.sentences[0]
    signals = analysis.for_span(first_sentence)
    assert [int(year.value) for year in signals.years] == [2019]


# ==========================================================================
# DOI
# ==========================================================================
def test_valid_doi_is_detected_and_prefix_accepted() -> None:
    analysis = analyze_text("参见 DOI: 10.1038/nature12373 的原文。")
    assert [doi.text for doi in analysis._dois] == ["10.1038/nature12373"]
    assert analysis._invalid_dois == []
    assert is_valid_doi_prefix("10.1038/nature12373")


def test_placeholder_doi_is_flagged_as_invalid_prefix() -> None:
    analysis = analyze_text("参见 DOI: 10.xxxx/example.2023.001 的原文。")
    assert analysis._dois == []
    assert [doi.text for doi in analysis._invalid_dois] == ["10.xxxx/example.2023.001"]
    assert looks_like_doi("10.xxxx/example.2023.001")
    assert not is_valid_doi_prefix("10.xxxx/example.2023.001")


def test_doi_trailing_punctuation_is_stripped() -> None:
    analysis = analyze_text("引用为 DOI: 10.1038/nature12373, 详见原文。")
    assert [doi.text for doi in analysis._dois] == ["10.1038/nature12373"]


# ==========================================================================
# 数值与其它信号
# ==========================================================================
def test_percent_and_strong_quantity_are_detected() -> None:
    signals = analyze_text("准确率提高了 40%，速度提升 3 倍。").for_span(
        analyze_text("准确率提高了 40%，速度提升 3 倍。").sentences[0]
    )
    assert [value.value for value in signals.percents] == [40]
    assert [value.value for value in signals.strong_quantities] == [3]


def test_weak_quantity_needs_study_context() -> None:
    with_context = analyze_text("该研究统计了 12000 名受试者。")
    without_context = analyze_text("车站今天发送旅客 12000 人。")
    signals_a = with_context.for_span(with_context.sentences[0])
    signals_b = without_context.for_span(without_context.sentences[0])
    assert signals_a.weak_quantities, "有研究语境时应识别规模性数据"
    assert signals_b.has_study_word is False
    assert signals_b.has_number_like is False, "没有研究语境时不应把它当成研究数据"


def test_opinion_and_vague_authority_signals() -> None:
    opinion = analyze_text("预计到 2030 年该技术将全面普及。")
    vague = analyze_text("大量研究表明该方法是有效的。")
    assert opinion.for_span(opinion.sentences[0]).is_opinion_like is True
    assert vague.for_span(vague.sentences[0]).vague_words
    assert vague.for_span(vague.sentences[0]).has_concrete_source is False


def test_concrete_source_detection() -> None:
    with_doi = analyze_text("该研究（DOI: 10.1038/nature12373）显示结论成立。")
    with_venue = analyze_text("该成果发表在《自然》杂志上。")
    without = analyze_text("该研究显示结论成立。")
    assert with_doi.for_span(with_doi.sentences[0]).has_concrete_source is True
    assert with_venue.for_span(with_venue.sentences[0]).has_concrete_source is True
    assert without.for_span(without.sentences[0]).has_concrete_source is False


def test_metric_words_are_detected() -> None:
    analysis = analyze_text("模型准确率提高到 95%，误差降低到 3%。")
    signals = analysis.for_span(analysis.sentences[0])
    metric_names = {metric.text for metric in signals.metrics}
    assert "准确率" in metric_names
    assert "误差" in metric_names
