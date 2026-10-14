"""核心枚举定义：全项目"词汇表"的唯一事实来源。

为什么把所有枚举集中在一个文件？
    检测状态（6 种）和问题类型（8 类）会被 claims / retrieval / verification /
    correction / review / ui 六个模块同时引用。如果各写各的字符串，
    改一个字就会静默失配。集中定义 + 用枚举代替字符串，能在导入期就暴露错误。

命名约定：
    - 枚举成员名用英文大写（代码里引用它）
    - 枚举值用中文（界面直接展示，也是报告 JSON 里的值）
"""

from __future__ import annotations

from enum import Enum


class _ZhEnum(str, Enum):
    """基类：让枚举既能当字符串用，又能打印出中文。"""

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return str(self.value)


# --------------------------------------------------------------------------
# 一、核验状态（用户要求：每条主张至少支持这 6 种状态）
# --------------------------------------------------------------------------
class VerificationStatus(_ZhEnum):
    VERIFIED = "已核实"           # 现有证据能够支持该主张
    CONTRADICTED = "存在矛盾"     # 找到可信证据与该主张冲突
    PARTIALLY_SUPPORTED = "部分支持"  # 证据只支持主张的一部分
    INSUFFICIENT = "证据不足"     # 已有材料不足以支持或反驳
    UNVERIFIABLE = "暂无法核验"   # 当前检索条件下无法作出可靠判断
    SOURCE_CONFLICT = "来源冲突"  # 不同可信来源给出不一致的信息


#: 会被算作"系统认为这条主张有问题"的状态（评测里作为正类）。
#: 注意："证据不足"和"暂无法核验"不是错误，绝不能拿来充数当检出。
SUSPICIOUS_STATUSES: tuple[VerificationStatus, ...] = (
    VerificationStatus.CONTRADICTED,
    VerificationStatus.PARTIALLY_SUPPORTED,
    VerificationStatus.SOURCE_CONFLICT,
)

#: 界面里状态的固定展示顺序（从好到坏，不是按字母）。
STATUS_DISPLAY_ORDER: tuple[VerificationStatus, ...] = (
    VerificationStatus.VERIFIED,
    VerificationStatus.PARTIALLY_SUPPORTED,
    VerificationStatus.CONTRADICTED,
    VerificationStatus.SOURCE_CONFLICT,
    VerificationStatus.INSUFFICIENT,
    VerificationStatus.UNVERIFIABLE,
)


# --------------------------------------------------------------------------
# 二、问题类型（用户要求：至少识别这 8 类）
# --------------------------------------------------------------------------
class ProblemType(_ZhEnum):
    FACT_ERROR = "事实错误"
    CITATION_MISSING = "引用不存在或元数据不匹配"
    CITATION_MISMATCH = "引用与结论不匹配"
    NUMERIC_ERROR = "数据或数字错误"
    METADATA_ERROR = "时间、人物或机构信息错误"
    OVERREACH = "逻辑推断过度"
    MISSING_EVIDENCE = "缺少必要证据"
    SOURCE_CONFLICT = "不同来源之间存在冲突"


#: 按"证据要求"分组，供阶段 4/5 的规则裁决使用：
#: 这些类型必须挂在至少一条真实证据上，否则不允许判定。
EVIDENCE_REQUIRED_PROBLEMS: tuple[ProblemType, ...] = (
    ProblemType.FACT_ERROR,
    ProblemType.CITATION_MISSING,
    ProblemType.CITATION_MISMATCH,
    ProblemType.NUMERIC_ERROR,
    ProblemType.METADATA_ERROR,
    ProblemType.SOURCE_CONFLICT,
)

#: 这些类型是"证据缺口"性质，可以由检索痕迹（没找到东西）支撑。
GAP_PROBLEMS: tuple[ProblemType, ...] = (
    ProblemType.MISSING_EVIDENCE,
    ProblemType.OVERREACH,
)


# --------------------------------------------------------------------------
# 三、主张类型与优先级
# --------------------------------------------------------------------------
class ClaimType(_ZhEnum):
    WORK_EXISTENCE = "文献是否存在"
    DOI_VALIDITY = "DOI 是否真实"
    METADATA_YEAR = "年份类"
    METADATA_AUTHOR = "作者类"
    METADATA_VENUE = "期刊/会议类"
    QUANTITATIVE = "数值/统计类"
    CAUSAL = "因果/机制类"
    ATTRIBUTION = "归属类"
    GENERAL_FACT = "一般事实类"
    OPINION = "观点/预测类"


class ClaimPriority(_ZhEnum):
    P0 = "高"
    P1 = "中"
    P2 = "低"


class Checkability(_ZhEnum):
    CHECKABLE = "可核验"
    PARTIALLY_CHECKABLE = "部分可核验"
    NOT_CHECKABLE = "不可核验"


class ExtractionConfidence(_ZhEnum):
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


# --------------------------------------------------------------------------
# 四、证据相关
# --------------------------------------------------------------------------
class EvidenceSource(_ZhEnum):
    CROSSREF = "Crossref"
    OPENALEX = "OpenAlex"
    ARXIV = "arXiv"
    WIKIDATA = "Wikidata"
    WIKIPEDIA = "Wikipedia"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    EUROPE_PMC = "Europe PMC"
    MANUAL = "人工提供"
    WEBPAGE = "网页"
    #: 证据来自用户原文自身（例如"文内自相矛盾"：同一文献在原文里出现了两个年份）。
    #: 这类证据不需要、也不允许伪造外部 URL，但必须给出原文定位。
    INTERNAL = "原文内部"

    @property
    def is_external(self) -> bool:
        """外部来源必须提供真实可点击的 URL；原文内部证据不需要。"""
        return self is not EvidenceSource.INTERNAL


class EvidenceLevel(_ZhEnum):
    """证据的"内容层级"。这是本项目的诚实性底线之一：

    只有"仅元数据"时，你只能判断文献是否存在、元数据是否一致，
    绝不能声称"这篇论文支持/反驳了该结论"。
    """

    METADATA_ONLY = "仅元数据"
    ABSTRACT = "摘要"
    FULLTEXT = "全文"
    WEBPAGE = "网页内容"
    USER_TEXT = "用户原文"


class EvidenceRelation(_ZhEnum):
    SUPPORTS = "支持"
    PARTIALLY_SUPPORTS = "部分支持"
    REFUTES = "反驳"
    IRRELEVANT = "无关"
    UNKNOWN = "无法判断"


class MatchBasis(_ZhEnum):
    METADATA = "元数据匹配"
    ABSTRACT = "摘要匹配"
    FULLTEXT = "全文匹配"
    INTERNAL = "文内比对"
    NONE = "未匹配"


class JudgedAspect(_ZhEnum):
    """一条证据被用来判断的是"哪个层面"。

    这是本项目防止**越界判断**的关键设计：
        · 只有元数据（DOI 记录、标题、年份）时，只能判断文献是否存在、书目信息是否一致；
          绝不能据此声称"这篇论文支持/反驳了某个结论"。
        · 要判断内容结论是否被支持，必须有摘要或全文。

    换句话说：『文献是真的』和『文献真的这么说』是两个问题，必须分开回答。
    """

    EXISTENCE = "文献是否存在"
    BIBLIOGRAPHIC = "书目信息（年份/作者/期刊）"
    CONTENT = "内容结论是否被支持"
    INTERNAL_CONSISTENCY = "原文内部一致性"

    @property
    def requires_text(self) -> bool:
        """该层面是否必须有摘要/全文才能判断。"""
        return self is JudgedAspect.CONTENT


class Severity(_ZhEnum):
    NONE = "无"
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


# --------------------------------------------------------------------------
# 五、修正与复核
# --------------------------------------------------------------------------
class ChangeType(_ZhEnum):
    KEEP = "保留原文"
    REPLACE_FACT = "更正事实"
    FIX_NUMBER = "更正数字"
    REPLACE_CITATION = "更正引用"
    WEAKEN = "弱化表述"
    ADD_UNCERTAINTY = "标注不确定性"
    DELETE = "删除该句"


#: 这些修改会写入"新的事实内容"，因此必须有证据支撑（无证据不许改）。
FACT_CHANGING_TYPES: tuple[ChangeType, ...] = (
    ChangeType.REPLACE_FACT,
    ChangeType.FIX_NUMBER,
    ChangeType.REPLACE_CITATION,
)


class ReviewCheckType(_ZhEnum):
    ORIGINAL_ISSUE_RESOLVED = "原始问题是否已解决"
    CORRECTION_EVIDENCE_BACKED = "修正内容是否有证据支持"
    NO_NEW_FACT_ERROR = "是否引入新的事实错误"
    NO_NEW_OVERREACH = "是否出现新的无依据结论"
    EVIDENCE_MATCHES_TEXT = "证据与修正后表述是否匹配"


class ReviewOverall(_ZhEnum):
    PASSED = "通过"
    PASSED_WITH_RISKS = "通过但有风险"
    FAILED = "未通过"


# --------------------------------------------------------------------------
# 六、角色、运行模式、调用状态
# --------------------------------------------------------------------------
class RoleName(_ZhEnum):
    CLAIM_EXTRACTOR = "主张拆解员"
    FACT_CHECKER = "事实审查员"
    EVIDENCE_CHECKER = "证据审查员"
    INDEPENDENT_REVIEWER = "独立复核员"
    CORRECTOR = "修正员"


class RoleStance(_ZhEnum):
    SUPPORT = "倾向支持"
    REFUTE = "倾向反驳"
    UNCERTAIN = "无法确定"
    NOT_APPLICABLE = "不适用"


class RunMode(_ZhEnum):
    """本次检测究竟用了什么模式——必须如实展示，不许含糊。

    这是比赛答辩的诚信底线：只有真的接了多家模型，才允许说"多模型协同"。
    """

    MOCK = "离线模拟"
    SINGLE_MODEL_MULTI_ROLE = "单模型多角色模拟"
    MULTI_PROVIDER = "多模型协同"


class LlmCallStatus(_ZhEnum):
    OK = "成功"
    TIMEOUT = "超时"
    ERROR = "调用失败"
    SCHEMA_RETRY = "格式修复后成功"
    CACHE_HIT = "命中缓存"
    BUDGET_EXCEEDED = "超出调用预算"


class SearchStatus(_ZhEnum):
    OK = "检索成功"
    NO_HIT = "已检索但无命中"
    ERROR = "检索失败"
    SKIPPED = "未检索"


#: 便于测试与界面校验的数量断言（6 状态 / 8 问题类型）。
EXPECTED_STATUS_COUNT = 6
EXPECTED_PROBLEM_TYPE_COUNT = 8
EXPECTED_REVIEW_CHECK_COUNT = 5
