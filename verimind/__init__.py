"""智鉴 AI（VeriMind AI）——多模型协同的 AI 幻觉检测与证据溯源系统。

包结构（阶段 0 设计，按阶段逐步填充实现）：
    core/        配置、枚举、核心数据结构（唯一事实来源）、流程编排
    claims/      原子事实主张提取
    models/      大模型适配层与角色调用
    retrieval/   外部文献与证据检索
    verification/证据与主张匹配、规则裁决
    attribution/ 问题类型归因与证据缺口分析
    correction/  基于证据的修正
    review/      修正后的独立复核
    evaluation/  评测指标与实验数据
    storage/     历史记录与数据持久化
    ui/          界面组件与主题
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
