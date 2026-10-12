"""大模型适配层的抽象接口。

这个文件是**阶段 3 的预留接口**，但从阶段 2 就开始使用：
    阶段 2 的实现是"离线规则引擎"（mock_provider），阶段 3 会新增
    "OpenAI 兼容协议的客户端"（openai_compat），两者实现同一个接口。
    因此流水线、报告、界面**一行都不用改**就能从"规则"切到"真模型"。

接口设计原则
------------
1. **只输出结构化数据**：调用方传 Pydantic 模型，拿回该模型的实例。
   校验失败由客户端负责（修复重试 1 次后仍失败则抛 SchemaValidationError），
   绝不把"看起来差不多"的半成品交给下游。
2. **如实标注来源**：每次调用都返回 `LlmUsage`，其中明确写着
   provider / model / 是否模拟 / token 数 / 耗时 / 状态。
   报告与界面据此标注"本次结果来自离线规则引擎"还是"来自真实模型"。
3. **离线实现必须自曝身份**：`is_simulated` 默认返回 True ——
   宁可多标一句"这是模拟的"，也不能让人误以为真的调用了大模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

from verimind.claims.extractor import ExtractionResult, drafts_to_claims
from verimind.core.config import Settings
from verimind.core.enums import LlmCallStatus, RoleName
from verimind.core.errors import LLMError
from verimind.core.schemas import ClaimExtraction, Span
from verimind.models.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True)
class LlmUsage:
    """一次调用的元信息（用于成本核算、问题排查与诚实标注）。"""

    provider: str
    model: str
    role: RoleName | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    status: LlmCallStatus = LlmCallStatus.OK
    cache_hit: bool = False
    simulated: bool = False
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def engine_id(self) -> str:
        return f"{self.provider}/{self.model}"


class LLMClient(ABC):
    """所有"会产出结构化判断的东西"的统一接口（规则引擎也算一种实现）。"""

    provider: str = "unknown"
    model: str = "unknown"

    @property
    def engine_id(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def is_simulated(self) -> bool:
        """是否为模拟实现。默认 True —— 安全默认值：不主动声称自己是真模型。"""
        return True

    # ------------------------------------------------------------------
    @abstractmethod
    def complete_structured(
        self,
        *,
        role: RoleName,
        prompt: str,
        schema: type[TModel],
        system: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> tuple[TModel, LlmUsage]:
        """按给定 schema 返回结构化结果。

        `context` 是给**离线实现**用的结构化输入（例如原文）。
        真实模型实现可以忽略它——因为同样的信息已经写在 prompt 里了。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def span_hints(
        self, text: str, extraction: ClaimExtraction
    ) -> list[Span | None] | None:
        """可选的"位置提示"。

        规则引擎自己就精确知道每个片段的位置，直接给出提示可以避免
        `find_span()` 在原文里有重复片段时定位到错的那一处。
        真实模型实现返回 None，由代码按引用文本自动定位。
        """
        return None

    def extract_claims(
        self, text: str, *, max_claims: int, extracted_by: str = ""
    ) -> tuple[ExtractionResult, LlmUsage]:
        """主张拆解的高层入口：构造提示词 → 调用 → 落地成带位置的 Claim。"""
        prompt = build_extraction_prompt(text, max_claims)
        extraction, usage = self.complete_structured(
            role=RoleName.CLAIM_EXTRACTOR,
            prompt=prompt,
            schema=ClaimExtraction,
            system=EXTRACTION_SYSTEM_PROMPT,
            context={"text": text, "max_claims": max_claims},
        )
        result = drafts_to_claims(
            text,
            extraction,
            max_claims=max_claims,
            span_hints=self.span_hints(text, extraction),
            extracted_by=extracted_by or self.engine_id,
        )
        return result, usage

    def health(self) -> tuple[bool, str]:
        """可用性自检（界面上可以显示"当前引擎是否可用"）。"""
        return True, "ok"


# ==========================================================================
# 工厂
# ==========================================================================
def build_client(settings: Settings) -> LLMClient:
    """根据配置构造客户端。

    阶段 2 只有离线规则引擎；配置了真实 provider 时**明确报错**，
    而不是偷偷降级成规则引擎假装在工作。
    """
    provider = (settings.llm_provider or "").strip().lower()
    if provider in ("", "mock"):
        from verimind.models.mock_provider import RuleBasedOfflineClient

        return RuleBasedOfflineClient()

    raise LLMError(
        f"当前版本（阶段 2）尚未接入真实模型调用：VERIMIND_LLM_PROVIDER={provider!r}。\n"
        "请在 .env 中设置 VERIMIND_LLM_PROVIDER=mock 以使用离线规则引擎；\n"
        "真实大模型适配器将在阶段 3 实现（接口已经预留，届时只需换实现）。"
    )
