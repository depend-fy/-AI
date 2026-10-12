"""大模型适配层的抽象接口。

设计原则
--------
1. **只输出结构化数据**：调用方传 Pydantic 模型，拿回该模型的实例。
   校验失败由客户端负责（修复重试后仍失败则抛 SchemaValidationError），
   绝不把"看起来差不多"的半成品交给下游。
2. **如实标注来源**：每次调用都返回 `LlmUsage`，明确写着 provider / model /
   是否模拟 / token 数 / 耗时 / 状态。报告与界面据此标注结果来自哪里。
3. **离线实现必须自曝身份**：`is_simulated` 默认 True ——
   宁可多标一句"这是模拟的"，也不能让人误以为真的调用了大模型。
4. **预算可控**：`CallBudget` 限制单次检测的调用次数，防止成本失控。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

from pydantic import BaseModel

from verimind.claims.extractor import ExtractionResult, drafts_to_claims
from verimind.core.config import Settings
from verimind.core.enums import LlmCallStatus, RoleName
from verimind.core.errors import LLMBudgetExceeded, LLMError
from verimind.core.schemas import (
    Claim,
    ClaimExtraction,
    FactCheckReport,
    LlmCallLog,
    Span,
)
from verimind.models.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_fact_check_prompt,
)

TModel = TypeVar("TModel", bound=BaseModel)

#: 不需要密钥就能用的 provider
_KEYLESS_PROVIDERS = frozenset({"mock", "ollama"})


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


class CallBudget:
    """单次检测的模型调用预算（防止一次检测烧掉太多钱）。"""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.used = 0
        self._roles: Counter[str] = Counter()

    def consume(self, role: RoleName | None = None) -> None:
        if self.used >= self.limit:
            raise LLMBudgetExceeded(
                f"本次检测的模型调用次数已达上限（{self.limit} 次），为避免成本失控已停止调用。"
                "可在 .env 中调整 VERIMIND_LLM_MAX_CALLS_PER_RUN。"
            )
        self.used += 1
        if role is not None:
            self._roles[str(role.value)] += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def summary(self) -> str:
        detail = "、".join(f"{role}×{count}" for role, count in self._roles.items())
        return f"共 {self.used}/{self.limit} 次" + (f"（{detail}）" if detail else "")


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
        真实模型实现返回 None，由代码按引用文本自动定位（模型只负责给引用原文）。
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

    def assess_claims(
        self, text: str, claims: list[Claim]
    ) -> tuple[FactCheckReport, LlmUsage]:
        """事实审查员（角色 A）：逐条审查已经拆解出来的主张。

        注意：审查员的结论**不直接决定**主张的对错，
        它只提供"怀疑方向与还要问什么"，最终状态仍由规则裁决层给出。
        离线规则引擎不支持这个角色，会抛 LLMUnsupportedRoleError（不会编结果）。
        """
        prompt = build_fact_check_prompt(text, claims)
        return self.complete_structured(
            role=RoleName.FACT_CHECKER,
            prompt=prompt,
            schema=FactCheckReport,
            system=FACT_CHECKER_SYSTEM_PROMPT,
            context={"text": text, "claims": [claim.model_dump(mode="json") for claim in claims]},
        )

    def health(self) -> tuple[bool, str]:
        """可用性自检（界面上可以显示"当前引擎是否可用"）。"""
        return True, "ok"


# ==========================================================================
# 工厂
# ==========================================================================
def build_client(
    settings: Settings,
    *,
    cache: Any = None,
    budget: CallBudget | None = None,
    recorder: Callable[[LlmCallLog], None] | None = None,
    run_id: str | None = None,
) -> LLMClient:
    """根据配置构造客户端。

    - `provider=mock` → 离线规则引擎（不联网、零成本）；
    - 其他 provider 且填了密钥 → OpenAI 兼容客户端（真实调用）；
    - 选了需要密钥的 provider 却没填密钥 → **明确报错**，不偷偷降级。
    """
    provider = (settings.llm_provider or "").strip().lower()
    if provider in ("", "mock"):
        from verimind.models.mock_provider import RuleBasedOfflineClient

        return RuleBasedOfflineClient()

    if provider not in _KEYLESS_PROVIDERS and settings.missing_credentials:
        raise LLMError(
            f"已选择 provider={provider}，但没有配置 VERIMIND_LLM_API_KEY。\n"
            "请在 .env 中填写密钥；或把 VERIMIND_LLM_PROVIDER 改回 mock 使用离线规则引擎。"
        )

    if not (settings.llm_base_url or "").strip():
        raise LLMError(
            f"provider={provider} 需要设置 VERIMIND_LLM_BASE_URL（OpenAI 兼容接口地址）。"
        )

    from verimind.models.openai_compat import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider=provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout_seconds=float(settings.llm_timeout_seconds),
        max_retries=settings.llm_max_retries,
        cache=cache,
        budget=budget or CallBudget(limit=settings.llm_max_calls_per_run),
        recorder=recorder,
        run_id=run_id,
    )
