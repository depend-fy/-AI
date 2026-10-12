"""离线规则引擎客户端：不联网、不花钱、结果完全确定。

它的定位必须说清楚（界面和报告里都会照实写）：
    **这不是大模型。** 它不会"理解"文本，只是按正则和关键词找线索。
    它的价值是：① 让整条流水线在没有任何 API 密钥时也能完整跑通并演示；
    ② 作为确定性的基线（评测页里的 B1 基线就是它）；
    ③ 阶段 3 接入真实模型后，用来与模型结果交叉比对。

它会拒绝自己做不到的事：
    除了"主张拆解"，其他角色（事实审查员 / 证据审查员 / 独立复核员）都会抛
    `LLMUnsupportedRoleError`，而不是编一个像模像样的假结论。
"""

from __future__ import annotations

import time
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel

from verimind.claims.extractor import (
    DEFAULT_EXTRACTOR,
    collect_drafts,
)
from verimind.core.enums import LlmCallStatus, RoleName
from verimind.core.errors import LLMError, LLMUnsupportedRoleError
from verimind.core.schemas import ClaimExtraction, Span
from verimind.models.base import LLMClient, LlmUsage

TModel = TypeVar("TModel", bound=BaseModel)

_NO_CLAIM_NOTE = "规则引擎未能从文本中识别出可核验的事实主张（可能整段都是观点或过渡性描述）。"


class RuleBasedOfflineClient(LLMClient):
    """基于正则与关键词的离线"模型"替身。"""

    provider = "mock-rule-engine"
    model = f"regex-v{DEFAULT_EXTRACTOR.version}"

    @property
    def is_simulated(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def complete_structured(
        self,
        *,
        role: RoleName,
        prompt: str,
        schema: type[TModel],
        system: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> tuple[TModel, LlmUsage]:
        if role is not RoleName.CLAIM_EXTRACTOR:
            raise LLMUnsupportedRoleError(
                f"离线规则引擎只实现了「{RoleName.CLAIM_EXTRACTOR.value}」这一项能力，"
                f"无法承担「{role.value}」。该角色需要真实大模型，将在阶段 3 接入——"
                "这里不会用假结果顶替。"
            )
        if schema is not ClaimExtraction:
            raise LLMError(f"离线规则引擎只产出 ClaimExtraction，收到的是 {schema.__name__}。")

        text = str((context or {}).get("text") or "")
        if not text.strip():
            raise LLMError("离线规则引擎需要 context['text']（待拆解的原文）才能工作。")

        started = time.perf_counter()
        drafts, _, _ = collect_drafts(text)
        latency_ms = int((time.perf_counter() - started) * 1000)

        extraction = ClaimExtraction(
            claims=drafts,
            notes=[] if drafts else [_NO_CLAIM_NOTE],
        )
        usage = LlmUsage(
            provider=self.provider,
            model=self.model,
            role=role,
            prompt_tokens=None,  # 规则引擎不消耗 token —— 如实标注为 None，不编造数字
            completion_tokens=None,
            latency_ms=latency_ms,
            status=LlmCallStatus.OK,
            simulated=True,
        )
        return extraction, usage  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def span_hints(self, text: str, extraction: ClaimExtraction) -> list[Span | None] | None:
        """规则引擎自己知道精确位置，直接告诉下游，避免重复片段被定位到错处。"""
        _, hints, _ = collect_drafts(text)
        if len(hints) != len(extraction.claims):
            return None
        return list(hints)

    def health(self) -> tuple[bool, str]:
        return True, "离线规则引擎可用（不联网、不消耗额度）"
