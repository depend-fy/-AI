"""OpenAI 兼容协议客户端：一套代码接 DeepSeek / 智谱 / Kimi / 通义 / OpenAI / 本地 Ollama。

只用 OpenAI 兼容的 `/chat/completions` 接口，换供应商只需要改 `.env` 里的三行
（`VERIMIND_LLM_PROVIDER` / `VERIMIND_LLM_BASE_URL` / `VERIMIND_LLM_MODEL`），代码零改动。

这个文件要解决四个真实世界的问题
--------------------------------
1. **模型不一定听话**：可能返回带 ``` 的代码块、可能夹带解释文字、可能字段残缺。
   → 先容错解析，解析失败**带着错误信息让模型修一次**，再失败就抛 `SchemaValidationError`
     （绝不把半成品交给下游，更不编造缺失字段）。
2. **网络会抖**：超时、连接失败、429 限流、5xx。
   → 只对**可重试**的错误退避重试；401/402/403/404 这类明确错误立刻失败并给中文建议。
3. **钱要看得见**：每次调用都记录 token、耗时、状态（`LlmUsage`），并由调用方写入数据库。
4. **提示词含用户原文**：日志里只记提示词哈希，不记全文。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Callable, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from verimind.core.enums import LlmCallStatus, RoleName
from verimind.core.errors import (
    LLMBudgetExceeded,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    SchemaValidationError,
)
from verimind.core.logging import get_logger, hash_prompt, redact_secrets
from verimind.core.schemas import LlmCallLog
from verimind.models.base import CallBudget, LLMClient, LlmUsage
from verimind.models.cache import LLMCache
from verimind.models.prompts import PROMPT_VERSION

TModel = TypeVar("TModel", bound=BaseModel)

logger = get_logger("models.openai_compat")

#: 可以重试的 HTTP 状态码（限流与服务端临时故障）
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: 明确不可重试的状态码 -> 中文处理建议
_FATAL_HINTS = {
    401: "密钥无效或已失效，请检查 VERIMIND_LLM_API_KEY（前后不要有空格或引号）。",
    402: "账户余额不足，请到供应商控制台充值。",
    403: "账号没有该模型的访问权限，请确认已开通或更换模型名。",
    404: "模型名或接口地址不正确，请核对 VERIMIND_LLM_MODEL 与 VERIMIND_LLM_BASE_URL。",
}


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型返回的文本里尽力取出一个 JSON 对象。

    能处理：纯 JSON、```json 包裹、JSON 前后夹带解释文字。
    取不到就抛 ValueError —— 由调用方决定是"让模型重写"还是"报错"。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("模型返回了空内容")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型返回的内容里找不到 JSON 对象") from None
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"期望 JSON 对象，实际得到 {type(parsed).__name__}")
    return parsed


class OpenAICompatibleClient(LLMClient):
    """基于 OpenAI 兼容接口的真实模型客户端。"""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        cache: LLMCache | None = None,
        budget: CallBudget | None = None,
        recorder: Callable[[LlmCallLog], None] | None = None,
        run_id: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._cache = cache
        self._budget = budget or CallBudget(limit=10**6)
        self._recorder = recorder
        self._run_id = run_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._json_mode_supported = True
        self._client = self._build_sdk_client()

    # ------------------------------------------------------------------
    def _build_sdk_client(self):  # noqa: ANN202 - 返回 openai.OpenAI
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "缺少 openai 库，请运行：python -m pip install -r requirements.txt"
            ) from exc
        return OpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self._timeout,
            max_retries=0,  # 重试策略由本类自己控制，方便记录每次尝试
        )

    @property
    def is_simulated(self) -> bool:
        return False

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
        started = time.perf_counter()
        cache_key = LLMCache.build_key(
            role=str(role.value),
            provider=self.provider,
            model=self.model,
            prompt=f"{system}\n---\n{prompt}",
            schema_name=schema.__name__,
            prompt_version=self._prompt_version,
        )

        # ---------------- ① 先看缓存 ----------------
        if self._cache is not None:
            cached_raw = self._cache.get(cache_key)
            if cached_raw is not None:
                try:
                    instance = schema.model_validate(extract_json_object(cached_raw))
                except (ValueError, ValidationError) as exc:
                    logger.warning("缓存内容无法解析（%s），改为重新调用模型", type(exc).__name__)
                else:
                    logger.info("命中缓存：%s（本次未产生费用）", role.value)
                    return instance, LlmUsage(
                        provider=self.provider,
                        model=self.model,
                        role=role,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status=LlmCallStatus.CACHE_HIT,
                        cache_hit=True,
                        simulated=False,
                    )

        # ---------------- ② 真实调用 ----------------
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        raw, usage = self._call_with_retry(messages, role, schema)
        parsed, parse_error = self._try_parse(raw, schema)

        # ---------------- ③ 格式不对就带着错误让模型修一次 ----------------
        if parsed is None:
            logger.warning("%s 返回格式不合格（%s），发起一次修复重试", role.value, parse_error)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw[:4000]},
                {
                    "role": "user",
                    "content": (
                        "上面的输出不符合要求，解析失败，错误信息：\n"
                        f"{redact_secrets(str(parse_error))}\n\n"
                        "请**只输出**符合下面 JSON Schema 的 JSON 对象，不要任何解释、不要 Markdown 代码块：\n"
                        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                    ),
                },
            ]
            raw2, usage2 = self._call_with_retry(repair_messages, role, schema, repair=True)
            usage = _merge_usage(usage, usage2)
            parsed, parse_error = self._try_parse(raw2, schema)
            if parsed is not None:
                raw = raw2

        if parsed is None:
            self._record(role, prompt, usage, status=LlmCallStatus.ERROR, error=str(parse_error))
            raise SchemaValidationError(
                f"模型（{self.model}）连续两次返回的内容都不符合 {schema.__name__} 结构：{parse_error}"
            )

        # ---------------- ④ 写缓存与日志 ----------------
        if self._cache is not None:
            self._cache.put(
                cache_key,
                raw,
                meta={
                    "role": str(role.value),
                    "provider": self.provider,
                    "model": self.model,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            )
        usage = LlmUsage(
            provider=usage.provider,
            model=usage.model,
            role=role,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=usage.status,
            simulated=False,
        )
        self._record(role, prompt, usage)
        return parsed, usage

    # ------------------------------------------------------------------
    def _try_parse(
        self, raw: str, schema: type[TModel]
    ) -> tuple[TModel | None, Exception | None]:
        try:
            return schema.model_validate(extract_json_object(raw)), None
        except (ValueError, ValidationError) as exc:
            return None, exc

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        role: RoleName,
        schema: type[TModel],
        *,
        repair: bool = False,
    ) -> tuple[str, LlmUsage]:
        """调用接口，带退避重试与预算控制。"""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._budget.consume(role)  # 超预算在这里抛 LLMBudgetExceeded
            started = time.perf_counter()
            response = None
            failure: Exception | None = None
            try:
                response = self._create_completion(messages)
            except Exception as exc:  # noqa: BLE001 - 需要分类处理
                failure = exc
                # 供应商不支持 response_format 时降级重试。
                # 注意：这**不消耗**重试额度——它是能力探测，不是网络故障。
                if (
                    self._json_mode_supported
                    and getattr(exc, "status_code", None) in (400, 404)
                    and "response_format" in str(exc)
                ):
                    logger.warning(
                        "该供应商不支持 response_format=json_object，已降级为纯提示词约束后重试"
                    )
                    self._json_mode_supported = False
                    try:
                        response = self._create_completion(messages)
                        failure = None
                    except Exception as inner:  # noqa: BLE001
                        failure = inner

            if response is not None:
                return self._extract_result(response, role, started)

            assert failure is not None
            latency_ms = int((time.perf_counter() - started) * 1000)
            retryable, message = self._classify(failure)
            last_error = failure
            self._record(
                role,
                _prompt_of(messages),
                LlmUsage(
                    provider=self.provider,
                    model=self.model,
                    role=role,
                    latency_ms=latency_ms,
                    status=LlmCallStatus.ERROR,
                    simulated=False,
                    error=message,
                ),
                status=LlmCallStatus.ERROR,
                error=message,
            )
            if not retryable or attempt == self._max_retries:
                raise self._translate(failure) from failure
            backoff = 0.8 * (2**attempt)
            logger.warning("调用失败（%s），%.1fs 后重试（第 %d 次）", message, backoff, attempt + 1)
            time.sleep(backoff)

        # 只有"每次都可重试且都用完了"才会走到这里
        raise self._translate(last_error) if last_error else LLMError("模型调用失败")

    def _extract_result(
        self, response, role: RoleName, started: float
    ) -> tuple[str, LlmUsage]:  # noqa: ANN001
        """从成功响应里取出文本与用量。"""
        usage = getattr(response, "usage", None)
        return (response.choices[0].message.content or ""), LlmUsage(
            provider=self.provider,
            model=self.model,
            role=role,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=LlmCallStatus.OK,
            simulated=False,
        )

    def _create_completion(self, messages: list[dict[str, str]]):  # noqa: ANN202
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _classify(exc: Exception) -> tuple[bool, str]:
        """判断异常是否可重试，并给出简短描述。"""
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__
        text = str(exc)
        if "timeout" in name.lower() or "timed out" in text.lower():
            return True, "请求超时"
        if "connection" in name.lower():
            return True, "网络连接失败"
        if isinstance(status, int):
            if status in _RETRYABLE_STATUS:
                return True, f"HTTP {status}（服务端临时故障或限流）"
            return False, f"HTTP {status}"
        return False, f"{name}"

    def _translate(self, exc: Exception) -> LLMError:
        """把 SDK 异常翻译成带中文建议的项目异常。"""
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__
        text = redact_secrets(str(exc))
        if "timeout" in name.lower() or "timed out" in text.lower():
            return LLMTimeoutError(
                f"调用 {self.provider}/{self.model} 超时（{self._timeout:.0f}s）。"
                "可调大 VERIMIND_LLM_TIMEOUT_SECONDS，或换用更快的模型。"
            )
        if "connection" in name.lower():
            return LLMConnectionError(
                f"无法连接 {self.base_url}：{text[:200]}。请检查网络、代理与接口地址。"
            )
        if isinstance(status, int) and status in _FATAL_HINTS:
            return LLMError(f"调用 {self.provider}/{self.model} 失败（HTTP {status}）：{_FATAL_HINTS[status]}")
        return LLMError(f"调用 {self.provider}/{self.model} 失败：{text[:300]}")

    def _record(
        self,
        role: RoleName,
        prompt: str,
        usage: LlmUsage,
        *,
        status: LlmCallStatus | None = None,
        error: str | None = None,
    ) -> None:
        """写一条调用日志（不记录完整提示词，只记哈希）。"""
        if self._recorder is None:
            return
        try:
            self._recorder(
                LlmCallLog(
                    call_id=uuid.uuid4().hex,
                    run_id=self._run_id,
                    role=role,
                    provider=self.provider,
                    model=self.model,
                    prompt_hash=hash_prompt(prompt),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_ms=usage.latency_ms,
                    status=status or usage.status,
                    error=error,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 日志失败绝不能影响检测
            logger.warning("写入调用日志失败：%s", exc)

    def health(self) -> tuple[bool, str]:
        return True, f"已配置 {self.provider}/{self.model}（接口 {self.base_url}）"


def _prompt_of(messages: list[dict[str, str]]) -> str:
    return "\n".join(item.get("content", "") for item in messages)


def _merge_usage(first: LlmUsage, second: LlmUsage) -> LlmUsage:
    """把"首次调用 + 修复重试"的用量合并，避免成本被低估。"""
    return LlmUsage(
        provider=second.provider,
        model=second.model,
        role=second.role,
        prompt_tokens=(first.prompt_tokens or 0) + (second.prompt_tokens or 0) or None,
        completion_tokens=(first.completion_tokens or 0) + (second.completion_tokens or 0) or None,
        latency_ms=first.latency_ms + second.latency_ms,
        status=LlmCallStatus.SCHEMA_RETRY,
        simulated=False,
    )


__all__ = ["OpenAICompatibleClient", "extract_json_object", "LLMBudgetExceeded"]
