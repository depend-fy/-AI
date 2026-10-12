"""检索用的 HTTP 底座：限速 + 缓存 + 原始响应留档 + 可重试。

为什么检索也要这么多讲究？
    1. **礼貌**：Crossref / OpenAlex / arXiv 都是免费公共服务。请求太猛会被限流甚至封禁。
       本项目按来源限速（默认 1 秒 1 次）、带上真实的 mailto（进入"礼貌池"）、设置合理超时。
    2. **可复现**：所有原始响应都留档到 `data/raw/<run_id>/`，报告里的每条证据都能回到原始数据。
       这也是"证据必须来自真实响应"这条约束的落地方式 —— 测试会断言证据 URL 出现在留档里。
    3. **省钱省时**：检索结果按 URL+参数哈希缓存（默认 24 小时），重复运行不重复打接口。
    4. **404 不是错误**：DOI 查不到是一个**有意义的结果**（说明该 DOI 可能未注册），
       必须和"网络挂了"区分开 —— 前者可以据实判断，后者只能说"暂无法核验"。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx

from verimind.core.config import Settings
from verimind.core.enums import EvidenceSource
from verimind.core.logging import get_logger

logger = get_logger("retrieval.http")


@dataclass(frozen=True)
class FetchResult:
    """一次 HTTP 抓取的结果（成功、404、失败都算"结果"，都要如实记录）。"""

    url: str
    status_code: int | None
    text: str = ""
    raw_ref: str | None = None
    from_cache: bool = False
    error: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300

    @property
    def not_found(self) -> bool:
        """服务端明确回答"没有这个东西"（DOI 未注册 / 文献不存在）。"""
        return self.status_code == 404

    @property
    def failed(self) -> bool:
        """网络或服务端故障 —— 这种情况**不能**据此判断文献不存在。"""
        return self.error is not None or (self.status_code is not None and self.status_code >= 500)

    def json(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.text)
        except (json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None


class RateLimiter:
    """按来源（主机）限速，避免把公共接口打爆。"""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval = max(0.0, min_interval_seconds)
        self._last_call: dict[str, float] = {}

    def wait(self, key: str) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        last = self._last_call.get(key)
        if last is not None:
            elapsed = now - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last_call[key] = time.monotonic()


@dataclass
class RetrievalStats:
    """本次运行的检索统计（会写进报告，让用户知道系统真的去查了）。"""

    requests: int = 0
    cache_hits: int = 0
    not_found: int = 0
    failures: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        sources = "、".join(f"{name} {count} 次" for name, count in sorted(self.by_source.items()))
        return (
            f"实际请求 {self.requests} 次（缓存命中 {self.cache_hits} 次）"
            f"，未命中 {self.not_found} 次，失败 {self.failures} 次"
            + (f"；来源：{sources}" if sources else "")
        )


class RetrievalHttpClient:
    """带限速、缓存与留档的检索客户端。"""

    def __init__(
        self,
        settings: Settings,
        *,
        run_id: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.stats = RetrievalStats()
        self._limiter = RateLimiter(settings.retrieval_min_interval_seconds)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=float(settings.retrieval_timeout_seconds),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    f"VeriMind-Demo/{settings.app_version} "
                    f"(mailto:{settings.retrieval_mailto})"
                ),
                "Accept": "application/json, application/atom+xml, */*",
            },
        )
        self._cache_dir = settings.data_dir / "cache" / "retrieval"
        self._raw_dir = settings.data_dir / "raw" / run_id
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RetrievalHttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def build_url(base: str, path: str, params: Mapping[str, Any] | None = None) -> str:
        url = base.rstrip("/") + path
        if params:
            filtered = {key: value for key, value in params.items() if value not in (None, "")}
            if filtered:
                url = f"{url}?{urlencode(filtered)}"
        return url

    def _cache_path(self, url: str) -> Path:
        return self._cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def _read_cache(self, url: str) -> FetchResult | None:
        if self.settings.retrieval_cache_hours <= 0:
            return None
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        created_at = payload.get("created_at")
        try:
            created = datetime.fromisoformat(created_at)
        except (TypeError, ValueError):
            return None
        age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        if age_hours > self.settings.retrieval_cache_hours:
            return None
        return FetchResult(
            url=url,
            status_code=payload.get("status_code"),
            text=payload.get("text") or "",
            raw_ref=payload.get("raw_ref"),
            from_cache=True,
            error=payload.get("error"),
        )

    def _write_cache(self, url: str, result: FetchResult) -> None:
        if self.settings.retrieval_cache_hours <= 0:
            return
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status_code": result.status_code,
            "text": result.text,
            "raw_ref": result.raw_ref,
            "error": result.error,
        }
        try:
            self._cache_path(url).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("写入检索缓存失败：%s", exc)

    def _archive(self, url: str, text: str, suffix: str) -> str | None:
        """把原始响应留档，返回相对于 data/ 的路径（便于报告里追溯）。"""
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        path = self._raw_dir / f"{digest}{suffix}"
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:  # pragma: no cover
            logger.warning("留档原始响应失败：%s", exc)
            return None
        try:
            return str(path.relative_to(self.settings.data_dir))
        except ValueError:  # pragma: no cover
            return str(path)

    # ------------------------------------------------------------------
    def fetch(
        self,
        url: str,
        *,
        source: EvidenceSource,
        params: Mapping[str, Any] | None = None,
        suffix: str = ".json",
    ) -> FetchResult:
        """抓取一个 URL。404 会被如实返回（不是错误）。"""
        full_url = self.build_url(url, "", params)

        cached = self._read_cache(full_url)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        self._limiter.wait(source.value)
        self.stats.requests += 1
        self.stats.by_source[source.value] = self.stats.by_source.get(source.value, 0) + 1

        attempts = 0
        last_error: str | None = None
        for attempt in range(self.settings.retrieval_max_retries + 1):
            attempts = attempt + 1
            try:
                response = self._client.get(full_url)
            except Exception as exc:  # noqa: BLE001 - 网络异常都要变成可读信息
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("检索失败（%s，第 %d 次）：%s", source.value, attempts, last_error)
                if attempt < self.settings.retrieval_max_retries:
                    time.sleep(0.8 * (2**attempt))
                continue

            if response.status_code == 404:
                # 明确回答"没有这个东西"：这是有意义的结果，不重试
                self.stats.not_found += 1
                result = FetchResult(
                    url=full_url,
                    status_code=404,
                    text=response.text[:2000],
                    raw_ref=self._archive(full_url, response.text[:2000], suffix),
                    attempts=attempts,
                )
                self._write_cache(full_url, result)
                return result

            if response.status_code >= 500 or response.status_code == 429:
                last_error = f"HTTP {response.status_code}"
                logger.warning("检索被限流或服务端故障（%s）：%s", source.value, last_error)
                if attempt < self.settings.retrieval_max_retries:
                    time.sleep(0.8 * (2**attempt))
                continue

            result = FetchResult(
                url=full_url,
                status_code=response.status_code,
                text=response.text,
                raw_ref=self._archive(full_url, response.text, suffix),
                attempts=attempts,
            )
            if result.ok:
                self._write_cache(full_url, result)
            else:
                last_error = f"HTTP {response.status_code}"
            return result

        self.stats.failures += 1
        return FetchResult(url=full_url, status_code=None, error=last_error, attempts=attempts)
