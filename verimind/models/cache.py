"""大模型结果缓存（按内容哈希）。

为什么需要：
    1. **省钱**：同一段文本、同一个模型、同一版提示词，第二次运行结果一样，没必要再付一次钱；
    2. **可复现**：演示时把示例先跑一遍，之后即使断网也能复现出完全一样的结果；
    3. **可控**：缓存键里包含提示词版本号，改了提示词就会自动失效，不会拿旧结果冒充新结果。

存储格式：`data/cache/llm/<哈希>.json`
    {"key":..., "created_at":..., "meta":{...}, "raw":"模型返回的原始文本"}
只存**原始文本**（不存解析后的对象），这样缓存文件和当时的解析逻辑解耦，
将来解析规则变了也能看出来是缓存内容而不是解析出错。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from verimind.core.logging import get_logger

logger = get_logger("models.cache")


class LLMCache:
    """磁盘缓存：键是内容哈希，值是模型原始返回文本。"""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    @staticmethod
    def build_key(
        *,
        role: str,
        provider: str,
        model: str,
        prompt: str,
        schema_name: str,
        prompt_version: str,
    ) -> str:
        payload = "\n".join(
            ["v1", role, provider, model, schema_name, prompt_version, prompt]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    # ------------------------------------------------------------------
    def get(self, key: str) -> str | None:
        """命中则返回原始文本，否则 None。"""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("raw")
        except (OSError, json.JSONDecodeError):
            logger.warning("缓存文件损坏，已忽略：%s", path.name)
            self._misses += 1
            return None
        if not isinstance(raw, str):
            self._misses += 1
            return None
        self._hits += 1
        return raw

    def put(self, key: str, raw: str, *, meta: dict[str, object] | None = None) -> None:
        if not self.enabled:
            return
        payload = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
            "raw": raw,
        }
        try:
            self._path(key).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - 磁盘问题
            logger.warning("写入缓存失败（不影响本次结果）：%s", exc)

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def clear(self) -> int:
        """清空缓存，返回删除的文件数。"""
        if not self.directory.exists():
            return 0
        count = 0
        for path in self.directory.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError:  # pragma: no cover
                continue
        self._hits = self._misses = 0
        return count

    def size(self) -> int:
        if not self.directory.exists():
            return 0
        return sum(1 for _ in self.directory.glob("*.json"))
