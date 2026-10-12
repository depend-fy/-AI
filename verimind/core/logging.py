"""日志与密钥脱敏。

两条规矩：
    1. **密钥永远不进日志**。所有日志都要过一遍脱敏过滤器（`RedactingFilter`），
       把 `sk-...`、`Bearer xxx`、`api_key=xxx` 这类模式替换成 `***`。
    2. **默认不记录完整提示词**。提示词里含有用户原文，属于用户数据。
       只记录提示词的哈希（可用来判断"两次调用是不是同样的输入"），
       需要排查问题时再用显式开关打开完整记录。
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys

# ==========================================================================
# 脱敏
# ==========================================================================
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)\b(api[_-]?key|apikey|token|secret)\b\s*[=:]\s*[^\s,;\"']+"),
)


def redact_secrets(text: str) -> str:
    """把疑似密钥的片段替换成 ***。用于日志与错误信息。"""
    if not text:
        return text
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: match.group(0).split("=")[0].split(":")[0] + "=***"
            if ("=" in match.group(0) or ":" in match.group(0))
            else "***",
            redacted,
        )
    return redacted


def hash_prompt(text: str) -> str:
    """提示词哈希（取代记录完整提示词）。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ==========================================================================
# 日志
# ==========================================================================
class RedactingFilter(logging.Filter):
    """给日志记录做脱敏：消息和参数都过一遍。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            record.msg = redact_secrets(str(record.msg))
            if record.args:
                record.args = tuple(
                    redact_secrets(str(arg)) if isinstance(arg, str) else arg
                    for arg in (record.args if isinstance(record.args, tuple) else (record.args,))
                )
        except Exception:  # pragma: no cover - 日志本身绝不能抛异常
            pass
        return True


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """初始化日志（重复调用不会重复添加 handler）。"""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    handler.addFilter(RedactingFilter())

    root = logging.getLogger("verimind")
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """取一个本项目命名空间下的 logger。"""
    setup_logging()
    return logging.getLogger(f"verimind.{name}" if not name.startswith("verimind") else name)
