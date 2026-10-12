"""本地存储：把每次检测的完整报告存进 SQLite，支持"重开页面还能看到上次结果"。

为什么用 SQLite 而不是一堆 JSON 文件？
    · Python 标准库自带（sqlite3），零依赖、零安装、单文件；
    · 历史列表、按 ID 查询、删除都很方便，不用手写索引；
    · 报告同时保留完整 JSON 字段，需要时可以原样导出。

存储位置：`data/verimind.db`（已被 .gitignore 排除，不会进 Git）。
注意：报告里含有用户输入的原文，属于本地数据，不上传任何地方。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from verimind.core.enums import LlmCallStatus, RoleName
from verimind.core.errors import StorageError
from verimind.core.schemas import DetectionReport, LlmCallLog

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    run_mode          TEXT NOT NULL,
    pipeline_version  TEXT,
    input_sha256      TEXT,
    input_preview     TEXT,
    total_claims      INTEGER NOT NULL DEFAULT 0,
    suspicious_claims INTEGER NOT NULL DEFAULT 0,
    flagged_claims    INTEGER NOT NULL DEFAULT 0,
    status_counts     TEXT,
    report_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);

-- 大模型调用日志：用于成本核算与问题排查。
-- 刻意**不存完整提示词**（提示词里有用户原文），只存哈希。
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id           TEXT PRIMARY KEY,
    run_id            TEXT,
    role              TEXT,
    provider          TEXT,
    model             TEXT,
    prompt_hash       TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    status            TEXT,
    error             TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls (created_at DESC);
"""

#: 建表之后新增的列：老数据库文件需要补列（简单迁移，避免"升级后打不开旧数据"）
_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("flagged_claims", "INTEGER NOT NULL DEFAULT 0"),
)


@dataclass(frozen=True)
class RunSummary:
    """历史列表用的摘要（不含完整报告，列表加载更快）。"""

    run_id: str
    created_at: datetime
    run_mode: str
    total_claims: int
    suspicious_claims: int
    flagged_claims: int
    input_preview: str


def _preview(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


class RunRepository:
    """运行记录的读写接口（SQLite 实现）。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """每次操作开一个短连接：避免 Streamlit 多线程重跑时的连接复用问题。"""
        try:
            connection = sqlite3.connect(self.db_path, timeout=10.0)
        except sqlite3.Error as exc:  # pragma: no cover - 环境问题
            raise StorageError(f"无法打开本地数据库 {self.db_path}：{exc}") from exc
        try:
            connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(f"本地数据库操作失败：{exc}") from exc
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            for column, definition in _COLUMN_MIGRATIONS:
                if column not in existing:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {definition}")

    # ------------------------------------------------------------------
    def save(self, report: DetectionReport) -> None:
        """保存（或覆盖）一次检测报告。"""
        stats = report.stats
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at, run_mode, pipeline_version, input_sha256,
                    input_preview, total_claims, suspicious_claims, flagged_claims,
                    status_counts, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.created_at.astimezone(timezone.utc).isoformat(),
                    str(report.run_mode.value),
                    report.pipeline_version,
                    report.input_sha256,
                    _preview(report.input_text),
                    stats.total_claims,
                    stats.suspicious_claims,
                    stats.flagged_claims,
                    json.dumps(stats.status_counts, ensure_ascii=False),
                    report.to_json(indent=None),
                ),
            )

    def get(self, run_id: str) -> DetectionReport | None:
        """按 ID 读取报告；不存在时返回 None。

        如果这条记录是**旧版本**（数据结构已经变化）保存的，会抛 `StorageError` 而不是
        让 pydantic 的校验错误直接冒到界面上——历史数据不该把页面搞崩。
        """
        if not run_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return DetectionReport.from_json(row["report_json"])
        except ValidationError as exc:
            raise StorageError(
                f"这条历史记录（{run_id[:8]}…）是用旧版本的数据结构保存的，"
                "与当前版本不兼容。可以重新检测一次生成新报告；"
                "如需清理历史：删除 data/verimind.db 即可（该文件不在 Git 里）。"
                f"（校验失败原因：{str(exc).splitlines()[0][:120]}）"
            ) from exc

    def list_recent(self, limit: int = 20) -> list[RunSummary]:
        """最近若干次运行的摘要（按时间倒序）。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, created_at, run_mode, total_claims, suspicious_claims,
                       flagged_claims, input_preview
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        summaries: list[RunSummary] = []
        for row in rows:
            try:
                created = datetime.fromisoformat(row["created_at"])
            except (TypeError, ValueError):
                created = datetime.now(timezone.utc)
            summaries.append(
                RunSummary(
                    run_id=row["run_id"],
                    created_at=created,
                    run_mode=row["run_mode"],
                    total_claims=int(row["total_claims"] or 0),
                    suspicious_claims=int(row["suspicious_claims"] or 0),
                    flagged_claims=int(row["flagged_claims"] or 0),
                    input_preview=row["input_preview"] or "",
                )
            )
        return summaries

    def delete(self, run_id: str) -> bool:
        """删除一条记录，返回是否真的删掉了东西。"""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # 大模型调用日志（成本核算与排查用）
    # ------------------------------------------------------------------
    def save_llm_call(self, call: LlmCallLog) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO llm_calls (
                    call_id, run_id, role, provider, model, prompt_hash,
                    prompt_tokens, completion_tokens, latency_ms, status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.call_id,
                    call.run_id,
                    str(call.role.value) if call.role else None,
                    call.provider,
                    call.model,
                    call.prompt_hash,
                    call.prompt_tokens,
                    call.completion_tokens,
                    call.latency_ms,
                    str(call.status.value),
                    call.error,
                    call.created_at.astimezone(timezone.utc).isoformat(),
                ),
            )

    def list_llm_calls(self, run_id: str | None = None, limit: int = 50) -> list[LlmCallLog]:
        query = "SELECT * FROM llm_calls"
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        calls: list[LlmCallLog] = []
        for row in rows:
            calls.append(
                LlmCallLog(
                    call_id=row["call_id"],
                    run_id=row["run_id"],
                    role=RoleName(row["role"]) if row["role"] else None,
                    provider=row["provider"] or "",
                    model=row["model"] or "",
                    prompt_hash=row["prompt_hash"] or "",
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    latency_ms=int(row["latency_ms"] or 0),
                    status=LlmCallStatus(row["status"]) if row["status"] else LlmCallStatus.OK,
                    error=row["error"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return calls

    def usage_summary(self, run_id: str | None = None) -> dict[str, int]:
        """汇总 token 用量（用于评测页的"单次检测成本"指标）。"""
        query = "SELECT COALESCE(SUM(prompt_tokens),0) AS p, COALESCE(SUM(completion_tokens),0) AS c, COUNT(*) AS n FROM llm_calls"
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        return {
            "calls": int(row["n"] or 0),
            "prompt_tokens": int(row["p"] or 0),
            "completion_tokens": int(row["c"] or 0),
        }
