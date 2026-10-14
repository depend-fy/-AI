"""配置管理：全项目所有可变参数集中在这里。

设计原则：
    1. 代码里不出现任何硬编码的密钥、邮箱、模型名 —— 一律从 .env / 环境变量读。
    2. 缺少密钥不能崩：默认 provider=mock，会自动进入"离线模拟"模式，
       界面上如实标注，功能照常演示（阶段 2 起）。
    3. 所有变量都以 VERIMIND_ 开头（env_prefix），避免和电脑上其他工具的变量重名。
    4. 提供一个"人可读的环境诊断"，让界面能如实告诉用户当前到底在什么模式下运行。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from verimind.core.enums import RunMode

#: 项目根目录（本文件位于 <项目根>/verimind/core/config.py）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 不需要密钥就能跑的 provider
_KEYLESS_PROVIDERS = frozenset({"mock", "ollama"})


class Settings(BaseSettings):
    """应用配置。字段名小写，对应环境变量 VERIMIND_<字段名大写>。"""

    model_config = SettingsConfigDict(
        env_prefix="VERIMIND_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env 里多写的变量不会导致启动失败
    )

    # ---------------- 应用信息 ----------------
    app_name: str = "智鉴 AI（VeriMind AI）"
    app_version: str = "0.1.0"
    pipeline_version: str = "0.1.0"

    # ---------------- 大模型（阶段 3 启用） ----------------
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""

    # 独立复核员（可选）。三项都填且与主模型不同 → 升级为"多模型协同"
    llm_reviewer_provider: str = ""
    llm_reviewer_base_url: str = ""
    llm_reviewer_model: str = ""
    llm_reviewer_api_key: str = ""

    llm_timeout_seconds: int = Field(default=60, ge=5, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_cache_enabled: bool = True
    llm_max_calls_per_run: int = Field(default=30, ge=1, le=500)

    # 计价（用于成本估算）。**故意留成字符串**：没填就是空，报告里显示"未计算"，
    # 绝不编一个单价出来。单价请到供应商官网查，格式示例：
    #   VERIMIND_LLM_PRICE_PROMPT_PER_MTOK=2      # 每百万 prompt token 多少元
    #   VERIMIND_LLM_PRICE_COMPLETION_PER_MTOK=8  # 每百万 completion token 多少元
    llm_price_prompt_per_mtok: str = ""
    llm_price_completion_per_mtok: str = ""

    # ---------------- 文献检索（免密钥） ----------------
    retrieve_enabled: bool = True
    retrieval_mailto: str = "verimind-demo@example.com"
    retrieval_timeout_seconds: int = Field(default=20, ge=3, le=120)
    retrieval_min_interval_seconds: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_cache_hours: int = Field(default=24, ge=0, le=24 * 30)
    retrieval_max_queries_per_run: int = Field(default=12, ge=1, le=100)
    #: 是否使用 Wikidata / Wikipedia（非学术事实与实体核验；均为免密钥公开接口）
    retrieval_web_sources_enabled: bool = True
    retrieval_max_retries: int = Field(default=1, ge=0, le=3)

    # ---------------- 输入与流程限制 ----------------
    max_input_chars: int = Field(default=3000, ge=100, le=20000)
    max_claims: int = Field(default=8, ge=1, le=50)

    # ---------------- 存储 ----------------
    save_runs: bool = True
    data_dir: Path = PROJECT_ROOT / "data"

    # ================= 派生信息（只读，不进配置文件） =================

    @property
    def has_llm_key(self) -> bool:
        """当前 provider 是否具备可用的凭据。"""
        return bool(self.llm_api_key.strip()) or self.llm_provider.lower() in _KEYLESS_PROVIDERS

    @property
    def missing_credentials(self) -> bool:
        """选了需要密钥的 provider，但密钥是空的。"""
        return self.llm_provider.lower() not in _KEYLESS_PROVIDERS and not self.llm_api_key.strip()

    @property
    def reviewer_is_distinct(self) -> bool:
        """独立复核员是否配置成了一个"真的不同"的模型。

        只要 provider 或 model 有一个与主模型不同就算不同——这意味着：
            · 换一家厂商（最独立）；
            · 或者同一家的另一个模型（例如 deepseek-chat → deepseek-reasoner）。
        同一家的另一个模型可以**复用主密钥**，不必重复填一遍。
        """
        provider = (self.llm_reviewer_provider or "").strip().lower()
        model = (self.llm_reviewer_model or "").strip()
        if not provider or not model:
            return False

        main_provider = self.llm_provider.strip().lower()
        different = provider != main_provider or model != self.llm_model
        if not different:
            return False  # 和主模型完全一样，那就不叫独立了

        if provider in _KEYLESS_PROVIDERS:
            return True
        if self.llm_reviewer_api_key.strip():
            return True
        # 同一家厂商 → 复用主模型密钥；跨厂商则必须单独提供密钥
        return provider == main_provider and bool(self.llm_api_key.strip())

    @property
    def run_mode(self) -> RunMode:
        """本次运行的真实模式。界面必须按这个值如实标注，不允许含糊。"""
        if self.llm_provider.lower() == "mock" or self.missing_credentials:
            return RunMode.MOCK
        if self.reviewer_is_distinct:
            return RunMode.MULTI_PROVIDER
        return RunMode.SINGLE_MODEL_MULTI_ROLE

    @property
    def masked_api_key(self) -> str:
        """只用于界面展示的脱敏密钥，绝不返回完整值。"""
        key = self.llm_api_key.strip()
        if not key:
            return "未配置"
        if len(key) <= 8:
            return "已配置（长度过短，建议检查）"
        return f"{key[:3]}****{key[-4:]}"

    @property
    def runtime_warnings(self) -> list[str]:
        """当前配置下用户必须知道的事（界面上直接显示）。"""
        warnings: list[str] = []
        if self.llm_provider.lower() == "mock":
            warnings.append("当前为「离线模拟」模式：不会调用任何真实大模型，结果仅供流程演示。")
        elif self.missing_credentials:
            warnings.append(
                f"已选择 provider={self.llm_provider} 但没有配置 VERIMIND_LLM_API_KEY，"
                "系统已自动降级为「离线模拟」模式。"
            )
        if 0 < self.llm_max_calls_per_run < 20:
            warnings.append(f"单次检测的模型调用上限被设为 {self.llm_max_calls_per_run} 次，超限部分会被降级处理。")
        return warnings

    def llm_config_summary(self) -> dict[str, dict[str, str]]:
        """写进报告的角色→模型映射（绝不含密钥）。"""
        reviewer_provider = (self.llm_reviewer_provider or self.llm_provider).strip()
        reviewer_model = (self.llm_reviewer_model or self.llm_model).strip()
        reviewer_base = (self.llm_reviewer_base_url or self.llm_base_url).strip()
        main = {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "base_url": self.llm_base_url,
            "api_key": self.masked_api_key,
        }
        reviewer = {
            "provider": reviewer_provider,
            "model": reviewer_model,
            "base_url": reviewer_base,
            "api_key": self.masked_reviewer_api_key,
        }
        return {
            "claim_extractor": dict(main),
            "fact_checker": dict(main),
            "evidence_checker": dict(main),
            "corrector": dict(main),
            "independent_reviewer": reviewer,
        }

    @property
    def masked_reviewer_api_key(self) -> str:
        key = self.llm_reviewer_api_key.strip()
        if not key:
            return self.masked_api_key
        if len(key) <= 8:
            return "已配置（长度过短，建议检查）"
        return f"{key[:3]}****{key[-4:]}"

    @property
    def token_prices(self) -> tuple[float, float] | None:
        """返回 (每百万 prompt token 单价, 每百万 completion token 单价)；未配置则返回 None。

        没有单价就**不计算成本**——报告里会写"未计算"并说明怎么配置，
        而不是拿一个编造的单价算出看起来很专业的数字。
        """
        try:
            prompt_price = float(self.llm_price_prompt_per_mtok)
            completion_price = float(self.llm_price_completion_per_mtok)
        except (TypeError, ValueError):
            return None
        if prompt_price < 0 or completion_price < 0:
            return None
        if prompt_price == 0 and completion_price == 0:
            return None
        return prompt_price, completion_price

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """按配置的单价估算成本（单位与单价一致，通常是人民币元）。"""
        prices = self.token_prices
        if prices is None:
            return None
        prompt_price, completion_price = prices
        return round(
            prompt_tokens / 1_000_000 * prompt_price
            + completion_tokens / 1_000_000 * completion_price,
            6,
        )

    def ensure_data_dirs(self) -> dict[str, Path]:
        """创建本地数据目录，返回路径表（首次运行会自动建）。"""
        paths = {
            "data": self.data_dir,
            "raw": self.data_dir / "raw",
            "cache": self.data_dir / "cache",
            "runs": self.data_dir / "runs",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "verimind.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。改了 .env 后需要重启 Streamlit 才会生效。"""
    return Settings()


def reload_settings() -> Settings:
    """清掉缓存重新读取（供界面上的"重新加载配置"按钮使用）。"""
    get_settings.cache_clear()
    return get_settings()
