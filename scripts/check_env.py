"""环境自检脚本：一条命令告诉你"这台电脑现在能不能跑这个项目"。

用法（在项目根目录、用项目自己的 venv 运行）：
    .\\.venv\\Scripts\\python.exe scripts\\check_env.py
    .\\.venv\\Scripts\\python.exe scripts\\check_env.py --network   # 额外测试外部 API 连通性

它会检查：
    1. Python 版本与解释器位置（是否在 venv 里）
    2. 必需依赖是否装齐、版本号
    3. 配置能否加载、当前运行模式是什么、密钥是否配置（只显示脱敏值）
    4. 数据目录能否创建
    5. （可选 --network）Crossref / OpenAlex / arXiv 与大模型端点是否连得通

退出码：0 = 核心检查全通过；1 = 有阻塞性问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接以脚本方式运行：把项目根目录加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OK = "  [ OK ]"
WARN = "  [WARN]"
FAIL = "  [FAIL]"

#: 依赖名 -> pip 包名（import 名与包名不一致的在这里映射）
REQUIRED_PACKAGES: dict[str, str] = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "httpx": "httpx",
    "openai": "openai",
    "pytest": "pytest",
    "respx": "respx",
}


def check_python() -> bool:
    print("\n[1/5] Python 环境")
    major, minor = sys.version_info[:2]
    print(f"{OK} Python 版本：{sys.version.split()[0]}（{sys.executable}）")
    if (major, minor) < (3, 11):
        print(f"{FAIL} 需要 Python 3.11 或更高版本。")
        return False
    if (major, minor) > (3, 13):
        print(f"{WARN} Python {major}.{minor} 比较新，个别第三方库可能还没有对应安装包。")
        print("        如果后面装依赖失败，改用 3.13 创建虚拟环境：py -3.13 -m venv .venv")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        print(f"{OK} 正在使用虚拟环境：{sys.prefix}")
    else:
        print(f"{WARN} 当前不在虚拟环境里。建议用项目自带的 .venv 运行，避免污染全局 Python。")
    return True


def check_packages() -> bool:
    print("\n[2/5] 依赖包")
    all_ok = True
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "未知版本")
            print(f"{OK} {package_name:<20} {version}")
        except ImportError:
            all_ok = False
            print(f"{FAIL} {package_name:<20} 未安装 -> 运行：python -m pip install -r requirements.txt")
    return all_ok


def check_config() -> bool:
    print("\n[3/5] 配置与运行模式")
    try:
        from verimind.core.config import Settings
    except Exception as exc:  # noqa: BLE001 - 自检脚本要把任何异常都变成可读信息
        print(f"{FAIL} 无法导入配置模块：{exc}")
        return False

    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 配置校验失败：{exc}")
        return False

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print(f"{OK} 已找到 .env 配置文件")
    else:
        print(f"{WARN} 没有 .env 文件（这不影响运行，会自动用默认的离线模拟模式）")
        print("        需要自定义配置时执行：Copy-Item .env.example .env")

    print(f"{OK} provider      : {settings.llm_provider}")
    print(f"{OK} model         : {settings.llm_model}")
    print(f"{OK} API Key       : {settings.masked_api_key}（仅显示脱敏值）")
    print(f"{OK} 运行模式      : {settings.run_mode.value}")
    for message in settings.runtime_warnings:
        print(f"{WARN} {message}")
    return True


def check_data_dirs() -> bool:
    print("\n[4/5] 数据目录")
    try:
        from verimind.core.config import Settings

        paths = Settings().ensure_data_dirs()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 无法创建数据目录：{exc}")
        return False
    for name, path in paths.items():
        print(f"{OK} {name:<6} {path}")
    return True


def check_network() -> bool:
    """可选：真实访问外部服务。只发一个最小请求，不消耗大模型额度。"""
    print("\n[5/5] 外部服务连通性（--network）")
    targets = {
        "Crossref": "https://api.crossref.org/works/10.1038/nature12373",
        "OpenAlex": "https://api.openalex.org/works/https://doi.org/10.1038/nature12373",
        "arXiv": "http://export.arxiv.org/api/query?search_query=all:electron&max_results=1",
    }
    try:
        import httpx
    except ImportError:
        print(f"{FAIL} httpx 未安装，无法测试网络。")
        return False

    from verimind.core.config import Settings

    settings = Settings()
    if settings.llm_base_url:
        targets["大模型端点"] = settings.llm_base_url

    headers = {"User-Agent": f"VeriMind-Demo/{settings.app_version} (mailto:{settings.retrieval_mailto})"}
    all_ok = True
    for name, url in targets.items():
        try:
            response = httpx.get(url, headers=headers, timeout=15.0)
            note = ""
            if name == "大模型端点" and response.status_code in (401, 403):
                note = "（可达，但未授权 = 还没填 API Key，属正常）"
            print(f"{OK} {name:<10} HTTP {response.status_code}{note}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"{FAIL} {name:<10} 连接失败：{type(exc).__name__}: {exc}")
    return all_ok


def check_offline_skipped() -> None:
    print("\n[5/5] 外部服务连通性：已跳过（加 --network 参数可测试）")


def main() -> int:
    print("=" * 68)
    print("智鉴 AI（VeriMind AI）环境自检")
    print("=" * 68)

    results = [check_python(), check_packages(), check_config(), check_data_dirs()]
    if "--network" in sys.argv:
        results.append(check_network())
    else:
        check_offline_skipped()

    print("\n" + "=" * 68)
    if all(results):
        print("结论：核心检查全部通过，可以运行：streamlit run app.py")
        print("=" * 68)
        return 0
    print("结论：存在阻塞性问题，请按上面的 [FAIL] 提示处理后重试。")
    print("=" * 68)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
