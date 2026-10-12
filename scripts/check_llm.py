"""验证大模型密钥与连通性（只做一次极小调用，成本可忽略）。

用法（在项目根目录）：
    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py
    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py --no-call   # 只看配置，不发请求、不花钱

它会：
    1. 打印当前配置（**密钥只显示脱敏值，绝不打印明文**）；
    2. 检查 provider 与密钥是否匹配（provider=mock 时密钥不会被使用，会明确提示）；
    3. 发一次最小请求（max_tokens 很小），报告延迟、token 用量与模型回复；
    4. 针对 401 / 402 / 404 / 超时 / 网络不可达给出中文处理建议。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from verimind.core.config import Settings  # noqa: E402

_OK = "  [ OK ]"
_WARN = "  [WARN]"
_FAIL = "  [FAIL]"

#: 不需要密钥的 provider
_KEYLESS = {"mock", "ollama"}

#: 常见错误码 -> 中文处理建议
_ERROR_HINTS = {
    401: "密钥无效或已失效：请确认复制完整（前后不要有空格、引号），或到供应商控制台重新生成。",
    402: "账户余额不足：请到供应商控制台充值后重试。",
    403: "没有权限访问该模型：请确认账号已开通该模型，或换一个模型名。",
    404: "模型名或接口地址不对：请核对 VERIMIND_LLM_MODEL 与 VERIMIND_LLM_BASE_URL。",
    429: "请求过于频繁（限流）：稍等片刻再试，或降低调用频率。",
    500: "供应商服务端错误：稍后重试。",
    503: "供应商服务暂时不可用：稍后重试。",
}


def describe_config(settings: Settings) -> None:
    print("【当前配置】")
    print(f"{_OK} provider   : {settings.llm_provider}")
    print(f"{_OK} model      : {settings.llm_model}")
    print(f"{_OK} base_url   : {settings.llm_base_url}")
    print(f"{_OK} API Key    : {settings.masked_api_key}（仅显示脱敏值）")
    print(f"{_OK} 运行模式    : {settings.run_mode.value}")
    print(f"{_OK} 超时/重试   : {settings.llm_timeout_seconds}s / {settings.llm_max_retries} 次")

    if (settings.llm_provider or "").strip().lower() in _KEYLESS:
        print()
        print(f"{_WARN} provider 是「{settings.llm_provider}」，密钥**不会**被使用。")
        print("        要用真实模型，请把 .env 里的 VERIMIND_LLM_PROVIDER 改成供应商名")
        print("        （deepseek / zhipu / moonshot / dashscope / openai），然后重启服务。")
    for message in settings.runtime_warnings:
        print(f"{_WARN} {message}")


def call_model(settings: Settings) -> bool:
    """发一次最小请求，验证凭据与连通性。返回是否成功。"""
    try:
        from openai import OpenAI
    except ImportError:
        print(f"{_FAIL} 没有安装 openai 库：python -m pip install -r requirements.txt")
        return False

    print()
    print("【连通性测试】发送一次最小请求（max_tokens=32，成本可忽略）…")

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=float(settings.llm_timeout_seconds),
        max_retries=0,  # 这里手动控制重试，方便看到真实错误
    )
    messages = [
        {"role": "system", "content": "你是一个测试用的助手，只输出 JSON。"},
        {"role": "user", "content": '请只输出这个 JSON，不要任何多余文字：{"ok": true, "note": "中文测试"}'},
    ]

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=32,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - 要把任何异常翻译成人话
        latency_ms = int((time.perf_counter() - started) * 1000)
        status = getattr(exc, "status_code", None)
        print(f"{_FAIL} 调用失败（{type(exc).__name__}，{latency_ms} ms）")
        if status is not None:
            print(f"        HTTP {status}：{_ERROR_HINTS.get(status, '未知错误码')}")
            if status == 404 and "response_format" in str(exc):
                print("        提示：该供应商可能不支持 response_format=json_object。")
        else:
            text = str(exc)
            if "timeout" in type(exc).__name__.lower() or "timed out" in text.lower():
                print("        请求超时：检查网络，或把 VERIMIND_LLM_TIMEOUT_SECONDS 调大。")
            elif "connection" in type(exc).__name__.lower() or "connect" in text.lower():
                print("        无法连接：检查网络、代理设置，或确认 VERIMIND_LLM_BASE_URL 是否正确。")
            else:
                print(f"        {text[:300]}")
        return False

    latency_ms = int((time.perf_counter() - started) * 1000)
    content = (response.choices[0].message.content or "").strip()
    usage = response.usage
    print(f"{_OK} 调用成功")
    print(f"        延迟      : {latency_ms} ms")
    print(f"        模型返回  : {content[:200]}")
    if usage is not None:
        print(
            f"        token 用量: prompt {usage.prompt_tokens} / completion {usage.completion_tokens}"
            f" / 合计 {usage.total_tokens}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="验证大模型密钥与连通性")
    parser.add_argument("--no-call", action="store_true", help="只检查配置，不发请求（不花钱）")
    args = parser.parse_args()

    print("=" * 68)
    print("智鉴 AI · 大模型密钥与连通性检查")
    print("=" * 68)

    settings = Settings()
    describe_config(settings)

    if args.no_call:
        print()
        print("（--no-call：已跳过真实请求）")
        return 0

    if settings.missing_credentials:
        print()
        print(f"{_FAIL} 没有配置密钥，无法测试。请在 .env 中填写 VERIMIND_LLM_API_KEY。")
        return 1

    if (settings.llm_provider or "").strip().lower() in _KEYLESS:
        print()
        print(f"{_WARN} 当前 provider 不需要密钥，跳过连通性测试。")
        return 0

    ok = call_model(settings)
    print("=" * 68)
    if ok:
        print("结论：密钥可用，可以在阶段 3 接入真实模型调用。")
        return 0
    print("结论：调用失败，请按上面的建议处理后重试。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
