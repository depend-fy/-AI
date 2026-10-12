"""统一异常类型。

为什么要自定义异常？
    界面上要显示"能看懂的中文原因"，而不是 Python 的堆栈。
    同时业务层需要区分"输入问题"（用户改文本即可）和"外部服务问题"（重试或降级），
    用不同的异常类型区分，才能在界面上给出不同的处理建议。
"""

from __future__ import annotations


class VeriMindError(Exception):
    """本项目所有自定义异常的基类。"""


# ---------------- 输入类 ----------------
class InputError(VeriMindError):
    """输入不合法（空文本、格式问题）。"""


class InputTooLongError(InputError):
    """文本超过配置的长度上限。"""

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"文本长度 {length} 字，超过上限 {limit} 字。"
            f"请缩短文本后重试（上限可在 .env 中通过 VERIMIND_MAX_INPUT_CHARS 调整）。"
        )


# ---------------- 大模型类 ----------------
class LLMError(VeriMindError):
    """大模型调用相关的错误。"""


class LLMTimeoutError(LLMError):
    """调用超时。"""


class LLMConnectionError(LLMError):
    """网络或服务不可达。"""


class SchemaValidationError(LLMError):
    """模型返回的内容不符合约定的数据结构（修复重试后仍然失败）。"""


class LLMUnsupportedRoleError(LLMError):
    """当前客户端不支持该角色。

    典型场景：阶段 2 的离线规则引擎只能做"主张拆解"，
    其他角色要等阶段 3 接入真实模型——这时必须明确报错，而不是编一个假结果。
    """


class LLMBudgetExceeded(LLMError):
    """超出单次检测的模型调用预算。"""


# ---------------- 检索类 ----------------
class RetrievalError(VeriMindError):
    """外部文献检索出错。"""


# ---------------- 存储类 ----------------
class StorageError(VeriMindError):
    """本地数据读写出错。"""
