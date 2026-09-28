"""LLM 客户端：可选增强，不引入新依赖。

## 为什么用标准库 urllib 而不是 openai SDK

这个仓库的 `requirements.txt` 只有 7 个包，全部围绕量化计算
（akshare / pandas / numpy / streamlit / plotly / matplotlib / pyarrow）。
为了一个**可选**的规划器增强再引入一个 HTTP SDK，会让「clone 下来就能跑」
多一个失败点（版本冲突、代理配置、SDK 变更）。OpenAI 兼容的
`/chat/completions` 接口是一个稳定的 JSON HTTP 端点，用 `urllib` 四十行就够。

## 三种客户端

- `NullClient` —— 未配置时使用。`available` 为假，规划器据此直接用规则规划，
  不会浪费时间尝试请求。
- `OpenAICompatibleClient` —— 任何兼容 OpenAI 协议的端点（OpenAI、DeepSeek、
  Moonshot、本地 vLLM / Ollama …），靠环境变量配置。
- `build_llm_client()` —— 工厂。读 `QFM_AGENT_LLM_BASE_URL` /
  `QFM_AGENT_LLM_API_KEY` / `QFM_AGENT_LLM_MODEL`，缺任一项返回 `NullClient`。

## 超时与失败语义

请求超时默认 30 秒，且**任何失败都不抛给上层**——由 `LLMPlanner` 捕获后回落到
规则规划。理由是：规划器不可用不应该让整个研究请求失败，用户要的是研究结果，
不是「模型没连上」的报错。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "LLMClient",
    "NullClient",
    "OpenAICompatibleClient",
    "build_llm_client",
    "LLMConfig",
]

DEFAULT_TIMEOUT = 30.0


class LLMClient(Protocol):
    """最小 LLM 接口：一次补全。"""

    name: str

    @property
    def available(self) -> bool: ...

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class NullClient:
    """未配置 LLM 时的空实现。`available` 为假，调用即抛错。"""

    name: str = "null"
    reason: str = "未配置 QFM_AGENT_LLM_* 环境变量"

    @property
    def available(self) -> bool:
        return False

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError(f"LLM 不可用：{self.reason}")


@dataclass
class LLMConfig:
    """OpenAI 兼容端点的连接配置。"""

    base_url: str
    api_key: str
    model: str
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = 0.0
    max_tokens: int = 1200

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url 不能为空")
        if not self.model:
            raise ValueError("model 不能为空")
        if self.timeout <= 0:
            raise ValueError("timeout 必须为正")

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "LLMConfig | None":
        env = environ if environ is not None else os.environ
        base_url = (env.get("QFM_AGENT_LLM_BASE_URL") or "").strip()
        api_key = (env.get("QFM_AGENT_LLM_API_KEY") or "").strip()
        model = (env.get("QFM_AGENT_LLM_MODEL") or "").strip()
        if not (base_url and api_key and model):
            return None
        timeout = _float(env.get("QFM_AGENT_LLM_TIMEOUT"), DEFAULT_TIMEOUT)
        return cls(base_url=base_url, api_key=api_key, model=model, timeout=timeout)

    def to_dict(self) -> dict[str, Any]:
        # 刻意不回传 api_key：运行记录会落盘，密钥不能进报告。
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": self.temperature,
        }


class OpenAICompatibleClient:
    """调用 OpenAI 兼容的 `/chat/completions`。只依赖标准库。"""

    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        timeout = float(kwargs.get("timeout", self.config.timeout))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001 - 读错误体失败不影响抛出主错误
                detail = ""
            raise RuntimeError(f"LLM 返回 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 连接失败：{exc.reason}") from exc

        return _extract_content(body)


def _extract_content(body: str) -> str:
    """从响应体里取出文本内容，兼容常见字段差异。"""
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM 响应不是合法 JSON：{body[:200]}") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        error = data.get("error")
        if error:
            raise RuntimeError(f"LLM 返回错误：{error}")
        raise RuntimeError(f"LLM 响应缺少 choices：{body[:200]}")

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # 部分端点返回分段内容
            return "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict)
            )
    text = first.get("text") if isinstance(first, dict) else None
    if isinstance(text, str):
        return text
    raise RuntimeError(f"无法从 LLM 响应中提取文本：{body[:200]}")


def build_llm_client(environ: dict[str, str] | None = None) -> LLMClient:
    """按环境变量构建客户端；未配置时返回 `NullClient`。"""
    config = LLMConfig.from_env(environ)
    if config is None:
        return NullClient()
    return OpenAICompatibleClient(config)


def _float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default
