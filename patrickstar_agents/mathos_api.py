"""Project-scoped Open Math OS Model API; no browser worker or tunnel."""

import asyncio
import re
from pathlib import Path

import httpx

from .config import ConfigError, _read_private_json
from .providers import ProviderError, Reply, _input_bytes, _usage

BASE_URL = "https://os.openmathmodel.org/v1"


class MathOSAPIProvider:
    name = "MathOS"
    route_note = "MathOS Model API · 使用团队额度；型号来自账号模型列表"

    def __init__(
        self,
        api_key,
        models,
        model,
        *,
        timeout=900,
        max_output_tokens=None,
        client=None,
    ):
        if not isinstance(api_key, str) or not api_key.startswith("nuc_ak_"):
            raise ConfigError("MathOS Model API key is missing or invalid.")
        self._api_key = api_key
        self._models = {
            m["id"]: m
            for m in models
            if isinstance(m, dict)
            and isinstance(m.get("id"), str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/\-]{0,150}", m["id"])
        }
        if model not in self._models:
            raise ConfigError(
                "MathOS default model is not in the verified model catalog."
            )
        self.model = model
        self.model_labels = {
            k: v.get("display_name") or v.get("name") or k
            for k, v in self._models.items()
        }
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._client = client

    @classmethod
    def from_file(cls, path):
        data = _read_private_json(Path(path))
        if not isinstance(data, dict) or data.get("enabled") is not True:
            return None
        return cls(data.get("api_key"), data.get("models", []), data.get("model"))

    @property
    def request_timeout(self):
        return self.timeout + 15

    def available(self):
        return bool(self._api_key and self.model in self._models)

    def selectable_models(self):
        return list(self._models)

    async def start(self):
        pass

    async def close(self):
        pass

    async def complete(self, prompt, *, system):
        _input_bytes(prompt, system)
        if self.model not in self._models:
            raise ProviderError(
                "MathOS 型号不在已验证的模型列表中。", outcome="not_started"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens
        client = self._client or httpx.AsyncClient()
        try:
            async with asyncio.timeout(self.timeout):
                async with client.stream(
                    "POST",
                    BASE_URL + "/chat/completions",
                    headers={"Authorization": "Bearer " + self._api_key},
                    json=payload,
                    timeout=self.timeout,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in (401, 403):
                        raise ProviderError(
                            "MathOS API 密钥或模型权限校验失败；未切换其他渠道。",
                            code="auth_failed",
                        )
                    if response.status_code == 429:
                        raise ProviderError(
                            "MathOS 团队额度或速率限制；未自动重试。",
                            code="rate_limited",
                        )
                    if response.status_code != 200:
                        raise ProviderError(
                            "MathOS API 请求失败（HTTP "
                            + str(response.status_code)
                            + "）；未自动重试。"
                        )
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 2_000_000:
                            raise ProviderError("MathOS API 回答超出接收大小限制。")
        except (TimeoutError, httpx.TimeoutException):
            raise ProviderError(
                "MathOS API 等待超时，结果未知；未自动重放。", code="response_timeout"
            ) from None
        except httpx.HTTPError:
            raise ProviderError("MathOS API 连接中断，结果未知；未自动重放。") from None
        finally:
            if self._client is None:
                await client.aclose()
        try:
            import json

            result = json.loads(raw)
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError()
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError()
        except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
            raise ProviderError(
                "MathOS API 未返回完整回答；未将截断内容当作完成。"
            ) from None
        usage = _usage(
            result.get("usage"),
            {
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
                "total_tokens": "total_tokens",
            },
        )
        usage.update(route="mathos-model-api", requested_model=self.model)
        actual_model = result.get("model")
        return Reply(
            self.name,
            actual_model
            if isinstance(actual_model, str) and actual_model
            else self.model,
            answer.strip(),
            usage,
        )
