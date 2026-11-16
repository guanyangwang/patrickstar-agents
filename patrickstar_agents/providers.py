import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Reply:
    name: str
    model: str
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A safe, human-readable failure with no request or credential contents."""

    def __init__(
        self, message, *, code="provider_error", outcome="unknown", job_id=None
    ):
        super().__init__(message)
        self.code = code
        self.outcome = outcome
        self.job_id = job_id


_MAX_INPUT_BYTES = 256_000
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "WINDIR",
        "PATHEXT",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "GNOME_KEYRING_CONTROL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)


def _cli_environment() -> dict[str, str]:
    """Retain login discovery and networking, excluding API/billing overrides."""
    return {
        key: value
        for key, value in os.environ.items()
        if key in _ENV_ALLOWLIST or key.startswith("LC_")
    }


def _input_bytes(prompt: str, system: str = "") -> bytes:
    if not isinstance(prompt, str) or not isinstance(system, str):
        raise ProviderError("The provider requires text input.")
    text = (
        f"Instructions:\n{system}\n\nDiscussion input:\n{prompt}" if system else prompt
    )
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_INPUT_BYTES:
        raise ProviderError("The discussion input exceeds the provider input limit.")
    return encoded


def _count(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _usage(raw: Any, fields: Mapping[str, str]) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {
        target: count
        for source, target in fields.items()
        if (count := _count(raw.get(source))) is not None
    }
