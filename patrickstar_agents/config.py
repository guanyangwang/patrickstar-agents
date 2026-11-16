"""Find local agent CLIs and load optional per-user settings."""

import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path

GEMINI_MODELS = (
    "gemini-4-argon-high",
    "gemini-4-argon-medium",
    "gemini-4-argon-low",
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CliConfig:
    executable: str
    expires: date | None = None

    def available(self):
        return (
            Path(self.executable).is_file()
            and os.access(self.executable, os.X_OK)
            and (self.expires is None or date.today() < self.expires)
        )


@dataclass(frozen=True)
class Config:
    claude: CliConfig
    antigravity: CliConfig | None


def default_config_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "patrickstar-agents" / "config.json"


def _read_private_json(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ConfigError(
                "Config must be a regular file owned by the current user."
            )
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ConfigError(f"Set permissions with chmod 600 {path}")
        data = stream.read(65537)
        if len(data) > 65536:
            raise ConfigError("Config exceeds 64 KiB.")
        return json.loads(data)


def _cli(settings, name):
    executable = settings.get("executable")
    if executable:
        path = Path(executable).expanduser()
        if not path.is_absolute():
            raise ConfigError("Use an absolute CLI executable path.")
    else:
        path = Path(shutil.which(name) or Path.home() / ".local/bin" / name)
    expires = settings.get("expires")
    return CliConfig(str(path), date.fromisoformat(expires) if expires else None)


def load_config(path):
    try:
        data = _read_private_json(path)
    except FileNotFoundError:
        data = {}
    native = data.get("antigravity", {})
    return Config(
        claude=_cli(data.get("claude", {}), "claude"),
        antigravity=_cli(native, "agy") if native.get("enabled", True) else None,
    )
