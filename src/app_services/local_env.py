from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def load_env_file(path: str | Path, env: MutableMapping[str, str] | None = None) -> bool:
    target_env = env if env is not None else os.environ
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        if not key or key in target_env:
            continue

        target_env[key] = _strip_optional_quotes(value.strip())

    return True


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value