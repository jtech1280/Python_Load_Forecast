from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.local.yaml"

_ENV_TOKEN_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
    r"|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
    r"|%(?P<windows>[A-Za-z_][A-Za-z0-9_]*)%"
)

_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "FORECAST_OUTPUT_DIR": ("project", "output_dir"),
    "FORECAST_SQL_DSN": ("sql", "dsn_name"),
    "FORECAST_OUTPUT_SQL_DSN": ("output_sql", "dsn_name"),
    "FORECAST_FIVE_MIN_DSN": ("five_min_load", "dsn_name"),
    "FORECAST_LOCAL_WEATHER_DSN": ("local_weather", "dsn_name"),
    "FORECAST_WEATHER_CACHE_DIR": ("openmeteo", "cache_dir"),
    "FORECAST_SOLAR_PARQUET_ROOT": ("solar", "parquet_root"),
    "FORECAST_SOLAR_DEST_SERVER": ("solar", "dest_server"),
    "FORECAST_SOLAR_DEST_DB": ("solar", "dest_db"),
    "FORECAST_SOLAR_DRIVER": ("solar", "driver"),
    "FORECAST_SOLAR_PRODUCTION_SOURCE": ("solar", "production_source"),
}

_PATH_KEYS: set[tuple[str, ...]] = {
    ("project", "output_dir"),
    ("openmeteo", "cache_dir"),
    ("btm", "tsv_path"),
    ("solar", "parquet_root"),
    ("solar", "weather_cache_dir"),
}


def load_forecast_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml with optional machine-local overrides and env expansion."""
    _load_dotenv(PROJECT_ROOT / ".env")
    _load_dotenv(PROJECT_ROOT / ".env.local")

    base_path = Path(
        os.environ.get("FORECAST_CONFIG")
        or config_path
        or DEFAULT_CONFIG_PATH
    )
    config = _read_yaml(base_path)

    local_path = Path(os.environ.get("FORECAST_CONFIG_LOCAL") or DEFAULT_LOCAL_CONFIG_PATH)
    if local_path.exists():
        config = deep_merge(config, _read_yaml(local_path))

    _apply_env_overrides(config)
    config = expand_env_values(config)
    return normalize_config_paths(config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, str):
        return _expand_env_string(value)
    return value


def normalize_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    for key_path in _PATH_KEYS:
        parent = _get_parent(config, key_path, create=False)
        if not parent:
            continue
        key = key_path[-1]
        value = parent.get(key)
        if value in {None, ""}:
            continue
        parent[key] = str(_resolve_path(value))

    solar_cfg = config.setdefault("solar", {})
    if not solar_cfg.get("parquet_root"):
        selected = _first_existing_path(solar_cfg.get("parquet_root_candidates", []))
        if selected is not None:
            solar_cfg["parquet_root"] = str(selected)

    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, _expand_env_string(value))


def _expand_env_string(value: str) -> str:
    defaults = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "REPO_ROOT": str(PROJECT_ROOT),
        "FORECAST_PROJECT_ROOT": str(PROJECT_ROOT),
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain") or match.group("windows")
        default = match.group("default") if match.group("braced") else None
        if name in os.environ:
            return os.environ[name]
        if name in defaults:
            return defaults[name]
        return default if default is not None else ""

    return _ENV_TOKEN_RE.sub(replace, value)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    for env_name, key_path in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        parent = _get_parent(config, key_path, create=True)
        parent[key_path[-1]] = value


def _get_parent(config: dict[str, Any], key_path: tuple[str, ...], *, create: bool) -> dict[str, Any] | None:
    node: dict[str, Any] = config
    for key in key_path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            if not create:
                return None
            child = {}
            node[key] = child
        node = child
    return node


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _first_existing_path(candidates: object) -> Path | None:
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if candidate in {None, ""}:
            continue
        path = _resolve_path(candidate)
        if path.exists():
            return path
    return None
