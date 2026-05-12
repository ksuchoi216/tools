from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class OpenAIReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: ReasoningEffort


class OpenAINodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    use_responses_api: bool = True
    prompt_cache_key: str | None = None
    temperature: float | None = None
    reasoning: OpenAIReasoningConfig | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


class OpenAIConfigCollection(dict[str, OpenAINodeConfig]):
    def __getattr__(self, name: str) -> OpenAINodeConfig:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config item not found: {name}") from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.keys()))


def _resolve_config_path(config_path: str | Path, *, is_test: bool = False) -> Path:
    path = Path(config_path)
    if is_test:
        return path.with_name(f"{path.stem}_test{path.suffix}")
    return path


def _load_models_config(
    config_path: str | Path,
    *,
    is_test: bool = False,
) -> dict[str, dict[str, Any]]:
    path = _resolve_config_path(config_path, is_test=is_test)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config root in {path}: expected dict")
    return data


@lru_cache(maxsize=None)
def load_config(
    session: str,
    config_path: str | Path,
    *,
    is_test: bool = False,
) -> OpenAIConfigCollection:
    session_config = _load_models_config(config_path, is_test=is_test).get(session)
    if not isinstance(session_config, dict):
        raise KeyError(f"Session config not found: {session}")

    configs = OpenAIConfigCollection()
    for config_name, node_data in session_config.items():
        if not isinstance(node_data, dict):
            raise ValueError(f"Invalid node config: {session}.{config_name}")
        configs[config_name] = OpenAINodeConfig.model_validate(node_data)

    return configs
