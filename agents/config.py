"""Agent configuration dataclass and YAML loader."""

import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for a single SheLLM agent profile."""
    name: str
    provider: str                          # label only; transport is OpenAI-compatible
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    stream: bool = True
    temperature: Optional[float] = None
    supports_tools: bool = True
    has_reasoning: bool = False
    vision: bool = False                   # set true when the model supports image inputs


def load_agent_configs(config_path: Optional[str] = None) -> tuple[dict, Optional[str]]:
    """Load agent configurations from YAML.

    Returns:
        (configs, default_name) where configs maps agent name -> AgentConfig and
        default_name is the profile to use when no --agent flag is passed
        (taken from the top-level `default:` field, or the first agent if unset).
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agent_config.yaml",
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    configs = {}
    for agent_data in raw.get("agents", []):
        name = agent_data["name"]
        configs[name] = AgentConfig(
            name=name,
            provider=agent_data.get("provider", ""),
            model=agent_data.get("model", ""),
            api_key_env=agent_data.get("api_key_env", ""),
            base_url=agent_data.get("base_url"),
            stream=agent_data.get("stream", True),
            temperature=agent_data.get("temperature"),
            supports_tools=agent_data.get("supports_tools", True),
            has_reasoning=agent_data.get("has_reasoning", False),
            vision=agent_data.get("vision", False),
        )

    default_name = raw.get("default")
    if default_name and default_name not in configs:
        raise ValueError(
            f"agent_config.yaml: default '{default_name}' is not one of the configured agents: "
            f"{list(configs)}"
        )
    if not default_name and configs:
        default_name = next(iter(configs))

    return configs, default_name
