"""Agent configuration dataclass and YAML loader."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AgentConfig:
    """Configuration for a single SheLLM agent."""
    name: str
    provider: str                          # "deepseek" | "openai"
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    stream: bool = True
    temperature: Optional[float] = None
    supports_tools: bool = True
    has_reasoning: bool = False
    tool_set: str = "full"                 # "full", "minimal", "mcp_only", "none"
    mcp_servers: List[str] = field(default_factory=list)
    delegations: List[str] = field(default_factory=list)
    system_role: str = "primary"           # "primary", "updater", "mcp_worker", "reasoner", "vision", "websearch", "researcher"


def load_agent_configs(config_path: Optional[str] = None) -> dict:
    """Load agent configurations from YAML file.

    Returns:
        Dict mapping agent name -> AgentConfig
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
            provider=agent_data.get("provider", "deepseek"),
            model=agent_data.get("model", "deepseek-chat"),
            api_key_env=agent_data.get("api_key_env", ""),
            base_url=agent_data.get("base_url"),
            stream=agent_data.get("stream", True),
            temperature=agent_data.get("temperature"),
            supports_tools=agent_data.get("supports_tools", True),
            has_reasoning=agent_data.get("has_reasoning", False),
            tool_set=agent_data.get("tool_set", "full"),
            mcp_servers=agent_data.get("mcp_servers", []),
            delegations=agent_data.get("delegations", []),
            system_role=agent_data.get("system_role", "primary"),
        )

    return configs
