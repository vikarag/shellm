"""Agent registry — singleton that manages all agent instances."""

import os
import threading
from typing import Optional

from openai import OpenAI

from agents.config import AgentConfig, load_agent_configs


# Map system_role -> specialization class import path
_AGENT_CLASSES = {
    "primary": ("agents.specs.chat", "ChatAgent"),
    "updater": ("agents.specs.updater", "UpdaterAgent"),
    "mcp_worker": ("agents.specs.mcp_agent", "MCPAgent"),
    "reasoner": ("agents.specs.reasoner", "ReasonerAgent"),
    "vision": ("agents.specs.image", "ImageAgent"),
    "websearch": ("agents.specs.websearch", "WebSearchAgent"),
    "researcher": ("agents.specs.researcher", "ResearcherAgent"),
}


class AgentRegistry:
    """Singleton registry for all SheLLM agents."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config_path=None):
        self._configs = load_agent_configs(config_path)
        self._agents = {}  # name -> BaseAgent instance
        self._clients = {}  # name -> OpenAI client
        self._server_to_agent = {}  # MCP server_name -> agent_name
        self._build_mcp_routing()

    @classmethod
    def get_instance(cls, config_path=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def _build_mcp_routing(self):
        """Build server_name -> agent_name lookup from config."""
        for name, config in self._configs.items():
            for server in config.mcp_servers:
                self._server_to_agent[server] = name

    def _create_client(self, config: AgentConfig) -> OpenAI:
        """Create an OpenAI client for an agent config."""
        api_key = os.environ.get(config.api_key_env, "")
        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAI(**kwargs)

    def _create_agent(self, name: str):
        """Lazily create an agent instance."""
        config = self._configs[name]

        # Get or create the OpenAI client
        if name not in self._clients:
            self._clients[name] = self._create_client(config)
        client = self._clients[name]

        # Import the specialization class
        role = config.system_role
        if role in _AGENT_CLASSES:
            module_path, class_name = _AGENT_CLASSES[role]
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        else:
            from agents.base_agent import BaseAgent
            cls = BaseAgent

        agent = cls(config=config, client=client, registry=self)
        self._agents[name] = agent
        return agent

    def get_agent(self, name: str):
        """Get an agent by name, creating it lazily."""
        if name not in self._configs:
            raise KeyError(f"Unknown agent: {name}")
        if name not in self._agents:
            self._agents[name] = self._create_agent(name)
        return self._agents[name]

    def get_primary(self):
        return self.get_agent("shellm-chat")

    def get_updater(self):
        return self.get_agent("shellm-updater")

    def get_mcp_agent_for_server(self, server_name: str) -> Optional[str]:
        """Return the agent name that owns a given MCP server."""
        return self._server_to_agent.get(server_name)

    def get_all_configs(self):
        """Return all agent configurations."""
        return self._configs

    def get_client(self, name: str) -> OpenAI:
        """Get the OpenAI client for an agent (creating if needed)."""
        if name not in self._clients:
            config = self._configs[name]
            self._clients[name] = self._create_client(config)
        return self._clients[name]
