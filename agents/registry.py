"""Agent registry — loads model profiles and lazily creates the active agent."""

import os
import threading

from openai import OpenAI

from agents.config import AgentConfig, load_agent_configs


class AgentRegistry:
    """Singleton registry for SheLLM model profiles."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config_path=None):
        self._configs, self._default = load_agent_configs(config_path)
        self._agents = {}
        self._clients = {}

    @classmethod
    def get_instance(cls, config_path=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def _create_client(self, config: AgentConfig) -> OpenAI:
        """Create an OpenAI-compatible client. Local providers like Ollama
        don't require a real key, so a placeholder is substituted."""
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key and config.provider in ("ollama", "local"):
            api_key = "ollama"
        kwargs = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return OpenAI(**kwargs)

    def _create_agent(self, name: str):
        config = self._configs[name]
        if name not in self._clients:
            self._clients[name] = self._create_client(config)
        from agents.base_agent import BaseAgent
        agent = BaseAgent(config=config, client=self._clients[name])
        self._agents[name] = agent
        return agent

    def get_agent(self, name: str):
        if name not in self._configs:
            raise KeyError(f"Unknown agent: {name}")
        if name not in self._agents:
            self._agents[name] = self._create_agent(name)
        return self._agents[name]

    def get_all_configs(self):
        return self._configs

    def get_default_name(self):
        """Name of the profile to use when --agent is not passed."""
        return self._default
