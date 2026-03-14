"""Reasoner agent — deep reasoning with DeepSeek Reasoner model."""

from agents.base_agent import BaseAgent


class ReasonerAgent(BaseAgent):
    """Deep reasoning agent using deepseek-reasoner model.

    No tool access — returns structured plans with reasoning traces.
    """

    def build_params(self, messages):
        """Reasoner has no tools and no temperature."""
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": self.config.stream,
        }

    def build_no_tool_params(self, messages):
        return self.build_params(messages)
