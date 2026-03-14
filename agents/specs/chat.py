"""Primary chat agent — main conversational agent with full tool access."""

from agents.base_agent import BaseAgent


class ChatAgent(BaseAgent):
    """The primary SheLLM agent — handles user conversation and tool routing."""
    pass  # All logic is in BaseAgent; system_role="primary" drives the system prompt
