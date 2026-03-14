"""MCP worker agent — handles MCP tool calls for assigned servers."""

from agents.base_agent import BaseAgent
from mcp_manager import MCPManager


class MCPAgent(BaseAgent):
    """Handles MCP tool calls for a specific set of assigned servers."""

    def __init__(self, config, client, registry=None):
        super().__init__(config, client, registry)
        # Create a filtered MCPManager for this agent's assigned servers
        server_filter = config.mcp_servers if config.mcp_servers else None
        self._mcp = MCPManager(server_filter=server_filter)

    def build_params(self, messages):
        """Include only MCP tools from assigned servers."""
        params = {"model": self.config.model, "messages": messages, "stream": self.config.stream}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        mcp_tools = self._mcp.get_tools()
        if mcp_tools:
            params["tools"] = mcp_tools
        return params

    def call_mcp_tool(self, namespaced_name, args):
        """Call an MCP tool through this agent's MCPManager."""
        return self._mcp.call_tool(namespaced_name, args or {})

    def get_mcp_tools(self):
        """Return MCP tools available to this agent."""
        return self._mcp.get_tools()
