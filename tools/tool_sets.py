"""Named tool subsets for different agent roles."""

from tools.definitions import TOOLS


def _filter_tools(names):
    """Return tools whose function name is in the given set."""
    return [t for t in TOOLS if t["function"]["name"] in names]


# Full tool set for shellm-chat (primary agent)
FULL_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

# Minimal tool set for shellm-updater
MINIMAL_TOOL_NAMES = {"memory_read", "memory_search", "chat_log_read"}

# No tools for reasoner, image, websearch, researcher
NONE_TOOL_NAMES = set()


def get_tool_set(name):
    """Get a tool set by name.

    Args:
        name: "full", "minimal", "mcp_only", "none"

    Returns:
        List of tool definitions (for "mcp_only", returns empty — MCP tools added separately)
    """
    if name == "full":
        return list(TOOLS)
    elif name == "minimal":
        return _filter_tools(MINIMAL_TOOL_NAMES)
    elif name == "mcp_only":
        return []  # MCP tools are added by the MCPAgent from its MCPManager
    elif name == "none":
        return []
    else:
        return list(TOOLS)
