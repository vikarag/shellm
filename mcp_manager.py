"""MCP (Model Context Protocol) manager for shellm.

Manages connections to external MCP servers, discovers their tools,
and routes tool calls. Uses a dedicated daemon thread with its own
asyncio event loop to bridge shellm's sync code with the async MCP SDK.
"""

import asyncio
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

# MCP SDK — graceful degradation if not installed
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _mcp_tool_to_openai(server_name, mcp_tool):
    """Convert an MCP tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{mcp_tool.name}",
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MCPManager:
    """Singleton manager for MCP server connections."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._servers = {}       # name -> config dict
        self._sessions = {}      # name -> ClientSession
        self._tools = {}         # name -> list of OpenAI-format tool dicts
        self._shutdown_events = {}  # name -> asyncio.Event
        self._tasks = {}         # name -> asyncio.Task
        self._connected = False
        self._config_loaded = False

        # Dedicated asyncio event loop in daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Config loading ───────────────────────────────────────────────

    def _load_config(self):
        """Load MCP server configuration from mcp_servers.json."""
        if self._config_loaded:
            return
        self._config_loaded = True

        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mcp_servers.json"
        )
        if not os.path.exists(config_path):
            logger.debug("No mcp_servers.json found — MCP disabled")
            return

        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read mcp_servers.json: {e}")
            return

        servers = config.get("mcpServers", {})
        if not isinstance(servers, dict):
            logger.warning("mcp_servers.json: 'mcpServers' must be an object")
            return

        self._servers = servers

    # ── Server lifecycle ─────────────────────────────────────────────

    async def _server_lifecycle(self, name, config):
        """Long-lived task that keeps a server connection open."""
        command = config.get("command", "")
        args = config.get("args", [])
        env = config.get("env")

        # Build environment: inherit current env, overlay server-specific env
        server_env = dict(os.environ)
        if env and isinstance(env, dict):
            server_env.update(env)

        params = StdioServerParameters(
            command=command, args=args, env=server_env
        )
        shutdown_event = asyncio.Event()
        self._shutdown_events[name] = shutdown_event

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Discover tools
                    result = await session.list_tools()
                    tools = [_mcp_tool_to_openai(name, t) for t in result.tools]

                    self._sessions[name] = session
                    self._tools[name] = tools
                    logger.info(
                        f"MCP server '{name}' connected — {len(tools)} tools"
                    )

                    # Block until shutdown requested
                    await shutdown_event.wait()
        except Exception as e:
            logger.warning(f"MCP server '{name}' failed: {e}")
            self._tools[name] = []
        finally:
            self._sessions.pop(name, None)

    def _connect_all(self):
        """Connect to all configured servers (lazy, called once)."""
        if self._connected:
            return
        self._connected = True

        if not MCP_AVAILABLE:
            logger.debug("MCP SDK not installed — MCP disabled")
            return

        self._load_config()

        if not self._servers:
            return

        for name, config in self._servers.items():
            future = asyncio.run_coroutine_threadsafe(
                self._start_server(name, config), self._loop
            )
            try:
                future.result(timeout=15)
            except Exception as e:
                logger.warning(f"MCP server '{name}' startup timed out or failed: {e}")

    async def _start_server(self, name, config):
        """Schedule a server lifecycle task and wait for it to be ready."""
        task = self._loop.create_task(self._server_lifecycle(name, config))
        self._tasks[name] = task

        # Wait for the session to appear (or the task to fail)
        for _ in range(100):  # up to 10 seconds
            if name in self._sessions or task.done():
                break
            await asyncio.sleep(0.1)

    # ── Public API ───────────────────────────────────────────────────

    def get_tools(self):
        """Return all MCP tools in OpenAI function-calling format.

        Lazy-connects to servers on first call. Returns [] on any failure.
        """
        if not MCP_AVAILABLE:
            return []

        try:
            self._connect_all()
        except Exception as e:
            logger.warning(f"MCP connect error: {e}")
            return []

        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools)
        return tools

    def call_tool(self, namespaced_name, arguments):
        """Call an MCP tool by its namespaced name (server__tool).

        Returns the result as a plain text string.
        """
        parts = namespaced_name.split("__", 1)
        if len(parts) != 2:
            return f"Invalid MCP tool name: {namespaced_name}"

        server_name, tool_name = parts
        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server '{server_name}' is not connected"

        try:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(tool_name, arguments or {}), self._loop
            )
            result = future.result(timeout=30)

            # Extract text from result content
            texts = []
            for item in result.content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                else:
                    texts.append(str(item))
            return "\n".join(texts) if texts else "(empty result)"
        except Exception as e:
            return f"MCP tool error ({namespaced_name}): {e}"

    def list_servers(self):
        """Return a formatted string listing all configured servers and status."""
        if not MCP_AVAILABLE:
            return "MCP SDK not installed. Install with: pip install mcp"

        self._load_config()

        if not self._servers:
            return "No MCP servers configured. Add servers to mcp_servers.json."

        lines = ["MCP Servers:\n"]
        for name, config in self._servers.items():
            connected = name in self._sessions
            status = "connected" if connected else "disconnected"
            tool_count = len(self._tools.get(name, []))
            cmd = config.get("command", "?")
            lines.append(
                f"  {name}: {status} ({tool_count} tools) — command: {cmd}"
            )
        return "\n".join(lines)

    def list_server_tools(self, server=None):
        """Return a formatted string listing tools, optionally filtered by server."""
        if not MCP_AVAILABLE:
            return "MCP SDK not installed."

        self._connect_all()

        if server:
            tools = self._tools.get(server, [])
            if not tools:
                return f"No tools found for server '{server}' (not connected or no tools)."
            lines = [f"Tools from '{server}':\n"]
            for t in tools:
                fn = t["function"]
                lines.append(f"  {fn['name']}: {fn.get('description', '(no description)')}")
            return "\n".join(lines)

        # All servers
        all_tools = self.get_tools()
        if not all_tools:
            return "No MCP tools available."

        lines = ["All MCP tools:\n"]
        for t in all_tools:
            fn = t["function"]
            lines.append(f"  {fn['name']}: {fn.get('description', '(no description)')}")
        return "\n".join(lines)

    def shutdown(self):
        """Cleanly disconnect all servers."""
        for name, event in self._shutdown_events.items():
            self._loop.call_soon_threadsafe(event.set)

        # Wait briefly for tasks to finish
        for name, task in self._tasks.items():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(asyncio.shield(task), timeout=3), self._loop
                )
                future.result(timeout=5)
            except Exception:
                pass

        self._sessions.clear()
        self._tools.clear()
        self._shutdown_events.clear()
        self._tasks.clear()
        self._connected = False
        logger.info("MCP manager shut down")
