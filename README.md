<p align="center">
  <img src="shellm-logo.svg" alt="SheLLM" width="360">
</p>

A lightweight [OpenClaw](https://github.com/openclaw) alternative. Nine specialized agents, 28 built-in tools, MCP extensibility, config-driven.

**SheLLM** (pronounced *shell-el-em*) is a multi-agent CLI framework for tool-using LLMs. A single entry point (`shellm.py`) routes work across nine purpose-built agents -- covering conversation, reasoning, vision, web search, academic research, and MCP server integration -- out of the box, with zero config beyond API keys.

```bash
echo "Summarize today's news" | ./shellm.py --daemon stdin
```

## Why SheLLM?

| | OpenClaw | SheLLM |
|---|---------|--------|
| Setup | Config files, plugin system, dependencies | `pip install -r requirements.txt`, fill `.env` |
| Add an agent | Write adapter, register, configure | Add a stanza to `agent_config.yaml` |
| Tool system | Plugin architecture | Built-in: search, shell, cron, memory, chat logs, MCP |
| Modes | Interactive | Interactive, daemon (stdin/file/socket), Telegram, API server |
| Footprint | Heavy | Modular packages (`agents/`, `tools/`) |
| **API cost** | Depends on model choice | **~$3/month** typical usage (see below) |

## Cost-Effective by Design

SheLLM runs nine agents across two cost-effective providers:

| Agent(s) | Provider | Model | Input | Output | Typical monthly cost |
|----------|----------|-------|-------|--------|---------------------|
| chat, updater, mcp-alpha/beta/gamma, reasoner | DeepSeek | deepseek-chat / deepseek-reasoner | $0.27/M | $1.10/M | ~$1.00 |
| image, websearch, researcher | OpenAI | gpt-5-mini | $1.25/M | $5.00/M | ~$2.00 |

**Total: ~$3/month for typical personal use** (a few dozen queries per day).

For comparison, Claude Pro or ChatGPT Plus subscriptions cost $20/month. SheLLM gives you nine specialized agents with tool calling, web search, vision, persistent memory, file editing, RAG, and Telegram access -- for a fraction of the cost, with no subscription lock-in. You pay only for what you use. Each agent uses its own API key to distribute load.

## Nine Agents

SheLLM ships with nine purpose-built agents, each chosen for its strength:

| Agent | Provider | Model | Role |
|-------|----------|-------|------|
| `shellm-chat` | DeepSeek | `deepseek-chat` | Main conversational agent, tool router |
| `shellm-updater` | DeepSeek | `deepseek-chat` | Observes progress, replies while chat is busy |
| `shellm-mcp-alpha` | DeepSeek | `deepseek-chat` | MCP server group A |
| `shellm-mcp-beta` | DeepSeek | `deepseek-chat` | MCP server group B |
| `shellm-mcp-gamma` | DeepSeek | `deepseek-chat` | MCP server group C |
| `shellm-reasoner` | DeepSeek | `deepseek-reasoner` | Deep reasoning / plan mode |
| `shellm-image` | OpenAI | `gpt-5-mini` | Image recognition (vision) |
| `shellm-websearch` | OpenAI | `gpt-5-mini` | Live web search |
| `shellm-researcher` | OpenAI | `gpt-5-mini` | Academic research |

```bash
./shellm.py                             # Chat -- default agent (shellm-chat)
./shellm.py --agent shellm-reasoner     # Deep reasoning / plan mode
./shellm.py --agent shellm-researcher   # Academic research
./shellm.py --list-agents               # Show all configured agents
```

## Quick Start

```bash
git clone https://github.com/vikarag/SheLLM.git
cd SheLLM
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy .env.example and fill in your keys
cp .env.example .env
# Edit .env with your API keys (DeepSeek + OpenAI)

./shellm.py
```

## Add Your Own Agent

Agents are declared in `agent_config.yaml`. Add a stanza and restart -- no code changes needed:

```yaml
agents:
  - name: shellm-myagent
    provider: deepseek          # or: openai
    model: deepseek-chat
    api_key_env: MY_API_KEY
    base_url: https://api.example.com/v1
    tool_set: full              # full | minimal | mcp_only | none
    system_prompt: |
      You are a specialist in ...
```

For custom behavior beyond config (e.g., a new agent class), add a spec under `agents/specs/` and register it in `agents/registry.py`.

## Built-in Tools (28)

Every agent gets its assigned tool set automatically:

| Tool | What it does |
|------|-------------|
| `read_file` | Read files from the project directory with line numbers |
| `write_file` | Write or append to files in workspace/ |
| `list_directory` | List files and directories in workspace/ with sizes |
| `search_files` | Regex search across files in workspace/ |
| `rag_index` | Index a document for semantic search (chunking + embeddings) |
| `rag_search` | Hybrid search: semantic similarity + FTS5 keyword matching |
| `rag_list` | List all indexed documents |
| `rag_delete` | Delete a document from the RAG index |
| `run_command` | Execute shell commands (with confirmation + blocklist) |
| `cron_create` | Schedule cron jobs |
| `cron_list` | List current cron jobs |
| `cron_delete` | Remove a cron job |
| `schedule_task` | Schedule a delayed Telegram message or shell command |
| `list_scheduled_tasks` | List scheduled tasks by status |
| `cancel_scheduled_task` | Cancel a pending scheduled task |
| `memory_write` | Save to persistent shared memory (SQLite) |
| `memory_read` | Read stored memories |
| `memory_search` | Search memories by keyword (FTS5 full-text search) |
| `memory_delete` | Delete a memory entry |
| `chat_log_read` | Query past conversations across all agents |
| `send_file` | Send files (images, documents) to the user via Telegram |
| `report_progress` | Send real-time progress updates via the updater agent |
| `mcp_list_servers` | List configured MCP servers and connection status |
| `mcp_list_tools` | List tools available from MCP servers |
| `delegate_websearch` | Delegate live web search to `shellm-websearch` |
| `delegate_image` | Delegate image recognition to `shellm-image` |
| `delegate_research` | Delegate academic research to `shellm-researcher` |
| `delegate_reason` | Delegate deep reasoning / planning to `shellm-reasoner` |

The model decides when to use them. Up to 20 tool-call rounds per turn.

Named tool sets (`tool_sets.py`) control which tools each agent receives: `full` (all 28), `minimal` (core file/memory/shell), `mcp_only` (MCP tools only), `none`.

### Auto-Delegation

`shellm-chat` automatically delegates specialized work to the right agent via delegation tools:

| Trigger | Tool | Target agent |
|---------|------|--------------|
| Web lookup, current events | `delegate_websearch` | `shellm-websearch` |
| Image or screenshot | `delegate_image` | `shellm-image` |
| Academic paper, deep research | `delegate_research` | `shellm-researcher` |
| Multi-step plan, hard reasoning | `delegate_reason` | `shellm-reasoner` |

No manual switching needed. The chat agent recognizes the task type and routes it autonomously. When the chat agent is busy executing a multi-step plan, `shellm-updater` observes the progress queue and replies to incoming messages so the user is never left waiting.

## MCP Server Support

SheLLM supports [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) -- connect external tool servers without changing code. MCP servers are distributed across three dedicated agents (`shellm-mcp-alpha`, `shellm-mcp-beta`, `shellm-mcp-gamma`) to isolate server groups and avoid tool-count limits.

```bash
# Create mcp_servers.json (Claude Desktop format):
cat > mcp_servers.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"],
      "env": {}
    }
  }
}
EOF
```

MCP tools are namespaced as `{server}__{tool}` (e.g., `filesystem__read_file`) to avoid collisions with built-in tools. Servers connect lazily on first use. Missing config, failed servers, or missing SDK = no crash, just no MCP tools. Assign servers to alpha/beta/gamma agents in `agent_config.yaml` to spread load.

## Telegram Bot

SheLLM runs natively as a Telegram bot with real-time streaming, HTML-formatted responses, and persistent sessions (conversations survive bot restarts). Incoming images are automatically routed to `shellm-image`; `/plan` commands are routed to `shellm-reasoner`; messages arriving while the chat agent is busy are handled by `shellm-updater`.

```bash
# Set your bot token in .env, then:
./shellm.py --telegram
```

**Bot commands:** `/search`, `/plan`, `/memory`, `/remember`, `/recall`, `/usage`, `/files`, `/download`, `/logs`, `/model`, `/clear`, `/forget`, `/help`

### Plan Mode

The `/plan` command lets you preview an execution plan before running a task:

1. Send `/plan <task>` -- routed to `shellm-reasoner` for a step-by-step plan with tools and delegation strategy
2. Reply **yes** to execute, **no** to cancel, or send feedback to revise the plan
3. Feedback is iterative -- keep refining until the plan looks right, then approve

During execution, the chat agent calls `report_progress` after each major step. The progress queue delivers updates to `shellm-updater`, which summarizes and sends a Telegram notification -- real-time visibility into multi-step plans without blocking execution.

```
You:  /plan install htop and verify it works
Bot:  [Plan with steps, tools, complexity]
You:  yes
Bot:  [Plan Progress 1/3] Installed htop via apt-get...
Bot:  [Plan Progress 2/3] Verified htop runs and displays process list...
Bot:  [Final response with full results]
```

Progress updates are best-effort -- if delivery fails, execution continues unaffected. In CLI mode, `report_progress` is a silent no-op.

Responses are automatically converted from Markdown to Telegram HTML with proper code blocks, bold, italic, links, and blockquotes.

## Web UI

SheLLM exposes an OpenAI-compatible API server for use with [Open WebUI](https://github.com/open-webui/open-webui) or any OpenAI-compatible frontend.

```bash
# Start the API server
python api_server.py

# Or with uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The server (`api_server.py`) is a FastAPI/uvicorn application that implements the `/v1/chat/completions` endpoint. Each configured agent appears as a selectable model in the UI. The production instance is accessible at `shellm.gsl.ee`.

## Run Modes

```bash
# Interactive (default agent: shellm-chat)
./shellm.py

# Specific agent
./shellm.py --agent shellm-reasoner

# List all configured agents
./shellm.py --list-agents

# Pipe a prompt
echo "What is 2+2?" | ./shellm.py --daemon stdin

# JSON output
echo "What is 2+2?" | ./shellm.py --daemon stdin --json

# Batch from file
./shellm.py --daemon file --input prompts.txt --output responses.txt

# Unix socket server (concurrent access)
./shellm.py --daemon socket --socket-path /tmp/shellm.sock

# Telegram bot
./shellm.py --telegram

# API server (Open WebUI / any OpenAI-compatible frontend)
python api_server.py
```

## Interactive Commands

| Command | Action |
|---------|--------|
| `/search <query>` | Force a web search workflow |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |

## Chat Logging

Every conversation turn is automatically saved to `chat_logs.json` with:
- Timestamp (KST/UTC+9), agent name, run mode
- Full user input and assistant response
- All tool calls made during the turn
- Response duration in milliseconds

Any agent can read past logs via the `chat_log_read` tool.

## Architecture

```
shellm.py                  -- unified entry point (interactive, daemon, telegram, agent select)
agent_config.yaml          -- declarative agent definitions (provider, model, tool_set, prompt)
api_server.py              -- OpenAI-compatible API server (FastAPI/uvicorn) for Open WebUI

agents/
  config.py                -- AgentConfig dataclass + YAML loader
  registry.py              -- AgentRegistry singleton, lazy agent creation
  base_agent.py            -- BaseAgent (streaming, tool loop, chat logging)
  progress.py              -- Thread-safe ProgressQueue (chat -> updater)
  specs/
    chat.py                -- ChatAgent       (primary router)
    updater.py             -- UpdaterAgent    (progress-aware, handles busy state)
    mcp_agent.py           -- MCPAgent        (filtered MCP server group)
    reasoner.py            -- ReasonerAgent   (no tools, deep reasoning)
    image.py               -- ImageAgent      (vision API)
    websearch.py           -- WebSearchAgent  (web_search tool)
    researcher.py          -- ResearcherAgent (academic focus)

tools/
  definitions.py           -- 28 tool definitions (JSON schema)
  executor.py              -- dispatch table + execute_tool()
  tool_sets.py             -- named subsets: full | minimal | mcp_only | none

telegram_adapter.py        -- Telegram bot (AgentRegistry-based, image/plan/busy routing)
daemon_mode.py             -- daemon modes (stdin, file, socket)
telegram_format.py         -- Markdown -> Telegram HTML

Modules:
  db.py                    -- Shared SQLite database (WAL mode, FTS5, thread-safe)
  file_tools.py            -- File ops (read from project dir, write/list/search in workspace/)
  task_scheduler.py        -- Heartbeat scheduler for delayed tasks (60s tick, SQLite-backed)
  rag_engine.py            -- Document indexing + hybrid search (cosine + BM25)
  command_runner.py        -- Shell command execution
  cron_manager.py          -- Cron job management
  memory_manager.py        -- Persistent shared memory (SQLite + FTS5 + auto-archival)
  mcp_manager.py           -- MCP server connections + tool routing (async bridge)

SheLLM.db    -- single SQLite database for memory + RAG (WAL mode)
workspace/   -- SheLLM's project directory for file output
```

Agent routing overview:

```
User input
    |
shellm.py
    |
    +-- text query ---------> shellm-chat
    |                              |-- delegate_websearch --> shellm-websearch
    |                              |-- delegate_image     --> shellm-image
    |                              |-- delegate_research  --> shellm-researcher
    |                              |-- delegate_reason    --> shellm-reasoner
    |                              +-- report_progress   --> shellm-updater (via ProgressQueue)
    |
    +-- image attached -----> shellm-image
    +-- /plan command ------> shellm-reasoner
    +-- --agent <name> -----> any agent directly
    +-- api_server.py ------> any agent (model selector in web UI)

MCP servers --> shellm-mcp-alpha / shellm-mcp-beta / shellm-mcp-gamma
```

## License

MIT
