# shellm

A lightweight [OpenClaw](https://github.com/openclaw) alternative. One base class, 18 built-in tools, extend in 15 lines.

**shellm** is a minimal CLI chat framework for tool-using LLMs. It gives any OpenAI-compatible model web search, shell access, cron scheduling, persistent memory, file editing, RAG document search, and chat logging -- out of the box, with zero config.

```bash
echo "Summarize today's news" | ./gpt5mini_chat.py --daemon stdin
```

## Why shellm?

| | OpenClaw | shellm |
|---|---------|--------|
| Setup | Config files, plugin system, dependencies | One Python class. `pip install openai numpy` |
| Add a model | Write adapter, register, configure | 15 lines: subclass, set 3 attributes, done |
| Tool system | Plugin architecture | Built-in: search, shell, cron, memory, chat logs |
| Modes | Interactive | Interactive, daemon (stdin/file/socket), Telegram |
| Footprint | Heavy | ~500 lines of core code |
| **API cost** | Depends on model choice | **< $1/month** typical usage (see below) |

## Cost-Effective by Design

shellm is built around the most cost-effective LLM APIs available today:

| Engine | Model | Input | Output | Typical monthly cost |
|--------|-------|-------|--------|---------------------|
| **Chat** | DeepSeek V3 | $0.27/M tokens | $1.10/M tokens | ~$0.30 |
| **Code** | Kimi K2.5 | $0.35/M tokens | $1.40/M tokens | ~$0.45 |
| **Research** | GPT-5 Mini | $1.25/M tokens | $5.00/M tokens | ~$1.50 |

**Total: ~$2-3/month for typical personal use** (a few dozen queries per day).

For comparison, Claude Pro or ChatGPT Plus subscriptions cost $20/month. shellm gives you three specialized engines with tool calling, web search, persistent memory, file editing, RAG, and Telegram access -- for a fraction of the cost, with no subscription lock-in. You pay only for what you use.

## Three Engines

shellm ships with three purpose-built engines, each chosen for its strength:

| Engine | Script | Model | Role |
|--------|--------|-------|------|
| **Chat** | `deepseek_chat.py` | DeepSeek | General conversation, reasoning, thinking mode |
| **Code** | `kimi_chat.py` | Kimi K2.5 | Code generation, analysis, refactoring (CoT) |
| **Research** | `gpt5mini_chat.py` | GPT-5 Mini | Web research, summarization, fact-finding |

```bash
./deepseek_chat.py    # Chat -- ask anything
./kimi_chat.py        # Code -- write and review code
./gpt5mini_chat.py    # Research -- search the web and synthesize answers
```

## Quick Start

```bash
git clone https://github.com/vikarag/shellm.git
cd shellm
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy .env.example and fill in your keys
cp .env.example .env
# Edit .env with your API keys (at least one)

./deepseek_chat.py
```

## Add Your Own Engine (15 lines)

```python
#!/usr/bin/env python3
from base_chat import BaseChatClient

class MyChat(BaseChatClient):
    MODEL = "my-model-id"
    BANNER_NAME = "My Engine"
    ENV_VAR = "MY_API_KEY"
    BASE_URL = "https://api.example.com/v1"  # omit for OpenAI

if __name__ == "__main__":
    MyChat().run()
```

Override `build_params()` for custom behavior (see `deepseek_chat.py` and `kimi_chat.py` for examples).

## Built-in Tools (18)

Every engine gets all of these automatically:

| Tool | What it does |
|------|-------------|
| `web_research` | Web search and synthesis via GPT-5 Mini (native web search) |
| `read_file` | Read files from workspace/ with line numbers |
| `write_file` | Write or append to files in workspace/ |
| `list_directory` | List files and directories in workspace/ with sizes |
| `search_files` | Regex search across files in workspace/ |
| `rag_index` | Index a document for semantic search (chunking + embeddings) |
| `rag_search` | Search indexed documents by semantic similarity |
| `rag_list` | List all indexed documents |
| `rag_delete` | Delete a document from the RAG index |
| `run_command` | Execute shell commands (with confirmation + blocklist) |
| `cron_create` | Schedule cron jobs |
| `cron_list` | List current cron jobs |
| `cron_delete` | Remove a cron job |
| `memory_write` | Save to persistent shared memory (JSON) |
| `memory_read` | Read stored memories |
| `memory_search` | Search memories by keyword |
| `memory_delete` | Delete a memory entry |
| `chat_log_read` | Query past conversations across all engines |

The model decides when to use them. Up to 10 tool-call rounds per turn.

## Telegram Bot

shellm runs natively as a Telegram bot with real-time streaming, HTML-formatted responses, and persistent sessions (conversations survive bot restarts).

```bash
# Set your bot token in .env, then:
./deepseek_chat.py --telegram
```

**Bot commands:** `/search`, `/memory`, `/remember`, `/recall`, `/logs`, `/model`, `/clear`, `/forget`, `/help`

Responses are automatically converted from Markdown to Telegram HTML with proper code blocks, bold, italic, links, and blockquotes.

## Run Modes

```bash
# Interactive (default)
./deepseek_chat.py

# Pipe a prompt
echo "What is 2+2?" | ./deepseek_chat.py --daemon stdin

# JSON output
echo "What is 2+2?" | ./deepseek_chat.py --daemon stdin --json

# Batch from file
./gpt5mini_chat.py --daemon file --input prompts.txt --output responses.txt

# Unix socket server (concurrent access)
./deepseek_chat.py --daemon socket --socket-path /tmp/deepseek.sock

# Telegram bot
./deepseek_chat.py --telegram
```

## Interactive Commands

| Command | Action |
|---------|--------|
| `/search <query>` | Force a web research workflow |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |

## Chat Logging

Every conversation turn is automatically saved to `chat_logs.json` with:
- Timestamp (KST/UTC+9), model name, run mode
- Full user input and assistant response
- All tool calls made during the turn
- Response duration in milliseconds

The LLM can read its own past logs via the `chat_log_read` tool.

## Architecture

```
BaseChatClient (base_chat.py, ~600 loc)
  |-- 18 built-in tools
  |-- Streaming + batch response handling
  |-- Optional reasoning/thinking display
  |-- Self-aware system prompt (knows its own codebase)
  |-- Chat logging (chat_logs.json)
  |-- System timezone awareness (KST)
  |-- Daemon mode (daemon_mode.py)
  +-- Telegram adapter (telegram_adapter.py + telegram_format.py)
        +-- Persistent sessions (telegram_sessions.json)

Modules:
  |-- file_tools.py      File ops (read/write/list/search) scoped to workspace/
  |-- rag_engine.py      Document indexing + semantic search (OpenAI embeddings + numpy)
  |-- command_runner.py   Shell command execution
  |-- cron_manager.py     Cron job management
  +-- memory_manager.py   Persistent shared memory (memory.json)

Engines: 15-50 lines each
  |-- deepseek_chat.py   Chat    (DeepSeek, +reasoner conditional logic)
  |-- kimi_chat.py       Code    (Kimi K2.5, +thinking mode toggle)
  +-- gpt5mini_chat.py   Research (GPT-5 Mini, config only + web search delegate)

workspace/   -- shellm's project directory for file output
rag_store/   -- RAG index, chunks, and embeddings
```

## License

MIT
