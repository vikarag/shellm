<p align="center">
  <img src="shellm-logo.svg" alt="SheLLM" width="360">
</p>

**SheLLM** (pronounced *shell-el-em*) is a minimal single-agent CLI for tool-using LLMs. One process, one model, one tool list — no delegation, no sub-agents, no Telegram, no web UI, no MCP.

```bash
./shellm.py                              # interactive REPL
./shellm.py --agent shellm-openrouter    # pick a different profile
./shellm.py --image diagram.png "explain this"
./shellm.py --list-agents
```

## Install

```bash
git clone https://github.com/vikarag/SheLLM.git
cd SheLLM
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Drop your API key(s) into .env, then:
./shellm.py
```

## Model profiles

Profiles live in `agent_config.yaml`. Any OpenAI-compatible provider works — DeepSeek, OpenRouter, Moonshot/Kimi, Ollama, vLLM, LM Studio, anything that speaks `/v1/chat/completions`. Set `provider`, `model`, `base_url`, `api_key_env`.

```yaml
default: shellm-deepseek   # profile used when --agent is not passed

agents:
  - name: shellm-deepseek
    provider: deepseek
    model: deepseek-chat
    api_key_env: SHELLM_DEEPSEEK_API_KEY
    base_url: "https://api.deepseek.com"
    stream: true
    temperature: 0.1
    supports_tools: true
    has_reasoning: true
    vision: false
```

Switch profiles at startup with `--agent <name>`. Only one runs at a time. To change which profile boots by default, edit the top-level `default:` field — no provider is special, DeepSeek is just one option among the four shipped (OpenRouter, Kimi, Ollama).

### Vision

Set `vision: true` on a profile whose model accepts image inputs (GPT-4o-class, Claude via OpenRouter, Kimi-VL, Gemini, llava-class Ollama models). Then attach an image:

```bash
# One-shot:
./shellm.py --image diagram.png "explain this architecture"

# In the REPL:
You: /image diagram.png explain this
```

Images go inline as multi-modal content — same chat turn, no separate model call.

## Tools (19)

The agent decides when to use them. Up to 20 tool-call rounds per turn.

| Tool | What it does |
|---|---|
| `web_search` | DuckDuckGo search (via the `ddgs` package) |
| `fetch_page` | Read a specific URL's text content (urllib + BeautifulSoup) |
| `read_file` / `write_file` / `list_directory` / `search_files` | Workspace and project file access |
| `run_command` | Shell execution (user confirmation; dangerous commands blocked) |
| `rag_index` / `rag_search` / `rag_list` / `rag_delete` | Semantic document store |
| `cron_list` / `cron_create` / `cron_delete` | Recurring jobs via the system crontab |
| `memory_read` / `memory_write` / `memory_search` / `memory_delete` | Persistent memory across sessions |
| `chat_log_read` | Review past conversations |

## REPL commands

| Command | Action |
|---|---|
| `/image <path> <prompt>` | Attach an image to the next turn (needs `vision: true`) |
| `clear` | Reset conversation history |
| `quit` / `exit` | End session |

## Layout

```
shellm.py                  CLI entry point (REPL + one-shot + --image)
agent_config.yaml          Model profiles
.env                       API keys
requirements.txt
chat_logs.json             Per-turn log (auto-created)
workspace/                 Where the agent writes files

agents/
  config.py                AgentConfig dataclass + YAML loader
  registry.py              Lazy agent creation
  base_agent.py            The one and only agent — streaming, tool loop, vision, logging
  progress.py              Tool-call progress events

tools/
  definitions.py           Tool JSON schemas
  executor.py              Dispatch + handlers

command_runner.py          Shell execution
cron_manager.py            Cron job management
file_tools.py              File ops in workspace/ and project dir
memory_manager.py          Persistent memory
rag_engine.py              RAG indexing + retrieval
```

## License

MIT
