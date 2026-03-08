#!/home/gslee/llm-api-vault/venv/bin/python3
"""Base chat client for LLM API Vault - shared logic for all model-specific scripts."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from openai import OpenAI
from web_search import search, fetch_page, format_results
from cron_manager import cron_list, cron_create, cron_delete
from command_runner import run_command
from memory_manager import memory_read, memory_write, memory_search, memory_delete

CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs.json")
KST = timezone(timedelta(hours=9))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when the user asks about recent events, real-time data, or anything you're unsure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Fetch and read the full text content of a webpage. Use after web_search to read detailed content from a specific search result URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_list",
            "description": "List all current cron jobs for this user.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_create",
            "description": "Create a new scheduled cron job. The user will be asked to confirm before it is added.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schedule": {"type": "string", "description": "Cron schedule expression, e.g. '0 9 * * *' for daily at 9am, '*/5 * * * *' for every 5 minutes"},
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["schedule", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cron_delete",
            "description": "Delete a cron job by its index number (from cron_list). The user will be asked to confirm before deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Index of the cron job to delete"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command on the system. The user will be asked to confirm before execution. Dangerous commands (rm -rf, shutdown, etc.) are automatically blocked. Use for tasks like sending emails, scheduling with 'at', checking system info, file operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max execution time in seconds (default 60)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read the shared memory file. Returns all stored memories chronologically, or the last N entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent entries to return (0 = all, default 0)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save something to shared memory. Use this to remember user preferences, important facts, task results, or anything that should persist across sessions. All models share this memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information to remember"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search shared memory by keyword. Searches content and tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search term"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory entry by its index (0-based chronological order). Use memory_read first to find the index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Index of the memory to delete"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_log_read",
            "description": "Read past chat conversation logs. Returns recent chat history across all models and sessions. Use to recall what was discussed previously, find past answers, or check conversation context. Supports filtering by keyword and model name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent entries to return (default 10, max 50)"},
                    "keyword": {"type": "string", "description": "Optional keyword to filter logs (searches user input and assistant response)"},
                    "model_filter": {"type": "string", "description": "Optional model name to filter by (e.g. 'gpt-5-mini', 'deepseek-chat')"},
                },
            },
        },
    },
]


class BaseChatClient:
    """Base class for all LLM chat clients.

    Subclass must set: MODEL, BANNER_NAME, ENV_VAR.
    Optional overrides: BASE_URL, STREAM, TEMPERATURE, SUPPORTS_TOOLS, HAS_REASONING.
    """

    MODEL = None
    BANNER_NAME = None
    ENV_VAR = None
    BASE_URL = None
    STREAM = True
    TEMPERATURE = None
    SUPPORTS_TOOLS = True
    HAS_REASONING = False

    def __init__(self):
        api_key = os.environ.get(self.ENV_VAR, "YOUR_API_KEY_HERE")
        kwargs = {"api_key": api_key}
        if self.BASE_URL:
            kwargs["base_url"] = self.BASE_URL
        self.client = OpenAI(**kwargs)
        self._silent = False
        self._current_tool_calls = []
        self._mode = "interactive"
        self._on_token = None  # callback(accumulated_text) for streaming consumers

    def _print(self, *args, **kwargs):
        if not self._silent:
            print(*args, **kwargs)

    # ── Chat logging ─────────────────────────────────────────────────

    def _log_chat(self, user_input, answer, duration_ms, error=None):
        entry = {
            "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
            "model": self.MODEL,
            "mode": self._mode,
            "stream": self.STREAM,
            "temperature": self.TEMPERATURE,
            "user_input": user_input,
            "assistant_response": answer,
            "tool_calls": self._current_tool_calls.copy(),
            "duration_ms": round(duration_ms),
        }
        if error:
            entry["error"] = str(error)
        try:
            if os.path.exists(CHAT_LOG_FILE):
                with open(CHAT_LOG_FILE) as f:
                    logs = json.load(f)
            else:
                logs = []
            logs.append(entry)
            with open(CHAT_LOG_FILE, "w") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _read_chat_logs(self, last_n=10, keyword=None, model_filter=None):
        last_n = min(max(last_n, 1), 50)
        try:
            if not os.path.exists(CHAT_LOG_FILE):
                return "No chat logs yet."
            with open(CHAT_LOG_FILE) as f:
                logs = json.load(f)
        except Exception:
            return "Error reading chat logs."

        if model_filter:
            logs = [e for e in logs if model_filter.lower() in e.get("model", "").lower()]
        if keyword:
            kw = keyword.lower()
            logs = [
                e for e in logs
                if kw in (e.get("user_input") or "").lower()
                or kw in (e.get("assistant_response") or "").lower()
            ]

        logs = logs[-last_n:]
        if not logs:
            return "No matching chat logs found."

        lines = [f"Chat Logs ({len(logs)} entries):\n"]
        for e in logs:
            lines.append(f"[{e.get('timestamp', '?')}] model={e.get('model', '?')} mode={e.get('mode', '?')} ({e.get('duration_ms', 0):.0f}ms)")
            lines.append(f"  User: {e.get('user_input', '')[:200]}")
            resp = e.get("assistant_response") or "(no response)"
            lines.append(f"  Assistant: {resp[:300]}")
            if e.get("tool_calls"):
                tools_used = ", ".join(tc.get("tool", "?") for tc in e["tool_calls"])
                lines.append(f"  Tools used: {tools_used}")
            if e.get("error"):
                lines.append(f"  Error: {e['error']}")
            lines.append("")
        return "\n".join(lines)

    # ── API params ──────────────────────────────────────────────────

    def build_params(self, messages):
        params = {"model": self.MODEL, "messages": messages, "stream": self.STREAM}
        if self.TEMPERATURE is not None:
            params["temperature"] = self.TEMPERATURE
        if self.SUPPORTS_TOOLS:
            params["tools"] = TOOLS
        return params

    def build_no_tool_params(self, messages):
        params = {"model": self.MODEL, "messages": messages, "stream": self.STREAM}
        if self.TEMPERATURE is not None:
            params["temperature"] = self.TEMPERATURE
        return params

    # ── Tool execution ──────────────────────────────────────────────

    def execute_tool(self, name, args):
        tool_record = {"tool": name, "args": args}
        self._current_tool_calls.append(tool_record)
        if name == "web_search":
            query = args.get("query", "")
            self._print(f"[Searching: {query}...]")
            try:
                results = search(query)
                result_text = format_results(results)
            except Exception as e:
                result_text = f"Search error: {e}"
            self._print(result_text)
            return result_text
        elif name == "read_webpage":
            url = args.get("url", "")
            self._print(f"[Reading: {url}...]")
            try:
                text = fetch_page(url)
                if len(text) > 15000:
                    text = text[:15000] + "\n\n[... content truncated ...]"
                return text
            except Exception as e:
                return f"Fetch error: {e}"
        elif name == "cron_list":
            result_text = cron_list()
            self._print(result_text)
            return result_text
        elif name == "cron_create":
            result_text = cron_create(args.get("schedule", ""), args.get("command", ""))
            self._print(result_text)
            return result_text
        elif name == "cron_delete":
            result_text = cron_delete(args.get("index", 0))
            self._print(result_text)
            return result_text
        elif name == "run_command":
            result_text = run_command(args.get("command", ""), timeout=args.get("timeout", 60))
            self._print(result_text)
            return result_text
        elif name == "memory_read":
            return memory_read(last_n=args.get("last_n", 0))
        elif name == "memory_write":
            result_text = memory_write(args.get("content", ""), source=self.MODEL, tags=args.get("tags", []))
            self._print(result_text)
            return result_text
        elif name == "memory_search":
            return memory_search(args.get("keyword", ""))
        elif name == "memory_delete":
            result_text = memory_delete(args.get("index", 0))
            self._print(result_text)
            return result_text
        elif name == "chat_log_read":
            return self._read_chat_logs(
                last_n=args.get("last_n", 10),
                keyword=args.get("keyword"),
                model_filter=args.get("model_filter"),
            )
        return f"Unknown tool: {name}"

    def handle_tool_calls(self, response_message, messages):
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result_text = self.execute_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })
        return self.client.chat.completions.create(**self.build_params(messages))

    # ── Response handling ───────────────────────────────────────────

    def handle_stream(self, response):
        reasoning_chunks = []
        answer_chunks = []
        tool_calls_data = {}
        in_reasoning = False

        for chunk in response:
            delta = chunk.choices[0].delta

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
                continue

            if self.HAS_REASONING:
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    if not in_reasoning:
                        self._print("\n[Thinking]")
                        in_reasoning = True
                    self._print(rc, end="", flush=True)
                    reasoning_chunks.append(rc)
                    continue

            if delta.content:
                if in_reasoning:
                    self._print()
                    in_reasoning = False
                    self._print("\nAssistant: ", end="")
                elif not answer_chunks:
                    self._print("\nAssistant: ", end="")
                self._print(delta.content, end="", flush=True)
                answer_chunks.append(delta.content)
                if self._on_token:
                    self._on_token("".join(answer_chunks))

        if tool_calls_data:
            return None, tool_calls_data

        self._print("\n")
        return "".join(answer_chunks), None

    def handle_batch(self, response):
        message = response.choices[0].message

        if getattr(message, "tool_calls", None):
            return None, message

        if self.HAS_REASONING:
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                self._print(f"\n[Thinking]\n{reasoning}")

        answer = message.content or ""
        self._print(f"\nAssistant: {answer}\n")
        return answer, None

    # ── Banner ──────────────────────────────────────────────────────

    def format_banner(self):
        return f"{self.BANNER_NAME} (model: {self.MODEL}, stream: {self.STREAM}, temp: {self.TEMPERATURE or 'default'})"

    # ── Core prompt processing ──────────────────────────────────────

    def _ensure_system_message(self, messages):
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
        system_content = (
            f"You are shellm, a helpful AI assistant. Current date/time: {now} (Asia/Seoul, UTC+9). "
            "You have persistent shared memory (use memory_read at the start of a conversation to recall "
            "who you are and what you know about the user). You can search the web, execute shell commands, "
            "manage cron jobs, and review past chat logs with chat_log_read. "
            "Proactively save useful information about the user to memory for future sessions.\n\n"
            "SELF-AWARENESS: Your own source code lives at ~/llm-api-vault/. You ARE shellm — "
            "when the user mentions 'your backend', 'your code', or 'your system', they mean YOUR files. "
            "Key files:\n"
            "  - base_chat.py — your core engine (tools, streaming, prompt processing)\n"
            "  - deepseek_chat.py — Chat engine config\n"
            "  - kimi_chat.py — Code engine config\n"
            "  - gpt5mini_chat.py — Research engine config\n"
            "  - telegram_adapter.py — your Telegram bot interface\n"
            "  - telegram_format.py — Markdown-to-HTML formatter for Telegram\n"
            "  - memory_manager.py — your persistent memory system (memory.json)\n"
            "  - web_search.py — web search via Camoufox browser\n"
            "  - command_runner.py — shell command execution\n"
            "  - cron_manager.py — cron job management\n"
            "  - daemon_mode.py — stdin/file/socket daemon modes\n"
            "  - workspace/ — your project directory for creating files and output\n"
            "You can use run_command to: read your own source (`cat ~/llm-api-vault/base_chat.py`), "
            "check git status/diff/log, view your Telegram bot logs (`cat /tmp/shellm_bot.log`), "
            "list files, inspect your memory.json, and manage your own processes. "
            "You can also modify your own config files in workspace/ or create projects there. "
            f"You are currently running as: {self.MODEL} (engine: {self.BANNER_NAME})."
        )
        if self._mode == "telegram":
            system_content += (
                "\n\nTELEGRAM MODE: Format responses for mobile readability — "
                "short paragraphs, **bold** for key terms, `backticks` for code/commands, "
                "```language blocks for code. Use bullet points (- ) for lists. "
                "Be concise — avoid long walls of text."
            )
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_content})
        else:
            messages[0]["content"] = system_content

    def process_prompt(self, user_input, messages):
        original_input = user_input
        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if not query:
                return None
            user_input = (
                "Research the following by searching the web, reading the most relevant "
                "pages in detail, and providing a comprehensive answer:\n\n" + query
            )

        self._current_tool_calls = []
        t0 = time.time()
        self._ensure_system_message(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(**self.build_params(messages))
            answer = None

            for _ in range(10):
                if self.STREAM:
                    answer, tool_data = self.handle_stream(response)
                    if tool_data:
                        from openai.types.chat import ChatCompletionMessage
                        from openai.types.chat.chat_completion_message_tool_call import (
                            ChatCompletionMessageToolCall, Function,
                        )
                        tc_list = []
                        for idx in sorted(tool_data.keys()):
                            tc = tool_data[idx]
                            tc_list.append(ChatCompletionMessageToolCall(
                                id=tc["id"],
                                type="function",
                                function=Function(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"],
                                ),
                            ))
                        msg = ChatCompletionMessage(
                            role="assistant", content=None, tool_calls=tc_list,
                        )
                        response = self.handle_tool_calls(msg, messages)
                        continue
                else:
                    answer, tool_msg = self.handle_batch(response)
                    if tool_msg:
                        response = self.handle_tool_calls(tool_msg, messages)
                        continue
                break

            if not answer:
                response = self.client.chat.completions.create(
                    **self.build_no_tool_params(messages)
                )
                if self.STREAM:
                    answer, _ = self.handle_stream(response)
                else:
                    answer, _ = self.handle_batch(response)

            if answer:
                messages.append({"role": "assistant", "content": answer})

            duration_ms = (time.time() - t0) * 1000
            self._log_chat(original_input, answer, duration_ms)
            return answer

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            self._log_chat(original_input, None, duration_ms, error=e)
            self._print(f"\nError: {e}\n")
            messages.pop()
            return None

    # ── Interactive loop ────────────────────────────────────────────

    def run_interactive(self):
        messages = []
        print(self.format_banner())
        print("Type 'quit' or 'exit' to end. Type 'clear' to reset conversation.")
        print("Type '/search <query>' to search the web manually.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Bye!")
                break
            if user_input.lower() == "clear":
                messages.clear()
                print("[Conversation cleared]\n")
                continue

            self.process_prompt(user_input, messages)

    # ── CLI entry point ─────────────────────────────────────────────

    def run(self):
        parser = argparse.ArgumentParser(description=f"{self.BANNER_NAME} Chat")
        parser.add_argument(
            "--daemon", choices=["stdin", "file", "socket"],
            help="Run in daemon mode (stdin, file, or socket)",
        )
        parser.add_argument(
            "--json", action="store_true",
            help="Output responses as JSON (daemon mode)",
        )
        parser.add_argument("--input", help="Input file path (daemon file mode)")
        parser.add_argument("--output", help="Output file path (daemon file mode)")
        parser.add_argument(
            "--socket-path",
            help="Unix socket path (daemon socket mode)",
        )
        parser.add_argument(
            "--telegram", action="store_true",
            help="Run as Telegram bot",
        )
        args = parser.parse_args()

        if args.daemon:
            from daemon_mode import run_daemon
            run_daemon(self, args.daemon, args)
        elif args.telegram:
            from telegram_adapter import TelegramAdapter
            adapter = TelegramAdapter(self)
            adapter.run()
        else:
            self.run_interactive()
