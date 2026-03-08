#!/home/gslee/shellm/venv/bin/python3
"""Base chat client for LLM API Vault - shared logic for all model-specific scripts."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from openai import OpenAI
from cron_manager import cron_list, cron_create, cron_delete
from command_runner import run_command
from memory_manager import memory_read, memory_write, memory_search, memory_delete
from file_tools import read_file, write_file, list_directory, search_files
from rag_engine import rag_index, rag_search, rag_list, rag_delete
from mcp_manager import MCPManager
from task_scheduler import schedule_task, list_scheduled_tasks, cancel_scheduled_task, TaskScheduler

CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs.json")
KST = timezone(timedelta(hours=9))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_research",
            "description": "Research a topic using the web. Searches, reads pages, and returns a comprehensive answer. Use when you need current information, facts, or real-time data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project directory (~/shellm/) with line numbers. Can read source code, configs, etc. Use offset and limit for large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the project directory (e.g. 'base_chat.py', 'workspace/notes.txt')"},
                    "offset": {"type": "integer", "description": "Start line (0-based, default 0)"},
                    "limit": {"type": "integer", "description": "Max lines to return (default 200)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or append to a file in workspace/. Creates parent directories automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within workspace/"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode (default: overwrite)"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories in workspace/ with sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within workspace/ (default: root)"},
                    "recursive": {"type": "boolean", "description": "List recursively (default: false)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Regex search across files in workspace/. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Subdirectory to search in (default: all of workspace/)"},
                    "file_glob": {"type": "string", "description": "File glob filter, e.g. '*.py' (default: '*')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Index a document for semantic search. Chunks the text, generates embeddings, and stores for later retrieval. Use when the user wants to save a document for future Q&A.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The document text to index"},
                    "filename": {"type": "string", "description": "Name/label for the document"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"},
                },
                "required": ["text", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search indexed documents by semantic similarity. Returns the most relevant chunks. Use when the user asks about previously indexed documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default: 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_list",
            "description": "List all documents in the RAG index.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_delete",
            "description": "Delete a document from the RAG index by its doc_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID to delete (from rag_list)"},
                },
                "required": ["doc_id"],
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
    {
        "type": "function",
        "function": {
            "name": "claude_code",
            "description": "Delegate a task to Claude Code (Anthropic's AI coding agent). ONLY use when the user explicitly requests Claude Code. Claude Code has full filesystem, shell, and git access in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task or instruction for Claude Code"},
                    "working_directory": {"type": "string", "description": "Directory to run in (default: ~/shellm)"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kimi_code",
            "description": "Delegate a technical task to Kimi K2.5 (code engine with thinking mode). "
                           "Use for: coding, debugging, installation, system administration, "
                           "file editing, package management, server configuration, and any "
                           "hands-on technical work. Kimi has full tool access (shell commands, "
                           "file operations, etc.) and will execute the task autonomously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The technical task to delegate to Kimi"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_file",
            "description": "Send a file to the user via Telegram. Use this to deliver files you created "
                           "(images, documents, scripts, etc.) directly in the chat. The file must exist "
                           "in workspace/ or be an absolute path. For images (png/jpg/gif/webp), sends as "
                           "a photo; otherwise sends as a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path — relative to workspace/ (e.g. 'st_kitts_flag.png') or absolute"},
                    "caption": {"type": "string", "description": "Optional caption to send with the file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_list_servers",
            "description": "List all configured MCP servers and their connection status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_list_tools",
            "description": "List tools available from MCP servers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "Server name to filter (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a delayed task. Supports sending a Telegram message or running a shell command at a future time. Use delay_minutes for relative scheduling or scheduled_at for absolute time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": ["telegram_message", "shell_command"], "description": "Type of task to schedule"},
                    "payload": {"type": "object", "description": "Task data. For telegram_message: {\"message\": \"...\"}. For shell_command: {\"command\": \"...\"}. chat_id is auto-injected for Telegram."},
                    "delay_minutes": {"type": "number", "description": "Minutes from now to execute (e.g. 60 for 1 hour, 1800 for 30 hours)"},
                    "scheduled_at": {"type": "string", "description": "Absolute time in 'YYYY-MM-DD HH:MM:SS' format (KST). Alternative to delay_minutes."},
                },
                "required": ["task_type", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_tasks",
            "description": "List scheduled tasks. Filter by status: pending, done, failed, cancelled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status (default: pending)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_scheduled_task",
            "description": "Cancel a pending scheduled task by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "Task ID to cancel (from list_scheduled_tasks)"},
                },
                "required": ["task_id"],
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
        self._current_chat_id = None  # set by Telegram adapter per message

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
            params["tools"] = TOOLS + MCPManager.get_instance().get_tools()
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
        if name == "web_research":
            query = args.get("query", "")
            self._print(f"[Researching: {query}...]")
            try:
                from gpt5mini_chat import GPT5MiniChat
                researcher = GPT5MiniChat()
                researcher._silent = True
                response = researcher.client.responses.create(
                    model=researcher.MODEL,
                    input=query,
                    tools=[{"type": "web_search"}],
                )
                result = response.output_text or "(No results)"
            except Exception as e:
                result = f"Research error: {e}"
            self._print("[Research complete]")
            return result
        elif name == "read_file":
            return read_file(args.get("path", ""), offset=args.get("offset", 0), limit=args.get("limit", 200))
        elif name == "write_file":
            result = write_file(args.get("path", ""), args.get("content", ""), mode=args.get("mode", "overwrite"))
            self._print(result)
            return result
        elif name == "list_directory":
            return list_directory(args.get("path", "."), recursive=args.get("recursive", False))
        elif name == "search_files":
            return search_files(args.get("pattern", ""), path=args.get("path", "."), file_glob=args.get("file_glob", "*"))
        elif name == "rag_index":
            result = rag_index(args.get("text", ""), filename=args.get("filename", "untitled"), tags=args.get("tags"))
            self._print(result)
            return result
        elif name == "rag_search":
            return rag_search(args.get("query", ""), top_k=args.get("top_k", 5))
        elif name == "rag_list":
            return rag_list()
        elif name == "rag_delete":
            result = rag_delete(args.get("doc_id", ""))
            self._print(result)
            return result
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
            result_text = run_command(
                args.get("command", ""),
                timeout=args.get("timeout", 60),
                auto_approve=(self._mode != "interactive"),
            )
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
        elif name == "claude_code":
            return self._run_claude_code(args.get("prompt", ""), args.get("working_directory"))
        elif name == "kimi_code":
            return self._run_kimi_code(args.get("task", ""))
        elif name == "send_file":
            return self._send_file_telegram(args.get("path", ""), args.get("caption"))
        elif name == "mcp_list_servers":
            return MCPManager.get_instance().list_servers()
        elif name == "mcp_list_tools":
            return MCPManager.get_instance().list_server_tools(args.get("server"))
        elif name == "schedule_task":
            result_text = schedule_task(
                task_type=args.get("task_type", ""),
                payload=args.get("payload", {}),
                delay_minutes=args.get("delay_minutes"),
                scheduled_at=args.get("scheduled_at"),
                chat_id=self._current_chat_id,
            )
            self._print(result_text)
            return result_text
        elif name == "list_scheduled_tasks":
            return list_scheduled_tasks(status=args.get("status", "pending"))
        elif name == "cancel_scheduled_task":
            result_text = cancel_scheduled_task(args.get("task_id", 0))
            self._print(result_text)
            return result_text
        elif "__" in name:
            try:
                return MCPManager.get_instance().call_tool(name, args)
            except Exception as e:
                return f"MCP tool error ({name}): {e}"
        return f"Unknown tool: {name}"

    def _run_claude_code(self, prompt, working_directory=None):
        """Run Claude Code CLI with the given prompt and return its output."""
        import subprocess as _sp

        cwd = working_directory or os.path.dirname(os.path.abspath(__file__))
        if not os.path.isdir(cwd):
            return f"Directory not found: {cwd}"

        # Build env without CLAUDECODE to avoid nested-session block
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        self._print(f"[Claude Code] Running in {cwd}...")

        try:
            result = _sp.run(
                cmd, capture_output=True, text=True,
                timeout=300, cwd=cwd, env=env,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output:
                output = f"(no output, exit code: {result.returncode})"
            elif result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            return output
        except _sp.TimeoutExpired:
            return "Claude Code timed out after 5 minutes."
        except FileNotFoundError:
            return "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        except Exception as e:
            return f"Claude Code error: {e}"

    def _run_kimi_code(self, task):
        """Delegate a technical task to Kimi K2.5 with full tool access."""
        if not task.strip():
            return "No task provided."
        self._print(f"[Kimi Code] Delegating: {task[:100]}...")
        try:
            from kimi_chat import KimiChat
            kimi = KimiChat()
            kimi._silent = True
            kimi._mode = self._mode
            kimi._current_chat_id = self._current_chat_id
            messages = []
            answer = kimi.process_prompt(task, messages)
            self._print("[Kimi Code] Done.")
            return answer or "(Kimi produced no output)"
        except Exception as e:
            return f"Kimi Code error: {e}"

    def _send_file_telegram(self, path, caption=None):
        """Send a file to the current Telegram chat."""
        if not self._current_chat_id:
            return "Cannot send file: not in Telegram mode."

        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            return "Cannot send file: TELEGRAM_BOT_TOKEN not set."

        # Resolve path: relative to workspace/ or absolute
        workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
        if os.path.isabs(path):
            filepath = path
        else:
            filepath = os.path.join(workspace, path)

        if not os.path.isfile(filepath):
            return f"File not found: {path}"

        ext = os.path.splitext(filepath)[1].lower()
        is_image = ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp')

        try:
            import urllib.request
            import urllib.parse

            if is_image:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                field_name = "photo"
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
                field_name = "document"

            # Build multipart form data
            boundary = "----SheLLMFileBoundary"
            body_parts = []

            # chat_id field
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{self._current_chat_id}")

            # caption field
            if caption:
                body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}")

            # file field
            filename = os.path.basename(filepath)
            with open(filepath, "rb") as f:
                file_data = f.read()

            file_header = (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n"
            )

            # Assemble body
            body = b""
            for part in body_parts:
                body += part.encode() + b"\r\n"
            body += file_header.encode() + file_data + b"\r\n"
            body += f"--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())

            if result.get("ok"):
                return f"File sent successfully: {filename}"
            else:
                return f"Telegram API error: {result.get('description', 'unknown error')}"

        except Exception as e:
            return f"Error sending file: {e}"

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
            reasoning = "".join(reasoning_chunks) if reasoning_chunks else None
            return None, tool_calls_data, reasoning

        self._print("\n")
        return "".join(answer_chunks), None, None

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
            f"You are SheLLM, a helpful AI assistant. Current date/time: {now} (Asia/Seoul, UTC+9). "
            "You have persistent shared memory (use memory_read at the start of a conversation to recall "
            "who you are and what you know about the user). You can search the web (via web_research, "
            "delegated to GPT-5 Mini), execute shell commands, manage cron jobs, read files from the "
            "project directory (read_file), write/search files in workspace/, and review past chat logs "
            "with chat_log_read. "
            "You also have a RAG system — use rag_index to store documents for semantic search, "
            "and rag_search to retrieve relevant chunks later. Suggest indexing when the user sends documents. "
            "You also have MCP (Model Context Protocol) support — external servers can provide "
            "additional tools. Use mcp_list_servers to see connected servers. "
            "You have a claude_code tool that delegates tasks to Claude Code (Anthropic's AI coding agent). "
            "IMPORTANT: Only use claude_code when the user explicitly asks to use Claude Code or Claude. "
            "For all other coding tasks, delegate to kimi_code (NOT run_command directly).\n\n"
            "You have a kimi_code tool that delegates technical tasks to Kimi K2.5 (a code-specialized AI with "
            "thinking/reasoning mode). AUTOMATICALLY delegate the following types of tasks to kimi_code:\n"
            "- Writing, editing, or debugging code\n"
            "- Installing packages or software (apt, pip, npm, etc.)\n"
            "- System administration (services, configs, networking, Docker, etc.)\n"
            "- File operations that require multiple steps\n"
            "- Server setup and configuration\n"
            "- Git operations and repository management\n"
            "Do NOT ask the user for permission — just delegate technical tasks automatically. "
            "After Kimi completes the task, summarize what was done for the user.\n\n"
            "You have a send_file tool that sends files to the user via Telegram. "
            "ALWAYS use send_file after creating or saving a file that the user requested "
            "(images, documents, scripts, etc.) — do NOT just tell them the file path. "
            "Send the file directly so they can view/download it in the chat.\n\n"
            "Proactively save useful information about the user to memory for future sessions.\n\n"
            "TOOL STRATEGY: You have up to 20 tool calls per turn — use them wisely.\n"
            "- When you don't know how to install, configure, or use something, use web_research FIRST "
            "to find the correct commands, package names, and documentation. Do NOT guess commands blindly.\n"
            "- web_research is powered by GPT-5 Mini with live web search — it can find package install "
            "instructions, GitHub repos, API docs, and current information that you may not know.\n"
            "- Plan your approach: research first (1-2 calls), then execute with confidence (remaining calls).\n"
            "- For package installations (apt, pip, npm), always set timeout=300 in run_command.\n"
            "- If a run_command fails, research the error with web_research before retrying.\n"
            "- Prefer pip/apt install commands from official docs over cloning random repos.\n\n"
            "SELF-AWARENESS: Your own source code lives at ~/shellm/. You ARE SheLLM — "
            "when the user mentions 'your backend', 'your code', or 'your system', they mean YOUR files. "
            "Key files:\n"
            "  - base_chat.py — your core engine (tools, streaming, prompt processing)\n"
            "  - deepseek_chat.py — Chat engine config\n"
            "  - kimi_chat.py — Code engine config\n"
            "  - gpt5mini_chat.py — Research engine config\n"
            "  - telegram_adapter.py — your Telegram bot interface\n"
            "  - telegram_format.py — Markdown-to-HTML formatter for Telegram\n"
            "  - memory_manager.py — your persistent memory system (shellm.db)\n"
            "  - task_scheduler.py — heartbeat scheduler for delayed tasks (schedule_task, list/cancel)\n"
            "  - file_tools.py — file operations (read from project dir, write/list/search in workspace/)\n"
            "  - rag_engine.py — RAG document indexing and semantic search\n"
            "  - command_runner.py — shell command execution\n"
            "  - cron_manager.py — cron job management\n"
            "  - daemon_mode.py — stdin/file/socket daemon modes\n"
            "  - workspace/ — your project directory for creating files and output\n"
            "You can use run_command to: read your own source (`cat ~/shellm/base_chat.py`), "
            "check git status/diff/log, view your Telegram bot logs (`cat /tmp/shellm_bot.log`), "
            "list files, inspect your memory.json, and manage your own processes. "
            "You can also modify your own config files in workspace/ or create projects there. "
            "You have a task scheduler — use schedule_task to send a Telegram message or run a shell "
            "command at a future time (e.g. 'message me in 30 minutes', 'run backup in 2 hours'). "
            "Use list_scheduled_tasks and cancel_scheduled_task to manage them. "
            "For read_file, you can read any file in the project directory (e.g. read_file('base_chat.py')). "
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

            for _ in range(20):
                if self.STREAM:
                    answer, tool_data, reasoning = self.handle_stream(response)
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
                        msg_kwargs = dict(
                            role="assistant", content=None, tool_calls=tc_list,
                        )
                        if reasoning and self.HAS_REASONING:
                            msg_kwargs["reasoning_content"] = reasoning
                        msg = ChatCompletionMessage(**msg_kwargs)
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
                    answer, _, _ = self.handle_stream(response)
                else:
                    answer, _ = self.handle_batch(response)

            # Strip DeepSeek internal markup that leaks into content
            if answer:
                answer = re.sub(r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", "", answer, flags=re.DOTALL).strip()
                answer = re.sub(r"<｜DSML｜[^>]*>", "", answer).strip()

            # If still no answer after all attempts, build a diagnostic
            if not answer and self._current_tool_calls:
                tool_errors = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") == "tool":
                        content = msg.get("content", "")
                        if any(kw in content for kw in (
                            "Error", "error", "BLOCKED", "not found",
                            "timed out", "Path escapes", "Traceback",
                        )):
                            tool_errors.append(content[:300])

                lines = [f"I used {len(self._current_tool_calls)} tool calls without producing a final answer.\n"]
                lines.append("Tools called:")
                for tc in self._current_tool_calls:
                    args_brief = json.dumps(tc.get("args", {}), ensure_ascii=False)[:120]
                    lines.append(f"  - {tc['tool']}({args_brief})")
                if tool_errors:
                    lines.append("\nErrors encountered:")
                    for err in tool_errors[-3:]:
                        lines.append(f"  >> {err}")
                answer = "\n".join(lines)
                answer += "\n\nPlease retry or refine your request. I'll continue from where I left off."

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
            return f"Error: {e}"

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
