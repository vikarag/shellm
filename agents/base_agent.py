"""Base agent class for SheLLM multi-agent system."""

import json
import os
import re
import time
from datetime import datetime, timezone, timedelta

from tools.executor import ToolContext, execute_tool
from tools.tool_sets import get_tool_set
from agents.progress import progress_queue

CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_logs.json")
KST = timezone(timedelta(hours=9))


class BaseAgent:
    """Base class for all SheLLM agents.

    Unlike BaseChatClient, configuration comes from AgentConfig rather than class variables.
    """

    def __init__(self, config, client, registry=None):
        """
        Args:
            config: AgentConfig dataclass
            client: OpenAI client instance (pre-configured with API key)
            registry: AgentRegistry instance for delegation
        """
        self.config = config
        self.client = client
        self.registry = registry
        self._silent = False
        self._current_tool_calls = []
        self._mode = "interactive"
        self._on_token = None
        self._current_chat_id = None
        self._plan_text = None

    @property
    def MODEL(self):
        return self.config.model

    @property
    def BANNER_NAME(self):
        return self.config.name

    def _print(self, *args, **kwargs):
        if not self._silent:
            print(*args, **kwargs)

    # ── Chat logging ─────────────────────────────────────────────────

    def _log_chat(self, user_input, answer, duration_ms, error=None):
        entry = {
            "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
            "model": self.config.model,
            "agent": self.config.name,
            "mode": self._mode,
            "stream": self.config.stream,
            "temperature": self.config.temperature,
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

    @staticmethod
    def _read_chat_logs_static(last_n=10, keyword=None, model_filter=None):
        """Static version for use by tool executor."""
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

    def _read_chat_logs(self, last_n=10, keyword=None, model_filter=None):
        return self._read_chat_logs_static(last_n, keyword, model_filter)

    # ── Tool context ─────────────────────────────────────────────────

    def _make_tool_context(self):
        return ToolContext(
            registry=self.registry,
            current_chat_id=self._current_chat_id,
            mode=self._mode,
            model=self.config.model,
            plan_text=self._plan_text,
            print_fn=self._print,
        )

    # ── API params ──────────────────────────────────────────────────

    def build_params(self, messages):
        params = {"model": self.config.model, "messages": messages, "stream": self.config.stream}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.supports_tools:
            tools = get_tool_set(self.config.tool_set)
            # Add MCP tools if applicable
            if self.config.tool_set in ("full",):
                from mcp_manager import MCPManager
                tools = tools + MCPManager.get_instance().get_tools()
            if tools:
                params["tools"] = tools
        return params

    def build_no_tool_params(self, messages):
        params = {"model": self.config.model, "messages": messages, "stream": self.config.stream}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    # ── Tool execution ──────────────────────────────────────────────

    def execute_tool(self, name, args):
        tool_record = {"tool": name, "args": args}
        self._current_tool_calls.append(tool_record)
        ctx = self._make_tool_context()
        return execute_tool(name, args, ctx)

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

            if self.config.has_reasoning:
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
                    self._on_token(self._total_streamed + "".join(answer_chunks))

        self._total_streamed += "".join(answer_chunks)

        if tool_calls_data:
            reasoning = "".join(reasoning_chunks) if reasoning_chunks else None
            return None, tool_calls_data, reasoning

        self._print("\n")
        return "".join(answer_chunks), None, None

    def handle_batch(self, response):
        message = response.choices[0].message

        if getattr(message, "tool_calls", None):
            return None, message

        if self.config.has_reasoning:
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                self._print(f"\n[Thinking]\n{reasoning}")

        answer = message.content or ""
        self._print(f"\nAssistant: {answer}\n")
        return answer, None

    # ── Banner ──────────────────────────────────────────────────────

    def format_banner(self):
        return f"{self.config.name} (model: {self.config.model}, stream: {self.config.stream}, temp: {self.config.temperature or 'default'})"

    # ── System message ──────────────────────────────────────────────

    def _ensure_system_message(self, messages):
        """Generate role-appropriate system prompt based on config.system_role."""
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

        base = (
            f"You are SheLLM, a helpful AI assistant. Current date/time: {now} (Asia/Seoul, UTC+9). "
            "You have persistent shared memory (use memory_read at the start of a conversation to recall "
            "who you are and what you know about the user)."
        )

        role = self.config.system_role

        if role == "primary":
            system_content = base + (
                " You can search the web (via delegate_websearch), fetch specific web pages (via fetch_page), "
                "analyze images (via delegate_image), "
                "conduct research (via delegate_research), use deep reasoning (via delegate_reason), "
                "and delegate complex coding tasks to Claude Code (via delegate_claude). "
                "You can execute shell commands, manage cron jobs, read files from the "
                "project directory (read_file), write/search files in workspace/, and review past chat logs "
                "with chat_log_read. "
                "You also have a RAG system — use rag_index to store documents for semantic search, "
                "and rag_search to retrieve relevant chunks later. Suggest indexing when the user sends documents. "
                "You also have MCP (Model Context Protocol) support — external servers can provide "
                "additional tools. Use mcp_list_servers to see connected servers.\n\n"
                "You have a send_file tool that sends files to the user via Telegram. "
                "ALWAYS use send_file after creating or saving a file that the user requested.\n\n"
                "Proactively save useful information about the user to memory for future sessions.\n\n"
                "TOOL STRATEGY: You have up to 20 tool calls per turn — use them wisely.\n"
                "- When you don't know how to do something, use delegate_websearch FIRST.\n"
                "- When you have a specific URL to read, use fetch_page to get the full page content.\n"
                "- delegate_research for academic topics, delegate_reason for complex planning.\n"
                "- delegate_claude for code analysis, refactoring, debugging, or tasks in specific project directories.\n"
                "- For package installations (apt, pip, npm), always set timeout=300 in run_command.\n\n"
                "SELF-AWARENESS: Your own source code lives at ~/shellm/.\n"
                "You have a task scheduler — use schedule_task to send a Telegram message or run a shell "
                "command at a future time. "
                f"You are currently running as: {self.config.model} (agent: {self.config.name})."
            )
        elif role == "updater":
            system_content = base + (
                " You are the updater agent. Your role is to inform the user about the current status "
                "of ongoing tasks when the main chat agent is busy. You can read memory and chat logs "
                "to provide context. Be concise and helpful."
                f"\n\nYou are currently running as: {self.config.model} (agent: {self.config.name})."
            )
        elif role == "mcp_worker":
            system_content = (
                f"You are an MCP worker agent ({self.config.name}). Current date/time: {now}. "
                "You handle MCP tool calls for your assigned servers. Execute tools and return results."
            )
        elif role == "reasoner":
            system_content = (
                f"You are a deep reasoning agent. Current date/time: {now}. "
                "Analyze problems thoroughly, break them into steps, and provide structured plans "
                "with clear reasoning traces. Focus on accuracy and completeness."
            )
        elif role == "vision":
            system_content = (
                f"You are an image analysis agent. Current date/time: {now}. "
                "Describe and analyze images in detail."
            )
        elif role == "websearch":
            system_content = (
                f"You are a web search agent. Current date/time: {now}. "
                "Search the web and provide comprehensive, accurate answers based on current information."
            )
        elif role == "researcher":
            system_content = (
                f"You are an academic research agent. Current date/time: {now}. "
                "Conduct thorough research with a focus on accuracy, citing sources when possible. "
                "Prioritize academic and authoritative sources."
            )
        else:
            system_content = base + f"\nAgent: {self.config.name}, Model: {self.config.model}."

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

    # ── Core prompt processing ──────────────────────────────────────

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
        self._total_streamed = ""
        t0 = time.time()
        self._ensure_system_message(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(**self.build_params(messages))
            answer = None

            for _ in range(20):
                if self.config.stream:
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
                        if reasoning and self.config.has_reasoning:
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
                if self.config.stream:
                    answer, _, _ = self.handle_stream(response)
                else:
                    answer, _ = self.handle_batch(response)

            # For streaming, use full accumulated text across all rounds
            if self.config.stream and self._total_streamed and self._total_streamed != answer:
                answer = self._total_streamed

            # Strip DeepSeek internal markup
            if answer:
                answer = re.sub(r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", "", answer, flags=re.DOTALL).strip()
                answer = re.sub(r"<｜DSML｜[^>]*>", "", answer).strip()

            # Diagnostic if no answer
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
