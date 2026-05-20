"""The single SheLLM agent.

One model profile, one tool list, one process. The agent calls all tools
itself — no delegation, no sub-agents.
"""

import base64
import json
import mimetypes
import os
import re
import time
from datetime import datetime, timezone, timedelta

from tools.executor import ToolContext, execute_tool
from tools.definitions import TOOLS
from agents.progress import progress_queue

CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_logs.json")
KST = timezone(timedelta(hours=9))

# Cap per tool result before appending to message history. Avoids quadratic
# context growth across multi-tool turns (each round re-sends every prior
# result to the model). 12 KB keeps page fetches and search snippets useful
# while bounding cost.
MAX_TOOL_RESULT_BYTES = 12_000


class BaseAgent:
    """SheLLM's only agent. Configuration comes from AgentConfig."""

    def __init__(self, config, client):
        self.config = config
        self.client = client
        self._current_tool_calls = []
        self._pending_images = []   # list of (mime, b64) attached for next turn

    @property
    def MODEL(self):
        return self.config.model

    def _print(self, *args, **kwargs):
        print(*args, **kwargs)

    # ── Chat logging ─────────────────────────────────────────────────

    def _log_chat(self, user_input, answer, duration_ms, error=None):
        entry = {
            "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
            "model": self.config.model,
            "agent": self.config.name,
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
            lines.append(f"[{e.get('timestamp', '?')}] model={e.get('model', '?')} ({e.get('duration_ms', 0):.0f}ms)")
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

    # ── Image attach ─────────────────────────────────────────────────

    def attach_image(self, path):
        """Queue an image for the next user turn. Returns an error string on
        failure, or None on success."""
        if not self.config.vision:
            return f"Model {self.config.model} is not configured for vision (set vision: true in agent_config.yaml)."
        if not os.path.isfile(path):
            return f"Image not found: {path}"
        mime, _ = mimetypes.guess_type(path)
        if not mime or not mime.startswith("image/"):
            return f"Not a recognized image type: {path}"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        self._pending_images.append((mime, b64))
        return None

    # ── API params ──────────────────────────────────────────────────

    def build_params(self, messages):
        params = {"model": self.config.model, "messages": messages, "stream": self.config.stream}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.supports_tools and TOOLS:
            # Vision-specific tools are hidden from non-vision profiles so the
            # model isn't tempted to call something it can't actually consume.
            tools = TOOLS
            if not self.config.vision:
                tools = [t for t in tools if t["function"]["name"] != "view_image"]
            params["tools"] = tools
        return params

    def build_no_tool_params(self, messages):
        params = {"model": self.config.model, "messages": messages, "stream": self.config.stream}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    # ── Tool execution ──────────────────────────────────────────────

    def _make_tool_context(self):
        return ToolContext(
            model=self.config.model,
            print_fn=self._print,
            queue_image=self._queue_image,
        )

    def _queue_image(self, mime, b64):
        """Callback invoked by view_image to attach an image to the next
        model round-trip. Drained in handle_tool_calls."""
        self._pending_images.append((mime, b64))

    def execute_tool(self, name, args):
        self._current_tool_calls.append({"tool": name, "args": args})
        return execute_tool(name, args, self._make_tool_context())

    @staticmethod
    def _parse_tool_arguments(raw):
        """Parse a tool-call arguments string.

        Some providers (notably Ollama) re-emit the *full* arguments string in
        every streaming chunk instead of sending JSON deltas, so the
        accumulated value ends up as concatenated objects like
        `{"a":1}{"a":1}`. raw_decode reads the first valid object and stops.
        """
        raw = (raw or "").strip() or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            args, _ = decoder.raw_decode(raw)
            return args

    @staticmethod
    def _cap_tool_result(text):
        """Bound a tool result before it lands in message history."""
        if not text or len(text) <= MAX_TOOL_RESULT_BYTES:
            return text
        elided = len(text) - MAX_TOOL_RESULT_BYTES
        return text[:MAX_TOOL_RESULT_BYTES] + f"\n\n[... truncated, {elided} bytes elided]"

    def handle_tool_calls(self, response_message, messages):
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            args = self._parse_tool_arguments(tool_call.function.arguments)
            result_text = self.execute_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": self._cap_tool_result(result_text),
            })
        # If a tool queued image content (view_image), inject it as a user
        # message before the next model round so the model can actually see
        # it. Tool messages can't carry image content per the OpenAI schema,
        # so a user-role injection is the standards-compliant way to surface
        # mid-turn vision input.
        if self._pending_images:
            content = [{"type": "text", "text": "[Image(s) attached by view_image:]"}]
            for mime, b64 in self._pending_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            self._pending_images = []
            messages.append({"role": "user", "content": content})
        return self.client.chat.completions.create(**self.build_params(messages))

    # ── Stream / batch handlers ─────────────────────────────────────

    @staticmethod
    def _extract_reasoning(obj):
        """Pull a reasoning trace from a streaming delta or batch message.

        Different providers use different field names and surface them
        differently through the OpenAI SDK:
          - DeepSeek/vLLM:  `reasoning_content` (as attr or in model_extra)
          - Ollama, some OpenRouter models:  `reasoning` (in model_extra,
            since the SDK schema doesn't know that field)
        """
        for name in ("reasoning_content", "reasoning"):
            val = getattr(obj, name, None)
            if val:
                return val
        extra = getattr(obj, "model_extra", None) or {}
        return extra.get("reasoning_content") or extra.get("reasoning")

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
                rc = self._extract_reasoning(delta)
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
            reasoning = self._extract_reasoning(message)
            if reasoning:
                self._print(f"\n[Thinking]\n{reasoning}")

        answer = message.content or ""
        self._print(f"\nAssistant: {answer}\n")
        return answer, None

    # ── Banner & system prompt ──────────────────────────────────────

    def format_banner(self):
        bits = [
            f"agent: {self.config.name}",
            f"model: {self.config.model}",
            f"provider: {self.config.provider}",
            f"stream: {self.config.stream}",
        ]
        if self.config.vision:
            bits.append("vision: on")
        if self.config.has_reasoning:
            bits.append("reasoning: on")
        return "SheLLM — " + ", ".join(bits)

    def _ensure_system_message(self, messages):
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

        # Capability flags, derived from the active profile.
        capability_lines = []
        if self.config.has_reasoning:
            capability_lines.append(
                "- You have native chain-of-thought reasoning. Use it for math, planning, "
                "and multi-step problems; the CLI surfaces it in a [Thinking] block."
            )
        if self.config.vision:
            capability_lines.append(
                "- You can read images natively. Use view_image(path) to load any image "
                "from workspace/ as visual input — DO NOT shell out to exiftool, "
                "identify, or tesseract. The user can also attach images directly with "
                "the /image command at the REPL."
            )
        capability_block = ("\n".join(capability_lines) + "\n\n") if capability_lines else ""

        system_content = (
            f"You are SheLLM, a single-agent CLI assistant. "
            f"Current date/time: {now} (Asia/Seoul, UTC+9).\n\n"

            "IDENTITY\n"
            f"- Running as: {self.config.model} via {self.config.provider} (profile: {self.config.name}).\n"
            "- One process, one model, one tool list — no sub-agents, no delegation. You execute every "
            "tool call yourself.\n"
            + capability_block +

            "SANDBOX\n"
            "- All file operations (read_file, write_file, list_directory, search_files) are restricted "
            "to workspace/. Paths are relative — use '.' for the root, or names like 'notes.txt' or "
            "'subdir/file.py'. Never use absolute paths like '/' or '/home'.\n"
            "- run_command also runs with cwd=workspace/, so `ls`, `pwd`, and relative shell paths see "
            "the same sandbox. The user must confirm every command; dangerous patterns are blocked.\n\n"

            "TOOLS\n"
            "- web_search(query, max_results=5): DuckDuckGo search. Returns titled snippets with URLs. "
            "Your first move for anything time-sensitive or outside training cutoff.\n"
            "- fetch_page(url): the readable text of a specific URL. Use after web_search for depth, "
            "or when the user provides a link.\n"
            + ("- view_image(path): load an image from workspace/ as visual input. "
               "Always use this for image questions instead of running exiftool/identify/tesseract.\n"
               if self.config.vision else "")
            + "- read_file / write_file / list_directory / search_files: workspace file ops "
              "(read_file is text-only; for images use view_image).\n"
            "- run_command(command, timeout=60): shell execution in workspace/. Pass timeout=300 for "
            "installs (apt/pip/npm) or long-running tasks.\n"
            "- memory_read / memory_write / memory_search / memory_delete: persistent memory that "
            "survives across sessions. Save anything useful about the user's preferences, projects, or "
            "ongoing work. Read it when context from prior sessions would help — don't read reflexively "
            "on greetings.\n"
            "- rag_index / rag_search / rag_list / rag_delete: semantic document store. Index long "
            "documents the user shares, then retrieve relevant chunks via rag_search later.\n"
            "- cron_list / cron_create / cron_delete: manage user crontab entries for scheduled tasks.\n"
            "- chat_log_read(last_n, keyword, model_filter): review past conversation history.\n\n"

            "EXECUTION\n"
            "- You have a 20-tool-call budget per user turn. Each tool result is capped at "
            f"{MAX_TOOL_RESULT_BYTES // 1000} KB before it lands in your history, so don't depend on "
            "huge per-call payloads — refine queries or fetch specific pages instead.\n"
            "- When a tool returns an error string, read it and adjust. Errors are observations, not "
            "stop signs."
        )
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_content})
        else:
            messages[0]["content"] = system_content

    # ── Core prompt processing ──────────────────────────────────────

    def _build_user_message(self, text):
        """Build a user message. If images are pending, use the multi-modal
        content format; otherwise a plain string."""
        if not self._pending_images:
            return {"role": "user", "content": text}

        content = [{"type": "text", "text": text}] if text else []
        for mime, b64 in self._pending_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        self._pending_images = []
        return {"role": "user", "content": content}

    def process_prompt(self, user_input, messages):
        original_input = user_input
        self._current_tool_calls = []
        self._total_streamed = ""
        t0 = time.time()
        self._ensure_system_message(messages)
        # Mark the boundary of this turn so we can roll back cleanly on error.
        # An exception inside the tool loop can leave the conversation in an
        # invalid state (e.g. a dangling assistant tool_call message with no
        # tool result), which would corrupt subsequent turns.
        turn_start = len(messages)
        messages.append(self._build_user_message(user_input))

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
                        msg_kwargs = dict(role="assistant", content=None, tool_calls=tc_list)
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
                response = self.client.chat.completions.create(**self.build_no_tool_params(messages))
                if self.config.stream:
                    answer, _, _ = self.handle_stream(response)
                else:
                    answer, _ = self.handle_batch(response)

            if self.config.stream and self._total_streamed and self._total_streamed != answer:
                answer = self._total_streamed

            if answer:
                answer = re.sub(r"<｜DSML｜function_calls>.*?</｜DSML｜function_calls>", "", answer, flags=re.DOTALL).strip()
                answer = re.sub(r"<｜DSML｜[^>]*>", "", answer).strip()

            if not answer and self._current_tool_calls:
                lines = [f"I used {len(self._current_tool_calls)} tool calls without producing a final answer.\n"]
                lines.append("Tools called:")
                for tc in self._current_tool_calls:
                    args_brief = json.dumps(tc.get("args", {}), ensure_ascii=False)[:120]
                    lines.append(f"  - {tc['tool']}({args_brief})")
                answer = "\n".join(lines)

            if answer:
                messages.append({"role": "assistant", "content": answer})

            duration_ms = (time.time() - t0) * 1000
            self._log_chat(original_input, answer, duration_ms)
            return answer

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000
            self._log_chat(original_input, None, duration_ms, error=e)
            self._print(f"\nError: {e}\n")
            # Roll back EVERY message added during this failed turn, not just
            # the user message. Otherwise an orphaned assistant(tool_calls)
            # entry breaks the conversation protocol on the next turn.
            del messages[turn_start:]
            return f"Error: {e}"

    # ── Interactive loop ────────────────────────────────────────────

    def run_interactive(self):
        messages = []
        print(self.format_banner())
        print("Commands: 'quit'/'exit' to end, 'clear' to reset, '/image <path> <prompt>' to attach an image.\n")

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
                self._pending_images = []
                print("[Conversation cleared]\n")
                continue
            if user_input.startswith("/image "):
                rest = user_input[len("/image "):].strip()
                parts = rest.split(None, 1)
                if not parts:
                    print("Usage: /image <path> <prompt>")
                    continue
                path = os.path.expanduser(parts[0])
                prompt_text = parts[1] if len(parts) > 1 else "Describe this image."
                err = self.attach_image(path)
                if err:
                    print(err)
                    continue
                user_input = prompt_text

            self.process_prompt(user_input, messages)
