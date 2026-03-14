#!/home/gslee/shellm/venv/bin/python3
"""Telegram bot adapter for SheLLM multi-agent system with streaming, vision, and file handling."""

import asyncio
import base64
import html as html_mod
import json
import os
import time
import urllib.request

from telegram_format import md_to_tg_html, split_message

WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_logs.json")
SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_sessions.json")
CHAT_ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".telegram_chat_id")


def _persist_chat_id(chat_id):
    """Save the most recent Telegram chat_id to disk for API server fallback."""
    try:
        with open(CHAT_ID_FILE, "w") as f:
            f.write(str(chat_id))
    except OSError:
        pass


def _extract_text(filepath):
    """Extract text content from various file formats."""
    ext = os.path.splitext(filepath)[1].lower()

    # Plain text formats
    text_exts = (
        '.txt', '.md', '.csv', '.json', '.py', '.js', '.ts', '.html', '.css',
        '.sh', '.yml', '.yaml', '.toml', '.cfg', '.ini', '.log', '.xml',
        '.sql', '.r', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.rb',
    )
    if ext in text_exts:
        with open(filepath, 'r', errors='replace') as f:
            return f.read()

    if ext == '.pdf':
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
            return "\n\n".join(parts)
        except ImportError:
            return "[Install pdfplumber to read PDFs: pip install pdfplumber]"

    if ext == '.docx':
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text)
        except ImportError:
            return "[Install python-docx to read DOCX: pip install python-docx]"

    # Fallback: try reading as text
    try:
        with open(filepath, 'r', errors='replace') as f:
            return f.read()
    except Exception:
        return None


def _fetch_usage(registry):
    """Fetch API balance from providers and local usage stats."""
    from datetime import datetime

    lines = ["<b>API Balances</b>\n"]

    # Collect unique DeepSeek API keys from all DeepSeek agents
    seen_keys = set()
    all_configs = registry.get_all_configs()
    for agent_name, config in all_configs.items():
        if config.provider != "deepseek":
            continue
        key = os.environ.get(config.api_key_env, "")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("is_available"):
                for info in data.get("balance_infos", []):
                    total = info.get("total_balance", "?")
                    granted = info.get("granted_balance", "0")
                    lines.append(f"• <b>DeepSeek</b> ({agent_name}): ${total} remaining")
                    if float(granted) > 0:
                        lines.append(f"  (granted: ${granted})")
            else:
                lines.append(f"• <b>DeepSeek</b> ({agent_name}): Account unavailable")
        except Exception as e:
            lines.append(f"• <b>DeepSeek</b> ({agent_name}): Error — {e}")

    # OpenAI agents — no balance API for standard keys
    openai_shown = False
    for agent_name, config in all_configs.items():
        if config.provider == "openai" and not openai_shown:
            openai_key = os.environ.get(config.api_key_env, "")
            if openai_key:
                lines.append(
                    '• <b>OpenAI</b> (image/websearch/researcher): '
                    '<a href="https://platform.openai.com/usage">Check dashboard</a>'
                )
                openai_shown = True
                break

    # Local usage stats from chat logs
    lines.append("\n<b>Usage Stats</b>\n")
    try:
        if os.path.exists(CHAT_LOG_FILE):
            with open(CHAT_LOG_FILE) as f:
                logs = json.load(f)

            today = datetime.now().strftime("%Y-%m-%d")
            model_stats = {}
            for entry in logs:
                model = entry.get("model", "unknown")
                if model not in model_stats:
                    model_stats[model] = {"total": 0, "today": 0, "total_ms": 0}
                model_stats[model]["total"] += 1
                model_stats[model]["total_ms"] += entry.get("duration_ms", 0)
                if entry.get("timestamp", "").startswith(today):
                    model_stats[model]["today"] += 1

            for model, s in sorted(model_stats.items()):
                avg = s["total_ms"] / s["total"] / 1000 if s["total"] else 0
                lines.append(
                    f"• <code>{model}</code>: "
                    f"{s['total']} total, {s['today']} today, "
                    f"avg {avg:.1f}s"
                )
            lines.append(f"\n<i>Total: {len(logs)} interactions logged</i>")
        else:
            lines.append("No chat logs yet.")
    except Exception as e:
        lines.append(f"Error reading logs: {e}")

    return "\n".join(lines)


class TelegramAdapter:
    """Connects an AgentRegistry to a Telegram bot interface.

    Features:
        - Real-time streaming via sendMessageDraft (Bot API 9.5)
        - HTML-formatted responses (Markdown -> Telegram HTML)
        - Document handling (PDF, DOCX, CSV, TXT, code files)
        - Image analysis via shellm-image agent (vision)
        - Multi-agent routing: chat, updater, image, reasoner
        - Workspace file management

    Usage:
        ./shellm.py --telegram
    """

    MAX_SESSION_MESSAGES = 50

    def __init__(self, registry, bot_token=None):
        self.registry = registry
        self.primary = registry.get_agent("shellm-chat")
        self.updater = registry.get_agent("shellm-updater")
        self.primary._silent = True
        self.primary._mode = "telegram"
        self.updater._silent = True
        self.updater._mode = "telegram"
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.sessions = {}  # chat_id -> messages list
        self._pending_plans = {}  # chat_id -> original prompt awaiting confirmation
        self._busy_chats: set = set()  # chat_ids where shellm-chat is processing
        self._load_sessions()

    # Keep self.client as alias for backward compat with command handlers that use adapter.client
    @property
    def client(self):
        return self.primary

    def _load_sessions(self):
        """Load sessions from disk on startup."""
        try:
            if os.path.exists(SESSIONS_FILE):
                with open(SESSIONS_FILE) as f:
                    data = json.load(f)
                # JSON keys are strings — convert back to int chat_ids
                self.sessions = {int(k): v for k, v in data.items()}
        except Exception:
            self.sessions = {}

    def _save_sessions(self):
        """Persist sessions to disk."""
        try:
            with open(SESSIONS_FILE, "w") as f:
                json.dump(self.sessions, f, ensure_ascii=False)
        except Exception:
            pass

    def _trim_session(self, messages):
        """Cap a session at MAX_SESSION_MESSAGES, preserving the system message if present."""
        if len(messages) <= self.MAX_SESSION_MESSAGES:
            return messages
        # Keep system message (index 0) if it exists, plus the most recent messages
        if messages and messages[0].get("role") == "system":
            return [messages[0]] + messages[-(self.MAX_SESSION_MESSAGES - 1):]
        return messages[-self.MAX_SESSION_MESSAGES:]

    def get_or_create_session(self, chat_id):
        if chat_id not in self.sessions:
            self.sessions[chat_id] = []
        return self.sessions[chat_id]

    async def _send_response(self, message, text):
        """Send an LLM response with HTML formatting, falling back to plain text."""
        try:
            formatted = md_to_tg_html(text)
            for chunk in split_message(formatted):
                await message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            for chunk in split_message(text):
                await message.reply_text(chunk)

    async def handle_message_streaming(self, chat_id, text, bot):
        """Handle a message: run LLM in background thread, return final answer."""
        messages = self.get_or_create_session(chat_id)

        self.primary._current_chat_id = chat_id
        self._busy_chats.add(chat_id)

        loop = asyncio.get_event_loop()
        try:
            final_answer = await loop.run_in_executor(
                None, self.primary.process_prompt, text, messages
            )
        finally:
            self._busy_chats.discard(chat_id)

        # Persist session after processing
        self.sessions[chat_id] = self._trim_session(messages)
        self._save_sessions()

        return final_answer or "(No response)"

    async def handle_updater_message(self, chat_id, text, bot):
        """Handle a message via the updater agent when primary is busy."""
        messages = []
        loop = asyncio.get_event_loop()
        final_answer = await loop.run_in_executor(
            None, self.updater.process_prompt, text, messages
        )
        return final_answer or "(No response)"

    def _analyze_image(self, b64_data, prompt):
        """Send image to shellm-image agent for vision analysis."""
        image_agent = self.registry.get_agent("shellm-image")
        return image_agent.analyze_image(b64_data, prompt)

    def run(self):
        """Start the Telegram bot with streaming support."""
        if not self.bot_token:
            raise RuntimeError(
                "Set TELEGRAM_BOT_TOKEN environment variable.\n"
                "Then run: ./shellm.py --telegram"
            )

        try:
            from telegram import Update, BotCommand
            from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
            from telegram.constants import ChatAction
        except ImportError:
            raise RuntimeError("pip install python-telegram-bot")

        adapter = self

        # Import memory functions for /memory and /forget commands
        from memory_manager import memory_read, memory_search

        # ── Message handlers ───────────────────────────────────────

        async def _on_message(update: Update, context):
            chat_id = update.effective_chat.id
            text = update.message.text.strip()
            if not text:
                return
            bot = context.bot

            # Persist chat_id so the API server can use it as fallback
            _persist_chat_id(chat_id)

            # Check if there's a pending plan awaiting confirmation
            if chat_id in adapter._pending_plans:
                lower = text.lower().strip()
                cancel_words = ("no", "cancel", "취소", "아니", "stop", "nope")
                confirm_words = ("yes", "y", "go", "ok", "okay", "proceed", "do it",
                                 "execute", "run", "확인", "실행", "ㅇㅇ", "응", "네", "좋아", "해줘", "고")

                if any(lower.startswith(w) for w in cancel_words):
                    del adapter._pending_plans[chat_id]
                    await update.message.reply_text("Plan cancelled.")
                    return
                elif any(lower.startswith(w) for w in confirm_words):
                    plan_data = adapter._pending_plans.pop(chat_id)
                    original_prompt = plan_data["prompt"]
                    plan_text = plan_data.get("plan", "")

                    execution_prompt = (
                        f"[PLAN EXECUTION MODE]\n"
                        f"You are executing an approved plan. Here is the plan:\n\n"
                        f"{plan_text}\n\n"
                        f"EXECUTION GUIDELINES:\n"
                        f"1. After completing each major step, call the report_progress tool "
                        f"with the step number, total steps, title, and what was accomplished.\n"
                        f"2. Actively use web_research for any research, data gathering, "
                        f"fact-checking, or finding resources/documentation needed during execution.\n"
                        f"3. Combine research and implementation as needed.\n\n"
                        f"Now execute: {original_prompt}"
                    )

                    adapter.primary._plan_text = plan_text
                    await update.message.chat.send_action(ChatAction.TYPING)
                    try:
                        response = await adapter.handle_message_streaming(
                            chat_id, execution_prompt, bot
                        )
                        await adapter._send_response(update.message, response)
                    except Exception as e:
                        await update.message.reply_text(f"Error: {e}")
                    finally:
                        adapter.primary._plan_text = None
                    return
                else:
                    # Treat as feedback — regenerate plan with the feedback
                    original_prompt = adapter._pending_plans[chat_id]["prompt"]
                    feedback_prompt = (
                        f"The user gave feedback on your previous plan. "
                        f"Original task: {original_prompt}\n\n"
                        f"User feedback: {text}\n\n"
                        "Create a REVISED execution plan incorporating this feedback. "
                        "Do NOT execute anything yet — only present the updated plan. "
                        "End with: 'Reply **yes** to execute, **no** to cancel, or provide more feedback.'"
                    )
                    await update.message.chat.send_action(ChatAction.TYPING)
                    try:
                        loop = asyncio.get_event_loop()
                        plan_messages = []
                        adapter.primary._ensure_system_message(plan_messages)
                        response = await loop.run_in_executor(
                            None, lambda: adapter.primary.process_prompt(feedback_prompt, plan_messages)
                        )
                        # Update stored plan text with the revised plan
                        adapter._pending_plans[chat_id]["plan"] = response or ""
                        await adapter._send_response(update.message, response or "(No plan generated)")
                    except Exception as e:
                        await update.message.reply_text(f"Error: {e}")
                    return

            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                # Route to updater if primary is busy, otherwise use primary
                if chat_id in adapter._busy_chats:
                    response = await adapter.handle_updater_message(chat_id, text, bot)
                else:
                    response = await adapter.handle_message_streaming(chat_id, text, bot)
                await adapter._send_response(update.message, response)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _on_photo(update: Update, context):
            """Handle photos — route to shellm-image agent for vision analysis."""
            photo = update.message.photo[-1]  # highest resolution
            caption = update.message.caption or "Describe and analyze this image in detail."
            chat_id = update.effective_chat.id

            await update.message.chat.send_action(ChatAction.TYPING)

            # Download image
            file = await photo.get_file()
            filepath = os.path.join(WORKSPACE, f"photo_{int(time.time())}.jpg")
            await file.download_to_drive(filepath)

            # Read and encode
            with open(filepath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            try:
                response = await asyncio.to_thread(adapter._analyze_image, b64, caption)

                # Add to conversation history so main LLM has context
                messages = adapter.get_or_create_session(chat_id)
                messages.append({"role": "user", "content": f"[Sent an image] {caption}"})
                messages.append({"role": "assistant", "content": response})
                adapter._save_sessions()

                await adapter._send_response(update.message, response)
            except Exception as e:
                await update.message.reply_text(f"Error analyzing image: {e}")

        async def _on_document(update: Update, context):
            """Handle documents — extract text, pass to LLM for analysis."""
            doc = update.message.document
            caption = update.message.caption or ""
            chat_id = update.effective_chat.id
            filename = doc.file_name or "unknown"

            await update.message.chat.send_action(ChatAction.TYPING)

            # Download to workspace
            filepath = os.path.join(WORKSPACE, filename)
            file = await doc.get_file()
            await file.download_to_drive(filepath)

            # Check if it's an image sent as document
            ext = os.path.splitext(filename)[1].lower()
            if ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'):
                with open(filepath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                try:
                    prompt = caption or "Describe and analyze this image in detail."
                    response = await asyncio.to_thread(adapter._analyze_image, b64, prompt)
                    messages = adapter.get_or_create_session(chat_id)
                    messages.append({"role": "user", "content": f"[Sent image: {filename}] {prompt}"})
                    messages.append({"role": "assistant", "content": response})
                    await adapter._send_response(update.message, response)
                except Exception as e:
                    await update.message.reply_text(f"Error analyzing image: {e}")
                return

            # Extract text content
            text_content = await asyncio.to_thread(_extract_text, filepath)

            if not text_content:
                await update.message.reply_text(
                    f"Saved <code>{html_mod.escape(filename)}</code> to workspace, "
                    "but couldn't extract text from this format.",
                    parse_mode="HTML",
                )
                return

            # Truncate if very long
            if len(text_content) > 15000:
                text_content = text_content[:15000] + "\n\n[... truncated ...]"

            prompt = (
                f"The user sent a file: **{filename}**\n\n"
                f"File content:\n```\n{text_content}\n```\n\n"
                f"{caption or 'Please analyze this document and provide a summary.'}\n\n"
                "Note: You can offer to index this document with rag_index for future "
                "semantic search if the user might want to ask questions about it later."
            )

            bot = context.bot
            try:
                response = await adapter.handle_message_streaming(chat_id, prompt, bot)
                await adapter._send_response(update.message, response)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        # ── Command handlers ───────────────────────────────────────

        async def _on_start(update: Update, context):
            model = html_mod.escape(adapter.primary.MODEL)
            await update.message.reply_text(
                f"<b>SheLLM</b>  <code>{model}</code>\n\n"
                "I'm an AI assistant with web search, shell access, "
                "persistent memory, and more.\n\n"
                "Send me text, images, or documents — or use /help for commands.",
                parse_mode="HTML",
            )

        async def _on_help(update: Update, context):
            await update.message.reply_text(
                "<b>Available commands</b>\n\n"
                "• /search <code>&lt;query&gt;</code> — Research with web search\n"
                "• /memory — Read shared memory\n"
                "• /remember <code>&lt;text&gt;</code> — Save to memory\n"
                "• /recall <code>&lt;keyword&gt;</code> — Search memory\n"
                "• /usage — API balances and usage stats\n"
                "• /files — List workspace files\n"
                "• /download <code>&lt;filename&gt;</code> — Get a file from workspace\n"
                "• /plan <code>&lt;task&gt;</code> — Plan before executing (confirm/revise/cancel)\n"
                "• /logs — Recent chat history\n"
                "• /model — Current engine info\n"
                "• /clear — Reset conversation\n"
                "• /forget — Clear conversation history\n"
                "• /help — Show this message\n\n"
                "<b>Media support</b>\n"
                "• Send an <b>image</b> — analyzed by shellm-image agent (vision)\n"
                "• Send a <b>document</b> (PDF, DOCX, CSV, code, etc.) — extracted and analyzed",
                parse_mode="HTML",
            )

        async def _on_clear(update: Update, context):
            import sys
            chat_id = update.effective_chat.id
            adapter.sessions.pop(chat_id, None)
            adapter._save_sessions()
            await update.message.reply_text("Conversation cleared. Restarting bot...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        async def _on_model(update: Update, context):
            await update.message.reply_text(adapter.primary.format_banner())

        async def _on_search(update: Update, context):
            query = " ".join(context.args) if context.args else ""
            if not query:
                await update.message.reply_text("Usage: /search <code>&lt;query&gt;</code>", parse_mode="HTML")
                return
            chat_id = update.effective_chat.id
            bot = context.bot
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                response = await adapter.handle_message_streaming(
                    chat_id, f"/search {query}", bot
                )
                await adapter._send_response(update.message, response)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _on_memory(update: Update, context):
            result = await asyncio.to_thread(memory_read)
            if len(result) > 4000:
                result = result[:4000] + "\n\n[... truncated ...]"
            await update.message.reply_text(result)

        async def _on_remember(update: Update, context):
            text = " ".join(context.args) if context.args else ""
            if not text:
                await update.message.reply_text("Usage: /remember <code>&lt;text&gt;</code>", parse_mode="HTML")
                return
            chat_id = update.effective_chat.id
            bot = context.bot
            await update.message.chat.send_action(ChatAction.TYPING)
            response = await adapter.handle_message_streaming(
                chat_id,
                f"Save this to memory using memory_write: {text}",
                bot,
            )
            await adapter._send_response(update.message, response)

        async def _on_recall(update: Update, context):
            keyword = " ".join(context.args) if context.args else ""
            if not keyword:
                await update.message.reply_text("Usage: /recall <keyword>")
                return
            result = await asyncio.to_thread(memory_search, keyword)
            if len(result) > 4000:
                result = result[:4000] + "\n\n[... truncated ...]"
            await update.message.reply_text(result)

        async def _on_logs(update: Update, context):
            result = adapter.primary._read_chat_logs(last_n=5)
            if len(result) > 4000:
                result = result[:4000] + "\n\n[... truncated ...]"
            await update.message.reply_text(result)

        async def _on_files(update: Update, context):
            """List files in the workspace directory."""
            try:
                files = os.listdir(WORKSPACE)
                if not files:
                    await update.message.reply_text("Workspace is empty.")
                    return
                listing = "\n".join(f"• {f}" for f in sorted(files))
                await update.message.reply_text(
                    f"<b>Workspace files</b>\n\n{html_mod.escape(listing)}",
                    parse_mode="HTML",
                )
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _on_download(update: Update, context):
            """Send a file from workspace to the user."""
            filename = " ".join(context.args) if context.args else ""
            if not filename:
                await update.message.reply_text(
                    "Usage: /download <code>&lt;filename&gt;</code>", parse_mode="HTML"
                )
                return
            filepath = os.path.join(WORKSPACE, filename)
            if not os.path.isfile(filepath):
                await update.message.reply_text(f"File not found: {filename}")
                return
            try:
                await update.message.reply_document(
                    document=open(filepath, "rb"),
                    filename=filename,
                )
            except Exception as e:
                await update.message.reply_text(f"Error sending file: {e}")

        async def _on_usage(update: Update, context):
            """Show API balances and local usage stats."""
            from telegram import LinkPreviewOptions
            await update.message.chat.send_action(ChatAction.TYPING)
            result = await asyncio.to_thread(_fetch_usage, adapter.registry)
            await update.message.reply_text(
                result, parse_mode="HTML",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )

        async def _on_forget(update: Update, context):
            chat_id = update.effective_chat.id
            adapter.sessions.pop(chat_id, None)
            adapter._save_sessions()
            await update.message.reply_text(
                "Conversation history cleared.\n"
                "Note: shared memory is preserved (use the LLM to delete specific entries)."
            )

        async def _on_plan(update: Update, context):
            """Present an execution plan before running a task."""
            prompt = " ".join(context.args) if context.args else ""
            if not prompt:
                await update.message.reply_text(
                    "Usage: /plan <code>&lt;task description&gt;</code>", parse_mode="HTML"
                )
                return
            chat_id = update.effective_chat.id
            await update.message.chat.send_action(ChatAction.TYPING)

            # Route /plan to the reasoner agent for deep planning
            try:
                reasoner = adapter.registry.get_agent("shellm-reasoner")
            except Exception:
                reasoner = adapter.primary

            plan_prompt = (
                f"The user wants you to plan (but NOT execute) the following task:\n\n"
                f"{prompt}\n\n"
                "Create a detailed execution plan that includes:\n"
                "1. **Steps** — numbered list of what you'll do\n"
                "2. **Tools** — which tools/delegations you'll use\n"
                "3. **Estimated complexity** — simple / moderate / complex\n\n"
                "Do NOT execute anything. Only present the plan.\n"
                "End with: 'Reply **yes** to execute, **no** to cancel, or provide feedback to adjust the plan.'"
            )

            try:
                loop = asyncio.get_event_loop()
                plan_messages = []
                reasoner._ensure_system_message(plan_messages)
                response = await loop.run_in_executor(
                    None, lambda: reasoner.process_prompt(plan_prompt, plan_messages)
                )
                adapter._pending_plans[chat_id] = {"prompt": prompt, "plan": response or ""}
                await adapter._send_response(update.message, response or "(No plan generated)")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _post_init(app):
            await app.bot.set_my_commands([
                BotCommand("search", "Research a topic with web search"),
                BotCommand("memory", "Read shared memory"),
                BotCommand("remember", "Save something to memory"),
                BotCommand("recall", "Search memory by keyword"),
                BotCommand("usage", "API balances and usage stats"),
                BotCommand("files", "List workspace files"),
                BotCommand("download", "Get a file from workspace"),
                BotCommand("plan", "Plan before executing a task"),
                BotCommand("logs", "Show recent chat history"),
                BotCommand("model", "Show current engine info"),
                BotCommand("clear", "Reset conversation history"),
                BotCommand("forget", "Clear conversation history"),
                BotCommand("help", "Show all commands"),
            ])

        app = ApplicationBuilder().token(self.bot_token).post_init(_post_init).build()
        app.add_handler(CommandHandler("start", _on_start))
        app.add_handler(CommandHandler("help", _on_help))
        app.add_handler(CommandHandler("clear", _on_clear))
        app.add_handler(CommandHandler("model", _on_model))
        app.add_handler(CommandHandler("search", _on_search))
        app.add_handler(CommandHandler("memory", _on_memory))
        app.add_handler(CommandHandler("remember", _on_remember))
        app.add_handler(CommandHandler("recall", _on_recall))
        app.add_handler(CommandHandler("plan", _on_plan))
        app.add_handler(CommandHandler("usage", _on_usage))
        app.add_handler(CommandHandler("files", _on_files))
        app.add_handler(CommandHandler("download", _on_download))
        app.add_handler(CommandHandler("logs", _on_logs))
        app.add_handler(CommandHandler("forget", _on_forget))
        app.add_handler(MessageHandler(filters.PHOTO, _on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, _on_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

        # Start heartbeat scheduler with bot token for proactive messages
        from task_scheduler import TaskScheduler
        TaskScheduler.get_instance().start(bot_token=self.bot_token)

        print(f"Telegram bot started with streaming (model: {self.primary.MODEL})", flush=True)
        app.run_polling(drop_pending_updates=True)
