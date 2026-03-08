#!/home/gslee/shellm/venv/bin/python3
"""Telegram bot adapter for LLM chat clients with streaming, vision, and file handling."""

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


def _fetch_usage():
    """Fetch API balance from providers and local usage stats."""
    from datetime import datetime

    lines = ["<b>API Balances</b>\n"]

    # DeepSeek balance
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {ds_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("is_available"):
                for info in data.get("balance_infos", []):
                    currency = info.get("currency", "USD")
                    total = info.get("total_balance", "?")
                    granted = info.get("granted_balance", "0")
                    lines.append(f"• <b>DeepSeek</b> (Chat): ${total} remaining")
                    if float(granted) > 0:
                        lines.append(f"  (granted: ${granted})")
            else:
                lines.append("• <b>DeepSeek</b>: Account unavailable")
        except Exception as e:
            lines.append(f"• <b>DeepSeek</b>: Error — {e}")

    # Moonshot/Kimi balance
    ms_key = os.environ.get("MOONSHOT_API_KEY", "")
    if ms_key:
        try:
            req = urllib.request.Request(
                "https://api.moonshot.ai/v1/users/me/balance",
                headers={"Authorization": f"Bearer {ms_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            bal = data.get("data", {})
            available = bal.get("available_balance", "?")
            cash = bal.get("cash_balance", 0)
            voucher = bal.get("voucher_balance", 0)
            lines.append(f"• <b>Kimi K2.5</b> (Code): ¥{available}")
            if cash or voucher:
                lines.append(f"  (cash: ¥{cash}, voucher: ¥{voucher})")
        except Exception as e:
            lines.append(f"• <b>Kimi K2.5</b>: Error — {e}")

    # OpenAI — no balance API for standard keys
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        lines.append(
            '• <b>GPT-5 Mini</b> (Research): '
            '<a href="https://platform.openai.com/usage">Check dashboard</a>'
        )

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
    """Connects a BaseChatClient to a Telegram bot interface.

    Features:
        - Real-time streaming via sendMessageDraft (Bot API 9.5)
        - HTML-formatted responses (Markdown → Telegram HTML)
        - Document handling (PDF, DOCX, CSV, TXT, code files)
        - Image analysis via GPT-5 Mini vision
        - Workspace file management

    Usage:
        ./deepseek_chat.py --telegram
    """

    MAX_SESSION_MESSAGES = 50

    def __init__(self, chat_client, bot_token=None):
        self.client = chat_client
        self.client._silent = True
        self.client._mode = "telegram"
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.sessions = {}  # chat_id -> messages list
        self._load_sessions()

        # Vision engine (GPT-5 Mini) for image analysis
        from gpt5mini_chat import GPT5MiniChat
        self._vision_engine = GPT5MiniChat()
        self._vision_engine._silent = True
        self._vision_engine._mode = "telegram"

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

        self.client._current_chat_id = chat_id

        loop = asyncio.get_event_loop()
        final_answer = await loop.run_in_executor(
            None, self.client.process_prompt, text, messages
        )

        # Persist session after processing
        self.sessions[chat_id] = self._trim_session(messages)
        self._save_sessions()

        return final_answer or "(No response)"

    def _analyze_image(self, b64_data, prompt):
        """Send image to GPT-5 Mini for vision analysis."""
        response = self._vision_engine.client.chat.completions.create(
            model=self._vision_engine.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
                ],
            }],
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content or "(No response)"

    def run(self):
        """Start the Telegram bot with streaming support."""
        if not self.bot_token:
            raise RuntimeError(
                "Set TELEGRAM_BOT_TOKEN environment variable.\n"
                "Then run: ./deepseek_chat.py --telegram"
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
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                response = await adapter.handle_message_streaming(chat_id, text, bot)
                await adapter._send_response(update.message, response)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _on_photo(update: Update, context):
            """Handle photos — route to GPT-5 Mini for vision analysis."""
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
            model = html_mod.escape(adapter.client.MODEL)
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
                "• /logs — Recent chat history\n"
                "• /model — Current engine info\n"
                "• /clear — Reset conversation\n"
                "• /forget — Clear conversation history\n"
                "• /help — Show this message\n\n"
                "<b>Media support</b>\n"
                "• Send an <b>image</b> — analyzed by GPT-5 Mini (vision)\n"
                "• Send a <b>document</b> (PDF, DOCX, CSV, code, etc.) — extracted and analyzed",
                parse_mode="HTML",
            )

        async def _on_clear(update: Update, context):
            chat_id = update.effective_chat.id
            adapter.sessions.pop(chat_id, None)
            adapter._save_sessions()
            await update.message.reply_text("Conversation cleared.")

        async def _on_model(update: Update, context):
            await update.message.reply_text(adapter.client.format_banner())

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
            result = adapter.client._read_chat_logs(last_n=5)
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
            result = await asyncio.to_thread(_fetch_usage)
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

        async def _post_init(app):
            await app.bot.set_my_commands([
                BotCommand("search", "Research a topic with web search"),
                BotCommand("memory", "Read shared memory"),
                BotCommand("remember", "Save something to memory"),
                BotCommand("recall", "Search memory by keyword"),
                BotCommand("usage", "API balances and usage stats"),
                BotCommand("files", "List workspace files"),
                BotCommand("download", "Get a file from workspace"),
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

        print(f"Telegram bot started with streaming (model: {self.client.MODEL})", flush=True)
        app.run_polling(drop_pending_updates=True)
