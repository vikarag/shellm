#!/home/gslee/llm-api-vault/venv/bin/python3
"""Telegram bot adapter for LLM chat clients with streaming via sendMessageDraft."""

import asyncio
import os
import time


class TelegramAdapter:
    """Connects a BaseChatClient to a Telegram bot interface.

    Uses Telegram Bot API 9.5 sendMessageDraft for real-time streaming
    of LLM responses in private chats.

    Supported Telegram commands:
        /start   -- Welcome message and help
        /clear   -- Reset conversation history
        /model   -- Show current engine info
        /search  -- Force web search research
        /memory  -- Read shared memory
        /forget  -- Clear conversation + memory for this chat
        /logs    -- Show recent chat history
        /help    -- Show all available commands

    Usage:
        ./deepseek_chat.py --telegram
    """

    def __init__(self, chat_client, bot_token=None):
        self.client = chat_client
        self.client._silent = True
        self.client._mode = "telegram"
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.sessions = {}  # chat_id -> messages list

    def get_or_create_session(self, chat_id):
        if chat_id not in self.sessions:
            self.sessions[chat_id] = []
        return self.sessions[chat_id]

    def _process_streaming(self, text, messages, token_queue):
        """Run blocking process_prompt with a token callback that feeds an asyncio queue."""
        def on_token(accumulated):
            token_queue.put_nowait(accumulated)

        self.client._on_token = on_token
        try:
            answer = self.client.process_prompt(text, messages)
        finally:
            self.client._on_token = None
        token_queue.put_nowait(None)
        return answer

    async def handle_message_streaming(self, chat_id, text, bot):
        """Handle a message with streaming draft updates."""
        messages = self.get_or_create_session(chat_id)

        token_queue = asyncio.Queue()
        draft_id = int(time.time() * 1000) % (2**31 - 1)

        loop = asyncio.get_event_loop()
        process_task = loop.run_in_executor(
            None, self._process_streaming, text, messages, token_queue
        )

        last_draft_time = 0
        last_draft_text = ""

        while True:
            try:
                accumulated = await asyncio.wait_for(token_queue.get(), timeout=120)
            except asyncio.TimeoutError:
                break

            if accumulated is None:
                break

            now = time.time()
            if now - last_draft_time >= 0.5 and accumulated != last_draft_text:
                try:
                    await bot.send_message_draft(
                        chat_id=chat_id,
                        draft_id=draft_id,
                        text=accumulated[:4096],
                    )
                    last_draft_text = accumulated
                    last_draft_time = now
                except Exception:
                    pass

        final_answer = await process_task
        return final_answer or "(No response)"

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

        async def _on_message(update: Update, context):
            chat_id = update.effective_chat.id
            text = update.message.text.strip()
            if not text:
                return
            bot = context.bot
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                response = await adapter.handle_message_streaming(chat_id, text, bot)
                if len(response) > 4000:
                    for i in range(0, len(response), 4000):
                        await update.message.reply_text(response[i:i + 4000])
                else:
                    await update.message.reply_text(response)
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        async def _on_start(update: Update, context):
            await update.message.reply_text(
                f"shellm ({adapter.client.MODEL})\n\n"
                "I'm an AI assistant with web search, shell access, "
                "persistent memory, and more.\n\n"
                "Just send a message to chat, or use /help for commands."
            )

        async def _on_help(update: Update, context):
            await update.message.reply_text(
                "Available commands:\n\n"
                "/search <query> -- Research a topic with web search\n"
                "/memory -- Read shared memory\n"
                "/remember <text> -- Save something to memory\n"
                "/recall <keyword> -- Search memory by keyword\n"
                "/logs -- Show recent chat history\n"
                "/model -- Show current engine info\n"
                "/clear -- Reset conversation history\n"
                "/forget -- Clear conversation + all memory\n"
                "/help -- Show this message"
            )

        async def _on_clear(update: Update, context):
            chat_id = update.effective_chat.id
            adapter.sessions.pop(chat_id, None)
            await update.message.reply_text("Conversation cleared.")

        async def _on_model(update: Update, context):
            await update.message.reply_text(adapter.client.format_banner())

        async def _on_search(update: Update, context):
            query = " ".join(context.args) if context.args else ""
            if not query:
                await update.message.reply_text("Usage: /search <query>")
                return
            chat_id = update.effective_chat.id
            bot = context.bot
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                response = await adapter.handle_message_streaming(
                    chat_id, f"/search {query}", bot
                )
                if len(response) > 4000:
                    for i in range(0, len(response), 4000):
                        await update.message.reply_text(response[i:i + 4000])
                else:
                    await update.message.reply_text(response)
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
                await update.message.reply_text("Usage: /remember <something to save>")
                return
            chat_id = update.effective_chat.id
            bot = context.bot
            await update.message.chat.send_action(ChatAction.TYPING)
            response = await adapter.handle_message_streaming(
                chat_id,
                f"Save this to memory using memory_write: {text}",
                bot,
            )
            await update.message.reply_text(response)

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

        async def _on_forget(update: Update, context):
            chat_id = update.effective_chat.id
            adapter.sessions.pop(chat_id, None)
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
        app.add_handler(CommandHandler("logs", _on_logs))
        app.add_handler(CommandHandler("forget", _on_forget))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

        print(f"Telegram bot started with streaming (model: {self.client.MODEL})", flush=True)
        app.run_polling(drop_pending_updates=True)
