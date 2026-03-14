"""Tool execution dispatcher for SheLLM agents."""

import json
import os
import threading

from cron_manager import cron_list, cron_create, cron_delete
from command_runner import run_command
from memory_manager import memory_read, memory_write, memory_search, memory_delete
from file_tools import read_file, write_file, list_directory, search_files
from rag_engine import rag_index, rag_search, rag_list, rag_delete
from task_scheduler import schedule_task, list_scheduled_tasks, cancel_scheduled_task
from agents.progress import progress_queue


class ToolContext:
    """Carries shared state needed by tool handlers."""

    def __init__(self, registry=None, current_chat_id=None, mode="interactive",
                 model=None, plan_text=None, print_fn=None):
        self.registry = registry
        self.current_chat_id = current_chat_id
        self.mode = mode
        self.model = model
        self.plan_text = plan_text
        self._print = print_fn or (lambda *a, **kw: None)


def _exec_delegate_websearch(args, ctx):
    """Delegate web search to shellm-websearch agent."""
    query = args.get("query", "")
    ctx._print(f"[Web Search: {query}...]")
    try:
        agent = ctx.registry.get_agent("shellm-websearch")
        result = agent.search(query)
    except Exception as e:
        result = f"Web search error: {e}"
    ctx._print("[Search complete]")
    progress_queue.push("tool_call", "delegate_websearch", f"Query: {query[:80]}")
    return result


def _fallback_fetch(url):
    """Basic URL fetch via urllib when camoufox is unavailable."""
    import re as _re
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags
        text = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()
        if len(text) > 50_000:
            text = text[:50_000] + "\n\n[...truncated]"
        # Extract title
        title_match = _re.search(r"<title[^>]*>(.*?)</title>", html, _re.IGNORECASE | _re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        return {"text": text, "title": title, "url": url, "error": None}
    except Exception as e:
        return {"text": "", "title": "", "url": url, "error": str(e)}


def _exec_fetch_page(args, ctx):
    """Fetch a web page and extract its text content."""
    url = args.get("url", "")
    wait_for = args.get("wait_for")
    ctx._print(f"[Fetching: {url}...]")

    from browser_engine import BrowserEngine

    if BrowserEngine.is_available():
        try:
            engine = BrowserEngine.get_instance()
            result = engine.fetch_page(url, wait_for_selector=wait_for)
        except Exception as e:
            ctx._print(f"[Browser failed, using fallback: {e}]")
            result = _fallback_fetch(url)
    else:
        result = _fallback_fetch(url)

    if result.get("error"):
        ctx._print(f"[Fetch error: {result['error']}]")
        return f"Error fetching {url}: {result['error']}"

    ctx._print("[Fetch complete]")
    title = result.get("title", "")
    text = result.get("text", "")
    progress_queue.push("tool_call", "fetch_page", f"URL: {url[:80]}")
    header = f"Title: {title}\nURL: {result.get('url', url)}\n\n" if title else ""
    return header + text


def _exec_delegate_image(args, ctx):
    """Delegate image analysis to shellm-image agent."""
    b64 = args.get("image_b64", "")
    prompt = args.get("prompt", "Describe this image.")
    ctx._print("[Analyzing image...]")
    try:
        agent = ctx.registry.get_agent("shellm-image")
        result = agent.analyze_image(b64, prompt)
    except Exception as e:
        result = f"Image analysis error: {e}"
    ctx._print("[Analysis complete]")
    progress_queue.push("tool_call", "delegate_image", f"Prompt: {prompt[:80]}")
    return result


def _exec_delegate_research(args, ctx):
    """Delegate research to shellm-researcher agent."""
    query = args.get("query", "")
    ctx._print(f"[Researching: {query}...]")
    try:
        agent = ctx.registry.get_agent("shellm-researcher")
        result = agent.research(query)
    except Exception as e:
        result = f"Research error: {e}"
    ctx._print("[Research complete]")
    progress_queue.push("tool_call", "delegate_research", f"Query: {query[:80]}")
    return result


def _exec_delegate_reason(args, ctx):
    """Delegate reasoning to shellm-reasoner agent."""
    task = args.get("task", "")
    ctx._print(f"[Reasoning: {task[:80]}...]")
    try:
        agent = ctx.registry.get_agent("shellm-reasoner")
        messages = []
        result = agent.process_prompt(task, messages)
    except Exception as e:
        result = f"Reasoning error: {e}"
    ctx._print("[Reasoning complete]")
    progress_queue.push("tool_call", "delegate_reason", f"Task: {task[:80]}")
    return result


def _exec_read_file(args, ctx):
    return read_file(args.get("path", ""), offset=args.get("offset", 0), limit=args.get("limit", 200))


def _exec_write_file(args, ctx):
    result = write_file(args.get("path", ""), args.get("content", ""), mode=args.get("mode", "overwrite"))
    ctx._print(result)
    return result


def _exec_list_directory(args, ctx):
    return list_directory(args.get("path", "."), recursive=args.get("recursive", False))


def _exec_search_files(args, ctx):
    return search_files(args.get("pattern", ""), path=args.get("path", "."), file_glob=args.get("file_glob", "*"))


def _exec_rag_index(args, ctx):
    result = rag_index(args.get("text", ""), filename=args.get("filename", "untitled"), tags=args.get("tags"))
    ctx._print(result)
    return result


def _exec_rag_search(args, ctx):
    return rag_search(args.get("query", ""), top_k=args.get("top_k", 5))


def _exec_rag_list(args, ctx):
    return rag_list()


def _exec_rag_delete(args, ctx):
    result = rag_delete(args.get("doc_id", ""))
    ctx._print(result)
    return result


def _exec_cron_list(args, ctx):
    result = cron_list()
    ctx._print(result)
    return result


def _exec_cron_create(args, ctx):
    result = cron_create(args.get("schedule", ""), args.get("command", ""))
    ctx._print(result)
    return result


def _exec_cron_delete(args, ctx):
    result = cron_delete(args.get("index", 0))
    ctx._print(result)
    return result


def _exec_run_command(args, ctx):
    result = run_command(
        args.get("command", ""),
        timeout=args.get("timeout", 60),
        auto_approve=(ctx.mode != "interactive"),
    )
    ctx._print(result)
    return result


def _exec_memory_read(args, ctx):
    return memory_read(last_n=args.get("last_n", 0))


def _exec_memory_write(args, ctx):
    result = memory_write(args.get("content", ""), source=ctx.model or "unknown", tags=args.get("tags", []))
    ctx._print(result)
    return result


def _exec_memory_search(args, ctx):
    return memory_search(args.get("keyword", ""))


def _exec_memory_delete(args, ctx):
    result = memory_delete(args.get("index", 0))
    ctx._print(result)
    return result


def _exec_chat_log_read(args, ctx):
    # Import here to avoid circular dep — uses the agent's own _read_chat_logs method
    from agents.base_agent import BaseAgent
    return BaseAgent._read_chat_logs_static(
        last_n=args.get("last_n", 10),
        keyword=args.get("keyword"),
        model_filter=args.get("model_filter"),
    )


def _get_fallback_chat_id():
    """Read persisted Telegram chat_id written by the telegram adapter."""
    chat_id_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".telegram_chat_id")
    try:
        with open(chat_id_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _exec_send_file(args, ctx):
    """Send a file to Telegram chat."""
    chat_id = ctx.current_chat_id or _get_fallback_chat_id()
    if not chat_id:
        return "Cannot send file: no Telegram chat_id available."
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return "Cannot send file: TELEGRAM_BOT_TOKEN not set."

    path = args.get("path", "")
    caption = args.get("caption")
    workspace = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

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

        if is_image:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            field_name = "photo"
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
            field_name = "document"

        boundary = "----SheLLMFileBoundary"
        body_parts = []
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}")
        if caption:
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}")

        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            file_data = f.read()

        file_header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        )

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


def _exec_mcp_list_servers(args, ctx):
    from mcp_manager import MCPManager
    return MCPManager.get_instance().list_servers()


def _exec_mcp_list_tools(args, ctx):
    from mcp_manager import MCPManager
    return MCPManager.get_instance().list_server_tools(args.get("server"))


def _exec_schedule_task(args, ctx):
    task_type = args.get("task_type", "")
    chat_id = ctx.current_chat_id
    if task_type == "telegram_message" and not chat_id:
        chat_id = _get_fallback_chat_id()
        if not chat_id:
            return "Cannot schedule Telegram message: no Telegram chat_id available."
    result = schedule_task(
        task_type=task_type,
        payload=args.get("payload", {}),
        delay_minutes=args.get("delay_minutes"),
        scheduled_at=args.get("scheduled_at"),
        chat_id=chat_id,
    )
    ctx._print(result)
    return result


def _exec_list_scheduled_tasks(args, ctx):
    return list_scheduled_tasks(status=args.get("status", "pending"))


def _exec_cancel_scheduled_task(args, ctx):
    result = cancel_scheduled_task(args.get("task_id", 0))
    ctx._print(result)
    return result


def _exec_report_progress(args, ctx):
    """Report plan progress and optionally notify via Telegram."""
    step_number = args.get("step_number", "?")
    step_title = args.get("step_title", "")
    details = args.get("details", "")

    # Push to progress queue
    progress_queue.push(
        "progress",
        f"Step {step_number}/{args.get('total_steps', '?')}: {step_title}",
        details,
    )

    # Background Telegram notification
    chat_id = ctx.current_chat_id
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if chat_id and bot_token:
        plan_text = (ctx.plan_text or "")[:2000]
        total_steps = args.get("total_steps", "?")

        def _bg_send():
            try:
                # Use updater agent's client for the summary
                if ctx.registry:
                    updater_client = ctx.registry.get_client("shellm-updater")
                    summary_prompt = (
                        f"Summarize this plan step completion in 1-2 concise sentences for a Telegram notification.\n\n"
                        f"Step {step_number}/{total_steps}: {step_title}\n"
                        f"Details: {details}\n\n"
                        f"Plan context:\n{plan_text}"
                    )
                    response = updater_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_completion_tokens=200,
                        stream=False,
                    )
                    summary = response.choices[0].message.content or details
                else:
                    summary = details

                text = f"[Plan Progress {step_number}/{total_steps}]\n\n{summary}"
                import urllib.request as _ur
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = json.dumps({"chat_id": chat_id, "text": text}).encode()
                req = _ur.Request(url, data=data, headers={"Content-Type": "application/json"})
                _ur.urlopen(req, timeout=15)
            except Exception:
                pass

        t = threading.Thread(target=_bg_send, daemon=True)
        t.start()

    return f"Progress reported: Step {step_number} - {step_title}"


# Dispatch table
TOOL_DISPATCH = {
    "delegate_websearch": _exec_delegate_websearch,
    "fetch_page": _exec_fetch_page,
    "delegate_image": _exec_delegate_image,
    "delegate_research": _exec_delegate_research,
    "delegate_reason": _exec_delegate_reason,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "list_directory": _exec_list_directory,
    "search_files": _exec_search_files,
    "rag_index": _exec_rag_index,
    "rag_search": _exec_rag_search,
    "rag_list": _exec_rag_list,
    "rag_delete": _exec_rag_delete,
    "cron_list": _exec_cron_list,
    "cron_create": _exec_cron_create,
    "cron_delete": _exec_cron_delete,
    "run_command": _exec_run_command,
    "memory_read": _exec_memory_read,
    "memory_write": _exec_memory_write,
    "memory_search": _exec_memory_search,
    "memory_delete": _exec_memory_delete,
    "chat_log_read": _exec_chat_log_read,
    "send_file": _exec_send_file,
    "mcp_list_servers": _exec_mcp_list_servers,
    "mcp_list_tools": _exec_mcp_list_tools,
    "schedule_task": _exec_schedule_task,
    "list_scheduled_tasks": _exec_list_scheduled_tasks,
    "cancel_scheduled_task": _exec_cancel_scheduled_task,
    "report_progress": _exec_report_progress,
}


def execute_tool(name, args, ctx):
    """Execute a tool by name with the given args and context.

    Args:
        name: Tool function name
        args: Dict of arguments
        ctx: ToolContext with shared state

    Returns:
        String result
    """
    # Check dispatch table first
    handler = TOOL_DISPATCH.get(name)
    if handler:
        progress_queue.push("tool_call", name, json.dumps(args, ensure_ascii=False)[:120])
        return handler(args, ctx)

    # MCP tool calls (server__tool format)
    if "__" in name:
        progress_queue.push("tool_call", name, json.dumps(args, ensure_ascii=False)[:120])
        # Route to the appropriate MCP agent
        server_name = name.split("__", 1)[0]
        if ctx.registry:
            owning_agent = ctx.registry.get_mcp_agent_for_server(server_name)
            if owning_agent:
                try:
                    agent = ctx.registry.get_agent(owning_agent)
                    return agent.call_mcp_tool(name, args)
                except Exception as e:
                    return f"MCP delegation error ({name}): {e}"
        # Fallback to global MCPManager
        try:
            from mcp_manager import MCPManager
            return MCPManager.get_instance().call_tool(name, args)
        except Exception as e:
            return f"MCP tool error ({name}): {e}"

    return f"Unknown tool: {name}"
