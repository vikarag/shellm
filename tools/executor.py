"""Tool execution dispatcher for SheLLM."""

import json
import re
import urllib.request

from cron_manager import cron_list, cron_create, cron_delete
from command_runner import run_command
from memory_manager import memory_read, memory_write, memory_search, memory_delete
from file_tools import read_file, write_file, list_directory, search_files
from rag_engine import rag_index, rag_search, rag_list, rag_delete
from agents.progress import progress_queue


class ToolContext:
    """Carries shared state needed by tool handlers."""

    def __init__(self, model=None, print_fn=None):
        self.model = model
        self._print = print_fn or (lambda *a, **kw: None)


# ── Web ──────────────────────────────────────────────────────────────

def _exec_web_search(args, ctx):
    """DuckDuckGo search via the ddgs package."""
    query = args.get("query", "")
    max_results = min(max(int(args.get("max_results", 5)), 1), 20)
    ctx._print(f"[web_search: {query}]")
    try:
        from ddgs import DDGS
    except ImportError:
        return "web_search unavailable: `pip install ddgs` first."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No results."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        href = r.get("href") or r.get("url", "")
        body = r.get("body", "")
        lines.append(f"[{i}] {title}\n{href}\n{body}")
    return "\n\n".join(lines)


def _exec_fetch_page(args, ctx):
    """Fetch a URL and return its readable text. urllib + BeautifulSoup if
    available, falling back to a regex-based tag stripper."""
    url = args.get("url", "")
    ctx._print(f"[fetch_page: {url}]")

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            ctype = resp.headers.get_content_type() or ""
            charset = resp.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")
    except Exception as e:
        return f"Error fetching {url}: {e}"

    if "html" not in ctype and "xml" not in ctype:
        text = html
        title = ""
    else:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = (soup.title.string.strip() if soup.title and soup.title.string else "")
            text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        except ImportError:
            stripped = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            stripped = re.sub(r"<style[^>]*>.*?</style>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
            stripped = re.sub(r"<[^>]+>", " ", stripped)
            text = re.sub(r"\s+", " ", stripped).strip()
            m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = m.group(1).strip() if m else ""

    if len(text) > 50_000:
        text = text[:50_000] + "\n\n[...truncated]"

    header = f"Title: {title}\nURL: {url}\n\n" if title else f"URL: {url}\n\n"
    return header + text


# ── Files ────────────────────────────────────────────────────────────

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


# ── RAG ──────────────────────────────────────────────────────────────

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


# ── Cron ─────────────────────────────────────────────────────────────

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


# ── Shell ────────────────────────────────────────────────────────────

def _exec_run_command(args, ctx):
    result = run_command(args.get("command", ""), timeout=args.get("timeout", 60))
    ctx._print(result)
    return result


# ── Memory ───────────────────────────────────────────────────────────

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


# ── Chat logs ────────────────────────────────────────────────────────

def _exec_chat_log_read(args, ctx):
    from agents.base_agent import BaseAgent
    return BaseAgent._read_chat_logs_static(
        last_n=args.get("last_n", 10),
        keyword=args.get("keyword"),
        model_filter=args.get("model_filter"),
    )


# ── Dispatch ─────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "web_search": _exec_web_search,
    "fetch_page": _exec_fetch_page,
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
}


def execute_tool(name, args, ctx):
    """Execute a tool by name with the given args and context."""
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    progress_queue.push("tool_call", name, json.dumps(args, ensure_ascii=False)[:120])
    return handler(args, ctx)
