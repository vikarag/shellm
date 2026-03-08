"""Convert standard Markdown (LLM output) to Telegram-compatible HTML."""

import html
import re


def md_to_tg_html(text: str) -> str:
    """Convert standard Markdown to Telegram HTML.

    Handles: fenced code blocks, inline code, bold, italic, strikethrough,
    links, blockquotes, headers, horizontal rules, and bullet lists.
    """
    if not text:
        return text

    placeholders = []

    def _save(content):
        idx = len(placeholders)
        placeholders.append(content)
        return f"\x00PH{idx}\x00"

    # Fenced code blocks: ```lang\n...\n```
    def _fenced(m):
        lang = m.group(1) or ""
        code = html.escape(m.group(2).strip())
        if lang:
            return _save(
                f'<pre><code class="language-{html.escape(lang)}">'
                f"{code}</code></pre>"
            )
        return _save(f"<pre>{code}</pre>")

    text = re.sub(r"```(\w*)\n(.*?)```", _fenced, text, flags=re.DOTALL)

    # Inline code: `...`
    def _inline(m):
        return _save(f"<code>{html.escape(m.group(1))}</code>")

    text = re.sub(r"`([^`\n]+)`", _inline, text)

    # Escape HTML entities in remaining text
    text = html.escape(text)

    # Bold: **text** or __text__ (before italic)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # Italic: *text* (not inside words)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Links: [text](url) — url was escaped, unescape href
    def _link(m):
        label = m.group(1)
        url = m.group(2).replace("&amp;", "&")
        return f'<a href="{url}">{label}</a>'

    text = re.sub(r"\[(.+?)\]\((.+?)\)", _link, text)

    # Headers: # text → bold (Telegram has no heading support)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Blockquotes: > text (escaped as &gt; by html.escape)
    def _bq_block(m):
        lines = m.group(0).split("\n")
        inner = "\n".join(re.sub(r"^&gt;\s?", "", line) for line in lines)
        return f"<blockquote>{inner}</blockquote>"

    text = re.sub(
        r"(^&gt;[^\n]*(?:\n&gt;[^\n]*)*)", _bq_block, text, flags=re.MULTILINE
    )

    # Horizontal rules: ---, ***, ___
    text = re.sub(r"^[-*_]{3,}$", "———", text, flags=re.MULTILINE)

    # Bullet lists: - item or * item → • item
    text = re.sub(r"^(\s*)[-*]\s+", r"\1• ", text, flags=re.MULTILINE)

    # Restore placeholders
    for idx, content in enumerate(placeholders):
        text = text.replace(f"\x00PH{idx}\x00", content)

    return text.strip()


def split_message(text: str, max_len: int = 4000) -> list:
    """Split text into chunks that fit Telegram's 4096-char limit.

    Prefers splitting at paragraph boundaries, then lines, then spaces.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Prefer paragraph boundary, then line, then space, then hard cut
        cut = text.rfind("\n\n", 0, max_len)
        if cut == -1:
            cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len

        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")

    return [c for c in chunks if c.strip()]
