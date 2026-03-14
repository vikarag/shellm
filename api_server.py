#!/home/gslee/SheLLM/venv/bin/python3
"""OpenAI-compatible API server for SheLLM multi-agent system.

Wraps SheLLM agents behind a standard /v1/chat/completions endpoint
so Open WebUI (or any OpenAI-compatible client) can use the full
multi-agent tool pipeline — memory, web search, delegation, etc.

Usage:
    ./api_server.py                    # default port 8091
    ./api_server.py --port 8091
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import uuid

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from agents.registry import AgentRegistry

app = FastAPI(title="SheLLM API", version="1.0.0")

# Global registry — initialized on startup
_registry = None

# Per-session message history (keyed by a session token or chat_id)
_sessions = {}
_sessions_lock = threading.Lock()

MAX_SESSION_MESSAGES = 50


def _get_registry():
    global _registry
    if _registry is None:
        AgentRegistry.reset()
        _registry = AgentRegistry.get_instance()
    return _registry


def _trim_messages(messages):
    """Cap session messages, preserving system message."""
    if len(messages) <= MAX_SESSION_MESSAGES:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0]] + messages[-(MAX_SESSION_MESSAGES - 1):]
    return messages[-MAX_SESSION_MESSAGES:]


def _resolve_agent(model_name):
    """All requests route to shellm-chat (the primary agent).

    shellm-chat internally delegates to other agents (websearch, image,
    reasoner, etc.) via its tool pipeline — just like the Telegram bot.
    """
    registry = _get_registry()
    return registry.get_agent("shellm-chat")


# ── Models endpoint ──────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    registry = _get_registry()
    config = registry.get_all_configs()["shellm-chat"]
    models = [{
        "id": "shellm-chat",
        "object": "model",
        "created": 0,
        "owned_by": "shellm",
        "permission": [],
        "root": config.model,
        "parent": None,
    }]
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    registry = _get_registry()
    config = registry.get_all_configs().get("shellm-chat")
    if model_id == "shellm-chat" and config:
        return {
            "id": "shellm-chat",
            "object": "model",
            "created": 0,
            "owned_by": "shellm",
            "root": config.model,
        }
    return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found"})


# ── Chat completions endpoint ────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model_name = body.get("model", "shellm-chat")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract the last user message and any attached images
    user_input = None
    image_analyses = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        # Extract base64 image data and analyze via shellm-image
                        url_data = part.get("image_url", {}).get("url", "")
                        if url_data.startswith("data:"):
                            # data:image/jpeg;base64,<data>
                            b64 = url_data.split(",", 1)[1] if "," in url_data else ""
                            if b64:
                                try:
                                    registry = _get_registry()
                                    image_agent = registry.get_agent("shellm-image")
                                    prompt = " ".join(text_parts) if text_parts else "Describe and analyze this image in detail."
                                    analysis = image_agent.analyze_image(b64, prompt)
                                    image_analyses.append(analysis)
                                except Exception as e:
                                    image_analyses.append(f"(Image analysis error: {e})")
                user_input = " ".join(text_parts) if text_parts else None
            else:
                user_input = content
            break

    # If we have image analyses, prepend them to the user input
    if image_analyses:
        analysis_text = "\n\n".join(image_analyses)
        if user_input:
            user_input = f"[Image Analysis Result]\n{analysis_text}\n\n[User Message]\n{user_input}"
        else:
            user_input = f"The user sent an image. Here is the analysis:\n\n{analysis_text}\n\nPlease respond based on this image analysis."

    # Intercept Open WebUI internal prompts (RAG wrappers, follow-ups, etc.)
    if user_input:
        user_input, canned = _unwrap_owui_input(user_input)
        if canned is not None:
            # Return canned response for internal OWUI tasks
            if stream:
                return StreamingResponse(_stream_canned(canned, model_name), media_type="text/event-stream")
            return JSONResponse(content=_make_response(canned, model_name))

    if not user_input:
        if stream:
            return StreamingResponse(_stream_error("No user message found"), media_type="text/event-stream")
        return JSONResponse(content=_make_response("No user message found.", model_name))

    agent = _resolve_agent(model_name)
    agent._silent = True
    agent._mode = "api"

    if stream:
        return StreamingResponse(
            _stream_response(agent, user_input, messages, model_name),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(content=_sync_response(agent, user_input, messages, model_name))


def _sync_response(agent, user_input, messages, model_name):
    """Run agent synchronously, return full response."""
    # Build internal message list from the conversation
    internal_messages = _build_internal_messages(messages)
    answer = agent.process_prompt(user_input, internal_messages)
    return _make_response(answer or "(No response)", model_name)


def _stream_response(agent, user_input, messages, model_name):
    """Run agent in background thread, stream tokens via SSE."""
    token_queue = queue.Queue()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _on_token(accumulated_text):
        """Callback invoked by BaseAgent as tokens arrive."""
        token_queue.put(("token", accumulated_text))

    def _run():
        internal_messages = _build_internal_messages(messages)
        agent._on_token = _on_token
        try:
            answer = agent.process_prompt(user_input, internal_messages)
            token_queue.put(("done", answer or "(No response)"))
        except Exception as e:
            token_queue.put(("error", str(e)))
        finally:
            agent._on_token = None

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    last_sent = ""
    while True:
        try:
            event_type, data = token_queue.get(timeout=120)
        except queue.Empty:
            break

        if event_type == "token":
            # Send the delta (new characters since last send)
            delta = data[len(last_sent):]
            if delta:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                last_sent = data

        elif event_type == "done":
            # Send any remaining text not yet streamed
            final = data
            if len(final) > len(last_sent):
                remaining = final[len(last_sent):]
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": remaining},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Send finish chunk
            finish_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            break

        elif event_type == "error":
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\nError: {data}"},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            break


def _stream_canned(message, model_name):
    """Yield a canned response as a streaming SSE for OWUI internal requests."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": message}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finish = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _stream_error(message):
    """Yield a single error chunk for streaming."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "error",
        "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _is_owui_internal(content):
    """Detect Open WebUI internal task prompts (RAG, follow-ups, search queries)."""
    if not content:
        return False
    markers = (
        "### Task:\nRespond to the user query using the provided context",
        "### Task:\nSuggest 3-5 relevant follow-up",
        "### Task:\nAnalyze the chat history to determine the necessity of generating search queries",
        "### Task:\nGenerate a concise",
    )
    return any(content.startswith(m) for m in markers)


def _unwrap_owui_input(content):
    """Handle Open WebUI internal prompts that wrap the real user message.

    Returns (user_message, canned_response):
    - For RAG wrappers: extracts the real user query, canned_response is None
    - For follow-ups/search/title: user_message is None, returns canned JSON
    - For normal messages: returns (content, None) unchanged
    """
    if not content:
        return content, None

    # RAG context wrapper — extract the real user message after </context>
    if content.startswith("### Task:\nRespond to the user query using the provided context"):
        # The actual user message comes after the closing </context> tag
        match = re.search(r"</context>\s*\n*(.*)", content, re.DOTALL)
        if match:
            real_msg = match.group(1).strip()
            if real_msg:
                return real_msg, None
        # Fallback: couldn't extract, skip
        return None, None

    # Follow-up suggestions — return canned empty response
    if content.startswith("### Task:\nSuggest 3-5 relevant follow-up"):
        return None, '{"follow_ups": []}'

    # Search query generation — return empty queries
    if content.startswith("### Task:\nAnalyze the chat history to determine the necessity of generating search queries"):
        return None, '{"queries": []}'

    # Title generation — return generic title
    if content.startswith("### Task:\nGenerate a concise"):
        return None, "SheLLM Chat"

    # Normal message
    return content, None


def _build_internal_messages(messages):
    """Convert Open WebUI messages to SheLLM internal format.

    - Strips image content blocks (handled separately by the endpoint)
    - Skips Open WebUI's internal "### Task:" wrapper prompts
    - Skips system messages from Open WebUI (SheLLM generates its own)
    """
    internal = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Skip Open WebUI system messages — SheLLM has its own
        if role == "system":
            continue

        # Handle multimodal content arrays
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts) if text_parts else ""

        # Skip Open WebUI internal task prompts
        if _is_owui_internal(content):
            continue

        if content and role in ("user", "assistant"):
            internal.append({"role": role, "content": content})

    return internal


def _make_response(content, model_name):
    """Build a standard OpenAI chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SheLLM OpenAI-compatible API Server")
    parser.add_argument("--port", type=int, default=8091, help="Port to listen on (default: 8091)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    # Pre-initialize registry
    _get_registry()
    print(f"SheLLM API server starting on {args.host}:{args.port}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
