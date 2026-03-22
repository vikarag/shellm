"""Heartbeat task scheduler for SheLLM — checks SQLite every 60s for due tasks.

Supports telegram_message and shell_command task types.
Singleton pattern matching MCPManager.
"""

import json
import logging
import subprocess
import threading
import urllib.request
from datetime import datetime, timezone, timedelta

from db import get_connection
from command_runner import is_blocked

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


class TaskScheduler:
    """Singleton heartbeat scheduler that ticks every 60 seconds."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._bot_token = None
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self, bot_token=None):
        """Start the heartbeat thread. Safe to call multiple times."""
        if self._started:
            return
        if bot_token:
            self._bot_token = bot_token
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        self._started = True
        logger.info("TaskScheduler started (token=%s)", "yes" if self._bot_token else "no")

    def stop(self):
        """Signal the heartbeat thread to stop."""
        self._stop_event.set()
        self._started = False

    def _heartbeat_loop(self):
        """Main loop — tick every 60 seconds until stopped."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error("TaskScheduler tick error: %s", e)
            self._stop_event.wait(timeout=60)

    def _tick(self):
        """Query pending tasks that are due and execute them."""
        now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, task_type, payload FROM scheduled_tasks "
            "WHERE status = 'pending' AND scheduled_at <= ?",
            (now,),
        ).fetchall()

        for row in rows:
            task_id, task_type, payload_str = row["id"], row["task_type"], row["payload"]
            try:
                payload = json.loads(payload_str)
                if task_type == "telegram_message":
                    result = self._send_telegram(payload)
                elif task_type == "shell_command":
                    result = self._run_shell(payload)
                else:
                    result = f"Unknown task_type: {task_type}"

                status = "failed" if result.startswith("BLOCKED") else "done"
                conn.execute(
                    "UPDATE scheduled_tasks SET status = ?, result = ?, executed_at = ? WHERE id = ?",
                    (status, result, datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), task_id),
                )
            except Exception as e:
                conn.execute(
                    "UPDATE scheduled_tasks SET status = 'failed', result = ?, executed_at = ? WHERE id = ?",
                    (str(e), datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), task_id),
                )
            conn.commit()

    def _send_telegram(self, payload):
        """Send a Telegram message via HTTP API."""
        if not self._bot_token:
            return "No bot token configured"
        chat_id = payload.get("chat_id")
        text = payload.get("message", "")
        if not chat_id or not text:
            return "Missing chat_id or message in payload"

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        success = result.get("ok", False)
        self._log_send("schedule_task", chat_id, text, success)
        return "sent" if success else f"Telegram error: {result}"

    @staticmethod
    def _log_send(tool_name, chat_id, text, success):
        """Log outgoing Telegram message for auditability."""
        import os
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_outbox.jsonl")
        entry = {
            "timestamp": datetime.now(KST).isoformat(),
            "tool": tool_name,
            "chat_id": chat_id,
            "text": text[:500],
            "success": success,
        }
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _run_shell(self, payload):
        """Execute a shell command with safety check."""
        command = payload.get("command", "")
        if not command:
            return "Missing command in payload"
        blocked = is_blocked(command)
        if blocked:
            return f"BLOCKED: matches dangerous pattern: {blocked}"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            return output or f"(exit code: {result.returncode})"
        except subprocess.TimeoutExpired:
            return "Command timed out after 120s"
        except Exception as e:
            return f"Error: {e}"


# ── Public functions for LLM tools ──────────────────────────────────


def schedule_task(task_type, payload, delay_minutes=None, scheduled_at=None, chat_id=None):
    """Schedule a task. Returns confirmation string.

    Args:
        task_type: 'telegram_message' or 'shell_command'
        payload: dict with task-specific data
        delay_minutes: minutes from now (alternative to scheduled_at)
        scheduled_at: ISO datetime string (alternative to delay_minutes)
        chat_id: Telegram chat_id to inject into telegram_message payloads
    """
    if task_type not in ("telegram_message", "shell_command"):
        return f"Invalid task_type: {task_type}. Use 'telegram_message' or 'shell_command'."

    now = datetime.now(KST)

    if scheduled_at:
        target = scheduled_at
    elif delay_minutes is not None:
        target = (now + timedelta(minutes=int(delay_minutes))).strftime("%Y-%m-%d %H:%M:%S")
    else:
        return "Provide either delay_minutes or scheduled_at."

    # Auto-inject chat_id for telegram tasks
    if task_type == "telegram_message" and chat_id and "chat_id" not in payload:
        payload["chat_id"] = chat_id

    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO scheduled_tasks (created_at, scheduled_at, task_type, payload, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (now.strftime("%Y-%m-%d %H:%M:%S"), target, task_type, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    task_id = cursor.lastrowid
    return f"Scheduled task #{task_id} ({task_type}) for {target} KST."


def list_scheduled_tasks(status="pending"):
    """List scheduled tasks filtered by status."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, created_at, scheduled_at, task_type, payload, status, result, executed_at "
        "FROM scheduled_tasks WHERE status = ? ORDER BY scheduled_at",
        (status,),
    ).fetchall()

    if not rows:
        return f"No {status} tasks found."

    lines = [f"Scheduled tasks ({status}): {len(rows)} found\n"]
    for r in rows:
        payload_preview = r["payload"][:80]
        lines.append(
            f"  #{r['id']}  {r['task_type']}  scheduled: {r['scheduled_at']}"
            f"  payload: {payload_preview}"
        )
        if r["executed_at"]:
            result_preview = (r["result"] or "")[:80]
            lines.append(f"       executed: {r['executed_at']}  result: {result_preview}")
    return "\n".join(lines)


def cancel_scheduled_task(task_id):
    """Cancel a pending task by ID."""
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE scheduled_tasks SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
        (task_id,),
    )
    conn.commit()
    if cursor.rowcount == 0:
        return f"Task #{task_id} not found or not pending."
    return f"Task #{task_id} cancelled."
