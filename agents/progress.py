"""Thread-safe progress queue for inter-agent communication."""

import threading
import time
from collections import deque


class ProgressQueue:
    """Thread-safe queue for progress events between chat and updater agents."""

    def __init__(self, maxlen=100):
        self._events = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, event_type, step="", detail="", tool_calls=None):
        """Push a progress event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "step": step,
            "detail": detail,
        }
        if tool_calls:
            event["tool_calls"] = tool_calls
        with self._lock:
            self._events.append(event)

    def snapshot(self):
        """Read all events (non-destructive)."""
        with self._lock:
            return list(self._events)

    def clear(self):
        """Reset after task completion."""
        with self._lock:
            self._events.clear()

    def summary(self):
        """Return a concise text summary of current progress."""
        events = self.snapshot()
        if not events:
            return "No activity recorded."

        lines = []
        for e in events[-10:]:  # Last 10 events
            ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
            if e["type"] == "tool_call":
                lines.append(f"[{ts}] Tool: {e['step']} — {e['detail']}")
            elif e["type"] == "progress":
                lines.append(f"[{ts}] Progress: {e['step']} — {e['detail']}")
            elif e["type"] == "plan_start":
                lines.append(f"[{ts}] Plan started: {e['detail']}")
            elif e["type"] == "plan_end":
                lines.append(f"[{ts}] Plan completed: {e['detail']}")
            else:
                lines.append(f"[{ts}] {e['type']}: {e['detail']}")

        return "\n".join(lines)


# Global progress queue instance
progress_queue = ProgressQueue()
