"""Updater agent — responds when chat is busy, reports progress."""

from agents.base_agent import BaseAgent
from agents.progress import progress_queue


class UpdaterAgent(BaseAgent):
    """Responds to users when the primary chat agent is busy processing."""

    def _ensure_system_message(self, messages):
        """Override to prepend progress snapshot to system message."""
        super()._ensure_system_message(messages)

        # Prepend progress context
        snapshot = progress_queue.summary()
        if snapshot and snapshot != "No activity recorded.":
            progress_context = (
                "\n\n--- CURRENT ACTIVITY (from the primary chat agent) ---\n"
                f"{snapshot}\n"
                "--- END ACTIVITY ---\n\n"
                "The user is messaging while the primary agent is busy. "
                "Use the activity log above to inform them about what's happening. "
                "Be concise and helpful."
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += progress_context
