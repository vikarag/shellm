#!/home/gslee/shellm/venv/bin/python3
"""SheLLM Code Engine -- Kimi K2.5 for code generation, analysis, and refactoring."""

from base_chat import BaseChatClient

# ── Settings ────────────────────────────────────────────────────────
THINKING = True   # True = thinking mode (CoT), False = instant mode
# ────────────────────────────────────────────────────────────────────


class KimiChat(BaseChatClient):
    MODEL = "kimi-k2.5"
    BANNER_NAME = "SheLLM Code"
    ENV_VAR = "MOONSHOT_API_KEY"
    BASE_URL = "https://api.moonshot.ai/v1"
    STREAM = True
    TEMPERATURE = None
    HAS_REASONING = True

    def build_params(self, messages):
        params = super().build_params(messages)
        if not THINKING:
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        return params

    def format_banner(self):
        mode = "thinking" if THINKING else "instant"
        return f"{self.BANNER_NAME} (model: {self.MODEL}, stream: {self.STREAM}, temp: {self.TEMPERATURE or 'default'}, mode: {mode})"


if __name__ == "__main__":
    KimiChat().run()
