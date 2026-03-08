#!/home/gslee/shellm/venv/bin/python3
"""SheLLM Chat Engine -- DeepSeek for general conversation and reasoning."""

from base_chat import BaseChatClient

# ── Settings ────────────────────────────────────────────────────────
MODEL = "deepseek-chat"       # "deepseek-reasoner" or "deepseek-chat"
STREAM = True                 # True for streaming, False for batch
TEMPERATURE = 0.1             # 0.0-2.0 (only works with deepseek-chat; ignored by deepseek-reasoner)
# ────────────────────────────────────────────────────────────────────


class DeepSeekChat(BaseChatClient):
    MODEL = MODEL
    BANNER_NAME = "SheLLM Chat"
    ENV_VAR = "DEEPSEEK_API_KEY"
    BASE_URL = "https://api.deepseek.com"
    STREAM = STREAM
    TEMPERATURE = TEMPERATURE
    HAS_REASONING = True

    @property
    def SUPPORTS_TOOLS(self):
        return self.MODEL != "deepseek-reasoner"

    def build_params(self, messages):
        params = {"model": self.MODEL, "messages": messages, "stream": self.STREAM}
        if self.TEMPERATURE is not None and self.MODEL != "deepseek-reasoner":
            params["temperature"] = self.TEMPERATURE
        if self.SUPPORTS_TOOLS:
            from base_chat import TOOLS
            from mcp_manager import MCPManager
            params["tools"] = TOOLS + MCPManager.get_instance().get_tools()
        return params

    def build_no_tool_params(self, messages):
        params = {"model": self.MODEL, "messages": messages, "stream": self.STREAM}
        if self.TEMPERATURE is not None and self.MODEL != "deepseek-reasoner":
            params["temperature"] = self.TEMPERATURE
        return params

    def format_banner(self):
        tools_note = " | Auto web search enabled" if self.SUPPORTS_TOOLS else ""
        return f"{self.BANNER_NAME} (model: {self.MODEL}, stream: {self.STREAM}, temp: {self.TEMPERATURE or 'default'}{tools_note})"


if __name__ == "__main__":
    DeepSeekChat().run()
