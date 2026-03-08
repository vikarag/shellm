#!/home/gslee/shellm/venv/bin/python3
"""SheLLM Research Engine -- GPT-5 Mini for web research, summarization, and fact-finding."""

from base_chat import BaseChatClient


class GPT5MiniChat(BaseChatClient):
    MODEL = "gpt-5-mini"
    BANNER_NAME = "SheLLM Research"
    ENV_VAR = "OPENAI_API_KEY"
    STREAM = True
    TEMPERATURE = None


if __name__ == "__main__":
    GPT5MiniChat().run()
