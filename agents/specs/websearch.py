"""Web search agent — live web search via OpenAI Responses API."""

from agents.base_agent import BaseAgent


class WebSearchAgent(BaseAgent):
    """Performs web searches using OpenAI's responses.create with web_search tool."""

    def search(self, query):
        """Search the web for a query.

        Args:
            query: Search query string

        Returns:
            Search results as text
        """
        try:
            response = self.client.responses.create(
                model=self.config.model,
                input=query,
                tools=[{"type": "web_search"}],
            )
            return response.output_text or "(No results)"
        except Exception as e:
            return f"Web search error: {e}"
