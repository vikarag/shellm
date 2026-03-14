"""Research agent — academic research via OpenAI Responses API."""

from agents.base_agent import BaseAgent


class ResearcherAgent(BaseAgent):
    """Conducts academic research using web search with scholarly focus."""

    def research(self, query):
        """Research a topic with academic focus.

        Args:
            query: Research question or topic

        Returns:
            Research findings as text
        """
        academic_query = (
            f"Conduct thorough academic research on: {query}\n\n"
            "Focus on peer-reviewed sources, official documentation, and authoritative references. "
            "Cite sources when possible."
        )
        try:
            response = self.client.responses.create(
                model=self.config.model,
                input=academic_query,
                tools=[{"type": "web_search"}],
            )
            return response.output_text or "(No results)"
        except Exception as e:
            return f"Research error: {e}"
