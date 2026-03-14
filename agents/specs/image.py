"""Image analysis agent — vision via GPT-5 Mini."""

from agents.base_agent import BaseAgent


class ImageAgent(BaseAgent):
    """Analyzes images using the vision API."""

    def analyze_image(self, b64_data, prompt="Describe and analyze this image in detail."):
        """Analyze a base64-encoded image.

        Args:
            b64_data: Base64-encoded image data
            prompt: Analysis prompt

        Returns:
            Analysis text
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
                ],
            }],
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content or "(No response)"
