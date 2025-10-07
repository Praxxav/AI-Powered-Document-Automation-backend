# backend/agent/bootstrap_agent.py
import httpx
import logging
from agent.templatizer import templatizer_agent
from config import settings

class WebBootstrapAgent:
    def __init__(self):
        self.api_key = settings.EXA_API_KEY  # or Gemini/Serper API key
        self.base_url = "https://api.exa.ai/search"

    async def fetch_public_examples(self, query: str):
        """Query exa.ai or a fallback search API for legal doc exemplars."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.base_url,
                    headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                    json={"query": query, "numResults": 3}
                )
                data = response.json()
                return [
                    {
                        "title": item["title"],
                        "url": item["url"],
                        "snippet": item.get("text", "")
                    }
                    for item in data.get("results", [])
                ]
        except Exception as e:
            logging.error(f"Exa.ai fetch error: {e}")
            return []

    async def bootstrap_template(self, query: str):
        """Fetch exemplar, clean it, templatize, and return new Template data."""
        examples = await self.fetch_public_examples(query)
        if not examples:
            return None

        # Pick best match (first for now)
        exemplar = examples[0]
        text = exemplar["snippet"]

        # Run templatization
        template_data = await templatizer_agent.templatize_text(text, query)

        return {
            "title": f"Auto Template: {exemplar['title']}",
            "source_url": exemplar["url"],
            **template_data
        }

bootstrap_agent = WebBootstrapAgent()
