from agent.base_agent import BaseAgent
from config import settings
import json

# CREATED BY "UOIONHHC"

PREFILLER_PROMPT_TEMPLATE = """
You are an expert data extraction assistant. Your task is to extract information from a user's query and map it to a list of predefined template variables.

Respond ONLY with a valid JSON object containing the keys of the variables that you were able to find in the query. Do not include keys for variables you could not find.

User Query:
"{query}"

Template Variables:
{variables_json}

Example: If the query is "Draft a rental agreement for Jane Smith" and a variable is `{{ "key": "tenant_name", "label": "Tenant's Name" }}`, your output should be `{{"tenant_name": "Jane Smith"}}`.
"""

class PrefillerAgent(BaseAgent):
    """An agent that pre-fills template variables from a user query."""
    async def process(self, input_data: dict, context=None) -> dict:
        prompt = PREFILLER_PROMPT_TEMPLATE.format(query=input_data['query'], variables_json=input_data['variables_json'])
        messages = [{"role": "user", "content": prompt}]
        raw_response = await self._make_api_call(messages, temperature=0.1, response_format="json")
        return json.loads(raw_response)

prefiller_agent = PrefillerAgent(name="PrefillerAgent", role="Prefills template variables from a query", api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash")