from .base_agent import BaseAgent
from config import settings

QUESTION_GENERATOR_PROMPT_TEMPLATE = """
You are an expert at creating user-friendly questions. Your task is to transform a variable's technical details into a polite, clear, and unambiguous question for an end-user.

**Instructions:**
1.  Generate a single, natural-language question based on the provided label and description.
2.  The question should be polite and easy to understand for someone who is not a technical user.
3.  If the description provides context or specific format requirements (like "ISO 8601" for a date or "as printed on schedule"), incorporate that as a helpful hint.
4.  Do NOT include the variable's `key` (e.g., `policy_number`) in the question.
5.  Respond with ONLY the generated question as a plain string, with no extra text, labels, or JSON formatting.

**Variable Details:**
- **Label:** "{label}"
- **Description:** "{description}"

**Example:**
- **Input:** Label: "Policy number", Description: "Insurance policy reference as printed on schedule."
- **Your Output:** What is the insurance policy number, exactly as it appears on the policy schedule?

**Your Turn:**
"""

class QuestionGeneratorAgent(BaseAgent):
    """An agent that generates a human-friendly question from variable metadata."""
    async def process(self, input_data: dict, context=None) -> str:
        prompt = QUESTION_GENERATOR_PROMPT_TEMPLATE.format(label=input_data.get('label', ''), description=input_data.get('description', ''))
        messages = [{"role": "user", "content": prompt}]
        question_text = await self._make_api_call(messages, temperature=0.2)
        return question_text.strip().replace('"', '') # Clean up potential quotes

question_generator_agent = QuestionGeneratorAgent(name="QuestionGeneratorAgent", role="Generates human-friendly questions for template variables.", api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash")