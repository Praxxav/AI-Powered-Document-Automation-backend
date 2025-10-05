from .base_agent import SimpleAgent
from config import settings

# --- Agent System Prompt ---

DOCUMENT_CLASSIFIER_PROMPT = """
You are an expert document classification AI. Your task is to determine the primary domain of the provided document text.
The main domains are "Banking" and "Legal".

- If the document contains financial terms, transaction details, account numbers, bank names, it is a "Banking" document.
- If the document contains legal arguments, case law, statutes, court names, plaintiff/defendant roles, it is a "Legal" document.

Based on your analysis, return a single word: "Banking", "Legal", or "Other".
Do not provide any explanation or other text.
"""

# --- Agent Instance ---

classifier_agent = SimpleAgent(
    name="DocumentClassifier",
    role="Classifies documents into 'Banking' or 'Legal' categories.",
    api_key=settings.GEMINI_API_KEY,
    system_prompt=DOCUMENT_CLASSIFIER_PROMPT,
    model="gemini-2.5-flash"
)