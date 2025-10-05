from .base_agent import SimpleAgent
from config import settings
from pydantic import BaseModel, Field
import json

# --- Agent System Prompts ---

DOC_TYPE_PROMPT = """
You are an expert banking document classification AI.
Your task is to identify what kind of banking or financial document has been provided.

Possible document types include:
- Bank Statement
- Loan Agreement
- KYC Form
- Account Opening Form
- Credit Report
- Transaction Report
- Compliance Document
- Audit Report
- Other

Return your result as a JSON object in this format:
{
  "document_type": "Bank Statement",
  "confidence": "0.93"
}
"""


SUMMARIZER_PROMPT = """
You are an expert banking assistant.
Your task is to summarize the given banking or financial document clearly and concisely.
Focus on:
- The main purpose of the document
- Key financial figures (amounts, rates, periods)
- Any parties or organizations involved
- Important decisions, approvals, or outcomes

The summary should be written in a professional, neutral tone suitable for a banking professional.
"""

ENTITY_EXTRACTOR_PROMPT = """
You are a precise financial data extraction AI. From the given banking or financial document, extract the following entities.

Return a valid JSON object only with these keys:
- "parties": list of customers, banks, or institutions mentioned
- "account_numbers": list of any account or application numbers found
- "dates": list of dates mentioned
- "amounts": list of money values (with currency symbols if available)
- "locations": list of branches, cities, or countries mentioned
- "financial_terms": list of financial or regulatory terms
- "document_references": list of any reference numbers or document IDs

If a field has no data, return an empty list.

Example output:
{
  "parties": ["John Smith", "HDFC Bank"],
  "account_numbers": ["XXXX-1234"],
  "dates": ["2023-11-01"],
  "amounts": ["₹1,50,000", "$2500"],
  "locations": ["Mumbai", "New York"],
  "financial_terms": ["interest rate", "EMI", "credit score"],
  "document_references": ["APP-394023"]
}
"""

QA_PROMPT = "You are a helpful legal assistant. Based *only* on the context provided from a legal document, answer the user's question. If the answer is not found in the context, state that the information is not available in the document."

# --- Pydantic Models for Data Validation ---

class BankingEntities(BaseModel):
    parties: list[str] = Field(default_factory=list)
    account_numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    financial_terms: list[str] = Field(default_factory=list)
    document_references: list[str] = Field(default_factory=list)

summarizer_agent = SimpleAgent(name="LegalSummarizer", role="Summarizes legal documents", api_key=settings.GEMINI_API_KEY, system_prompt=SUMMARIZER_PROMPT, model="gemini-2.5-flash")
entity_extractor_agent = SimpleAgent(name="EntityExtractor", role="Extracts structured legal entities", api_key=settings.GEMINI_API_KEY, system_prompt=ENTITY_EXTRACTOR_PROMPT, model="gemini-2.5-flash")
qa_agent = SimpleAgent(name="QuestionAnsweringAgent", role="Answers questions about a document", api_key=settings.GEMINI_API_KEY, system_prompt=QA_PROMPT, model="gemini-2.5-flash")