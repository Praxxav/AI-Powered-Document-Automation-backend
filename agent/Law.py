from .base_agent import SimpleAgent
from config import settings
from pydantic import BaseModel, Field
import json

# --- Agent System Prompts ---

SUMMARIZER_PROMPT = """
You are an expert legal assistant. Your task is to provide a concise, neutral summary of the provided legal document text. 
Focus on the key facts, the main legal arguments presented, and any stated outcomes or decisions.
The summary should be clear and easy for a legal professional to quickly understand the document's essence.
"""

ENTITY_EXTRACTOR_PROMPT = """
You are a highly accurate legal data extraction AI. From the document text provided, extract the following entities.
Your response MUST be a valid JSON object that conforms to the required schema. Do not include any text outside of the JSON object.

The JSON object should have the following keys:
- "parties": A list of all individuals or organizations mentioned as parties (e.g., Plaintiff, Defendant, Appellant).
- "dates": A list of all specific dates mentioned.
- "locations": A list of all geographical locations (cities, states, courts).
- "legal_terms": A list of key legal terms, statutes, or case law citations.
- "case_numbers": A list of any case numbers or docket numbers identified.

If a category has no entities, return an empty list for that key.

Example Output:
{
  "parties": ["John Doe (Plaintiff)", "ACME Corp (Defendant)"],
  "dates": ["2023-01-15", "2023-03-20"],
  "locations": ["Southern District of New York"],
  "legal_terms": ["breach of contract", "Rule 12(b)(6)"],
  "case_numbers": ["1:23-cv-01234"]
}
"""

QA_PROMPT = "You are a helpful legal assistant. Based *only* on the context provided from a legal document, answer the user's question. If the answer is not found in the context, state that the information is not available in the document."

# --- Pydantic Models for Data Validation ---

class LegalEntities(BaseModel):
    parties: list[str] = Field(..., description="A list of all individuals or organizations mentioned as parties (e.g., Plaintiff, Defendant, Appellant).")
    dates: list[str] = Field(..., description="A list of all specific dates mentioned.")
    locations: list[str] = Field(..., description="A list of all geographical locations (cities, states, courts).")
    legal_terms: list[str] = Field(..., description="A list of key legal terms, statutes, or case law citations.")
    case_numbers: list[str] = Field(..., description="A list of any case numbers or docket numbers identified.")



summarizer_agent = SimpleAgent(name="LegalSummarizer", role="Summarizes legal documents", api_key=settings.GEMINI_API_KEY, system_prompt=SUMMARIZER_PROMPT, model="gemini-2.5-flash")
entity_extractor_agent = SimpleAgent(name="EntityExtractor", role="Extracts structured legal entities", api_key=settings.GEMINI_API_KEY, system_prompt=ENTITY_EXTRACTOR_PROMPT, model="gemini-2.5-flash")
qa_agent = SimpleAgent(name="QuestionAnsweringAgent", role="Answers questions about a document", api_key=settings.GEMINI_API_KEY, system_prompt=QA_PROMPT, model="gemini-2.5-flash")