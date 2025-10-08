

import httpx
import logging
import json
import re
import yaml
from agent.templatizer import templatizer_agent
from config import settings

class WebBootstrapAgent:
    def __init__(self):
        self.api_key = settings.EXA_API_KEY
        self.base_url = "https://api.exa.ai/search"

    async def fetch_public_examples(self, query: str):
        """Query exa.ai for legal document exemplars with full text content."""
        try:
            # Enhanced query for better legal document results
            enhanced_query = f"{query} legal document template sample format"
            
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": enhanced_query,
                        "numResults": 3,
                        "type": "neural",
                        "useAutoprompt": True,
                        "contents": {
                            "text": True  # Get full text content
                        }
                    }
                )
                
                if response.status_code != 200:
                    logging.error(f"Exa.ai API error: {response.status_code} - {response.text}")
                    return []
                
                data = response.json()
                results = []
                
                for item in data.get("results", []):
                    # Extract text from different possible fields
                    text_content = (
                        item.get("text") or 
                        item.get("content") or 
                        item.get("snippet") or 
                        ""
                    )
                    
                    if text_content:
                        results.append({
                            "title": item.get("title", "Untitled"),
                            "url": item.get("url", ""),
                            "text": text_content
                        })
                
                logging.info(f"✅ Exa.ai returned {len(results)} results with content")
                return results
                
        except Exception as e:
            logging.error(f"❌ Exa.ai fetch error: {e}", exc_info=True)
            return []

    async def bootstrap_template(self, query: str):
        """
        Fetch exemplar from web, templatize it with variable extraction.
        Returns complete template data with proper YAML front-matter.
        """
        try:
            # Step 1: Search for examples
            examples = await self.fetch_public_examples(query)
            if not examples:
                logging.warning("No examples found from Exa.ai")
                return None

            # Step 2: Pick best match
            exemplar = examples[0]
            text = exemplar["text"]
            
            if not text or len(text.strip()) < 100:
                logging.warning(f"Insufficient text content: {len(text)} chars")
                # Try next result
                if len(examples) > 1:
                    exemplar = examples[1]
                    text = exemplar["text"]
                else:
                    return None

            logging.info(f"📄 Processing: {exemplar['title']} ({len(text)} chars)")

            # Step 3: Templatize with FULL variable extraction
            # This is the KEY fix - ensure templatizer extracts variables
            template_markdown = await templatizer_agent.process(text)
            
            if not template_markdown:
                logging.error("Templatizer returned empty result")
                return None

            # Step 4: Parse and validate the template
            template_data = self._parse_template_markdown(template_markdown)
            
            if not template_data:
                logging.error("Failed to parse template markdown")
                return None

            # Step 5: Ensure variables exist
            if not template_data.get("variables") or len(template_data["variables"]) == 0:
                logging.warning("⚠️  No variables extracted, adding fallback extraction")
                # Re-extract variables more aggressively
                template_data = await self._aggressive_variable_extraction(
                    text, 
                    template_markdown,
                    template_data
                )

            # Step 6: Return complete data structure
            return {
                "template_markdown": template_markdown,
                "full_markdown": template_markdown,  # Alias for compatibility
                "title": template_data.get("title") or f"Template: {exemplar['title'][:50]}",
                "file_description": template_data.get("file_description") or f"Sourced from web for: {query}",
                "jurisdiction": template_data.get("jurisdiction", ""),
                "doc_type": template_data.get("doc_type") or self._infer_doc_type(query),
                "similarity_tags": template_data.get("similarity_tags") or self._generate_tags(query),
                "source_url": exemplar["url"],
                "source_title": exemplar["title"],
                "variables": template_data.get("variables", [])
            }

        except Exception as e:
            logging.error(f"❌ Bootstrap template failed: {e}", exc_info=True)
            return None

    def _parse_template_markdown(self, markdown: str) -> dict:
        """Parse YAML front-matter from markdown template."""
        try:
            if not markdown.strip().startswith("---"):
                logging.warning("No YAML front-matter found")
                return {"body": markdown, "variables": []}

            parts = markdown.split("---", 2)
            if len(parts) < 3:
                return {"body": markdown, "variables": []}

            yaml_str = parts[1].strip()
            body = parts[2].strip()

            # Parse YAML
            metadata = yaml.safe_load(yaml_str) or {}
            
            return {
                "title": metadata.get("title", ""),
                "file_description": metadata.get("file_description", ""),
                "jurisdiction": metadata.get("jurisdiction", ""),
                "doc_type": metadata.get("doc_type", ""),
                "similarity_tags": metadata.get("similarity_tags", []),
                "variables": metadata.get("variables", []),
                "body": body
            }

        except Exception as e:
            logging.error(f"Error parsing markdown: {e}")
            return {"body": markdown, "variables": []}

    async def _aggressive_variable_extraction(
        self, 
        original_text: str,
        template_markdown: str,
        existing_data: dict
    ) -> dict:
        """
        Fallback: Use Gemini directly to extract variables if templatizer missed them.
        This ensures variables are ALWAYS extracted.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')

            prompt = f"""Extract ALL variable fields from this legal document.
Look for: names, dates, addresses, amounts, case numbers, policy numbers, parties, etc.

DOCUMENT:
{original_text[:3000]}

Return ONLY JSON array:
[
  {{
    "key": "party_name",
    "label": "Party Name",
    "description": "Name of the primary party",
    "example": "John Doe",
    "required": true,
    "type": "string"
  }}
]

Return ONLY the JSON array, no other text."""

            response = await model.generate_content_async(prompt)
            text = response.text.strip()
            
            # Clean markdown
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\n?", "", text).rstrip("`").strip()

            variables = json.loads(text)
            
            # Validate
            if isinstance(variables, list) and len(variables) > 0:
                existing_data["variables"] = variables
                logging.info(f"✅ Extracted {len(variables)} variables via aggressive extraction")
            
            return existing_data

        except Exception as e:
            logging.error(f"Aggressive extraction failed: {e}")
            return existing_data

    def _infer_doc_type(self, query: str) -> str:
        """Infer document type from query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["notice", "notification"]):
            return "legal_notice"
        elif any(word in query_lower for word in ["contract", "agreement"]):
            return "contract"
        elif any(word in query_lower for word in ["lease", "rent"]):
            return "lease_agreement"
        elif any(word in query_lower for word in ["complaint", "petition"]):
            return "court_filing"
        else:
            return "legal_document"

    def _generate_tags(self, query: str) -> list:
        """Generate similarity tags from query."""
        words = query.lower().split()
        # Remove common words
        stop_words = {"a", "the", "in", "on", "for", "to", "of", "draft", "create"}
        tags = [w for w in words if w not in stop_words and len(w) > 2]
        return tags[:5]

bootstrap_agent = WebBootstrapAgent()