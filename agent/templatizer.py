from .base_agent import SimpleAgent
from config import settings
import json
import logging

# --- Agent System Prompt ---

TEMPLATE_GENERATOR_PROMPT = """
You are an expert legal and financial document templating assistant. Your primary goal is to analyze the provided document text and transform it into a reusable Markdown template with a YAML front-matter block.

Follow these steps precisely:
1.  **Identify Reusable Fields**: Scan the document for specific, instance-level details that would change with each use. These are your variables. Examples include names, dates, addresses, policy numbers, monetary amounts, case numbers, etc.
2.  **Deduplicate and Generalize**: Logically group similar fields. For example, "Claimant Name" and "Name of Claimant" should become a single variable. Use clear, generic, snake_case keys for variables (e.g., `claimant_full_name`, `incident_date`).
3.  **Generate Variable Metadata**: For each variable, create a JSON object with the following keys:
    - `key`: The snake_case identifier.
    - `label`: A human-readable name for the variable (e.g., "Claimant's Full Name").
    - `description`: A brief explanation of what the variable is for.
    - `example`: A realistic example value from the text.
    - `required`: A boolean (`true` or `false`) indicating if the document would be incomplete without it.
4.  **Create Similarity Tags**: Generate a list of 5-7 relevant lowercase keywords for search and retrieval (e.g., "insurance", "notice", "india", "motor").
5.  **Construct the Template Body**: Replace the identified variable values in the original text with `{{key}}` placeholders.
6.  **Assemble the Final Output**: Combine everything into a single Markdown file format. The final output MUST start with `---` and end with `---` for the YAML block, followed by the template body. All keys in the YAML block must be snake_case.

**Example Output Format:**
---
title: Incident Notice to Insurer
file_description: A standard notice sent to an insurance company to report an incident and initiate a claim.
jurisdiction: IN
doc_type: legal_notice
variables:
  - key: claimant_full_name
    label: "Claimant's full name"
    description: "The full name of the person or entity raising the claim."
    example: "John Doe"
    required: true
similarity_tags: ["insurance", "notice", "claim", "india", "incident report"]
---

Dear Sir/Madam,

On {{incident_date}}, {{claimant_full_name}} hereby notifies you under Policy {{policy_number}}...
"""

# --- Agent Instance ---

templatizer_agent = SimpleAgent(
    name="TemplatizerAgent",
    role="Generates a structured Markdown template from raw document text.",
    api_key=settings.GEMINI_API_KEY,
    system_prompt=TEMPLATE_GENERATOR_PROMPT,
    model="gemini-2.5-flash" 
)


# --- Wrapper Function for Bootstrap Agent ---

async def templatize_text(text: str, query: str) -> dict:
    """
    Wrapper function to convert text into a template using the templatizer_agent.
    
    Args:
        text: The source text to templatize
        query: The original query/context for better understanding
        
    Returns:
        Dict containing template data in the expected format
    """
    prompt = f"""
Query Context: {query}

Document Text to Templatize:
{text}

Please analyze this document and generate a complete Markdown template with YAML front-matter following the format specified in your instructions.
"""
    
    try:
        # Call the agent's process method
        result = await templatizer_agent.process(prompt)
        
        # Parse the markdown result to extract components
        result = result.strip()
        
        # Split YAML front-matter from body
        if result.startswith('---'):
            parts = result.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                template_body = parts[2].strip()
            else:
                logging.warning("Invalid YAML format, using full text as body")
                yaml_content = ""
                template_body = result
        else:
            yaml_content = ""
            template_body = result
        
        # Parse YAML-like content (simplified parsing)
        import yaml
        try:
            yaml_data = yaml.safe_load(yaml_content) if yaml_content else {}
        except:
            logging.warning("Failed to parse YAML, using empty dict")
            yaml_data = {}
        
        # Return in the expected format
        return {
            "title": yaml_data.get("title", f"Template for {query}"),
            "file_description": yaml_data.get("file_description", ""),
            "jurisdiction": yaml_data.get("jurisdiction", ""),
            "doc_type": yaml_data.get("doc_type", ""),
            "variables": yaml_data.get("variables", []),
            "similarity_tags": yaml_data.get("similarity_tags", []),
            "template_content": template_body,
            "template_markdown": result,  # Full markdown with YAML front-matter
            "full_markdown": result  # Keep for backwards compatibility
        }
        
    except Exception as e:
        logging.error(f"Templatization failed: {e}")
        raise


# Make the function available as an attribute of the agent for backwards compatibility
templatizer_agent.templatize_text = templatize_text