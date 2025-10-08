# Lexi Backend
FastAPI backend for transforming unstructured legal documents into structured, reusable templates and enabling intelligent drafting workflows.

## Overview
Backend powers an AI-assisted document automation system that:

Ingests .docx or .pdf files.

Extracts reusable variables (names, dates, clauses, IDs, etc.).

Converts them into Markdown templates with YAML front-matter metadata.

Enables interactive drafting — automatically filling, questioning, and generating legal drafts.

Optionally uses Exa.ai for web bootstrapping when no suitable local template exists.


## Architecture

![alt text](image-3.png)
Core Flow:

Upload → Extract → Templatize using Gemini.

Save templates with variables into a PostgreSQL database.

Retrieve templates by similarity or classification.

Draft → Q&A → Generate Markdown with variables fully substituted.

(Bonus) Web Bootstrap via Exa.ai when no local match is found.

## Features

- **AI-Powered Template Generation**: Upload a `.docx`, `.pdf`, or `.txt` file to automatically generate a Markdown template with detected variables (`/create-template-from-upload/`).
- **Template Management**: Save, retrieve, and manage templates in the database (`/save-template/`, `/templates/`).
- **Intelligent Template Discovery**: Find relevant local templates based on a natural language query. If no local match is found, it can bootstrap a new template from a web search (`/find-templates`).
- **Interactive Drafting Workflow**:
    - **Prefill**: Automatically fill template variables from an initial user query (`/prefill-variables-from-query`).
    - **Question Generation**: Generates human-friendly questions for any required variables that are still empty (`/generate-questions-for-missing-variables`).
    - **Drafting**: Fills the template with user-provided variables to create a final document draft (`/fill-template`).
- **Document Export**: Export the final drafted Markdown document to `.docx` or `.pdf` format (`/export/`).
- **Asynchronous Document Analysis**:
    - Upload documents for deep analysis in the background (`/upload-document/`).
    - Uses specialized AI agents for classification, summarization, and entity extraction.
    - Check document status and retrieve insights (`/document/{id}/status`, `/document/{id}/insights`).
- **Document Q&A**: Ask specific questions about a processed document and get AI-generated answers (`/document/{id}/query`).
- **Persistent Storage**: Uses Prisma and PostgreSQL to store templates, variables, documents, and generated insights.

Images
![alt text](image.png)

The application is built around two core workflows: **Template-based Drafting** and **Document Analysis**.
## API Endpoints

Here is a summary of the available API endpoints, grouped by functionality:

### Template Management
*   **`POST /create-template-from-upload/`**: Upload a document (`.docx`, `.pdf`, `.txt`) to have an AI agent generate a reusable Markdown template with variables.
*   **`POST /save-template/`**: Save a reviewed and edited Markdown template to the database.
*   **`GET /templates/`**: Retrieve a list of all saved templates, including their variables.

### Drafting Workflow
*   **`POST /find-templates`**: Search for relevant templates using a natural language query. If no local match is found, it attempts to bootstrap a new template from a web search.
*   **`POST /prefill-variables-from-query`**: Takes a `template_id` and a user `query` to intelligently pre-fill template variables.
*   **`POST /generate-questions-for-missing-variables`**: For a given template, it generates user-friendly questions for any required variables that are still empty.
*   **`POST /fill-template`**: Fills a specified template with provided variable values to generate a final document draft in Markdown.

### Document Analysis & Query (Asynchronous)
*   **`POST /upload-document/`**: Upload a document for background analysis. The API returns immediately while agents perform classification, summarization, and entity extraction.
*   **`GET /documents/`**: Retrieve a list of all uploaded documents and their metadata.
*   **`GET /document/{document_id}/status`**: Check the processing status of a specific document (`uploading`, `processing`, `completed`, `failed`).
*   **`GET /document/{document_id}/insights`**: Get the generated insights (summary, entities) for a document that has been successfully processed.
*   **`POST /document/{document_id}/query`**: Ask a specific question about the content of a processed document and receive an AI-generated answer.

### Utilities
*   **`POST /export/`**: Export a generated Markdown draft into a `.docx` or `.pdf` file by providing the content, filename, and desired filetype.

---




### Prerequisites

- Python 3.9+
- A PostgreSQL database (e.g., from Supabase, Neon, or a local instance).
- A Google Gemini API Key.

### 1. Clone the Repository

```bash
git clone 
cd /backend
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```
fastapi
uvicorn[standard]
python-multipart
pypdf2
python-docx
py-markdown-it
prisma
google-generativeai
python-dotenv
```

Then, install them:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the `backend` directory and add the following variables. **Do not commit this file to version control.**

```properties
GEMINI_API_KEY="your_google_gemini_api_key_here"
EVA_API_KEY="your_eva_api_key_here"
DATABASE_URL="your_postgresql_connection_string_here"
```

- `GEMINI_API_KEY`: Your API key from Google AI Studio.
- `DATABASE_URL`:"postgresql://postgres:dvd.supabase.co:5432/postgres""
- `EVA_API_KEY`:"your_eva_api_key_here"


### 5. Initialize Prisma

Generate the Prisma client for Python.

```bash

npx prisma migrate dev 
python -m prisma generate
```

### 6. Run the Application

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.