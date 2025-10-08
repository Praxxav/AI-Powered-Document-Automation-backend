# Intelligent Document Drafting & Analysis API

This project is a FastAPI backend for an AI-powered platform that transforms unstructured documents into structured, reusable templates and facilitates the drafting of new documents. It uses a suite of specialized AI agents for document analysis, template creation, and interactive drafting workflows.

## Architecture

![alt text](image-3.png)

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

## Architecture

![alt text](image-2.png)
The application is built around two core workflows: **Template-based Drafting** and **Document Analysis**.

### Template & Drafting Workflow
1.  **Generation**: A user uploads a document to `/create-template-from-upload/`. The `Templatizer` agent processes the text and returns a Markdown template with a YAML front-matter defining variables.
2.  **Save**: The user reviews and saves the template via `/save-template/`. The backend parses the YAML and Markdown, storing the template and its variables in the database.
3.  **Drafting**: A user initiates a draft with a query. The system can find a template (`/find-templates`), pre-fill variables (`/prefill-variables-from-query`), ask clarifying questions (`/generate-questions-for-missing-variables`), and finally generate the document (`/fill-template`).

### Document Analysis Workflow (Asynchronous)
1.  **Upload**: A user uploads a document to `/upload-document/`.
2.  **Store & Queue**: The file is saved, a database record is created, and a background task is scheduled.
3.  **Processing**: The background task extracts text, classifies the document's domain (e.g., "Legal"), and routes it to specialized agents for summarization and entity extraction.
4.  **Store Insights**: The results (summary, entities, full text) are saved to the database, and the document status is updated to `completed`.
5.  **Query**: The user can retrieve insights or ask questions about the document via its specific endpoints.

## Setup and Installation

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
- `DATABASE_URL`:"postgresql://postgres:Pranav@940@db.wmtgzowbnjeiyohexqng.supabase.co:5432/postgres""
- `EVA_API_KEY`:"your_eva_api_key_here"


### 5. Initialize Prisma

Generate the Prisma client for Python.

```bash
prisma generate
```

### 6. Run the Application

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.