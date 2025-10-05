# Intelligent Document Analysis Agent

This project is a backend API for an AI-powered document analysis platform. It allows users to upload documents (PDF, DOCX, TXT), which are then processed asynchronously by a team of specialized AI agents. The system first classifies the document's domain (e.g., Banking, Legal) and then routes it to the appropriate agents for summarization, entity extraction, and question-answering.

## Features

- **Document Upload**: Supports PDF, DOCX, and plain text files.
- **Asynchronous Processing**: Uses FastAPI's `BackgroundTasks` to process documents without blocking the API.
- **Agent-based Routing**: A "Router" agent classifies documents to determine their primary domain.
- **Specialized AI Agents**: Separate sets of agents for different domains (Banking, Law) to provide more accurate and context-aware insights.
- **Insight Generation**: Extracts a concise summary and structured entities (JSON) from each document.
- **Interactive Q&A**: Allows users to ask specific questions about a processed document.
- **Persistent Storage**: Uses Prisma and a PostgreSQL database to store document status, text, and generated insights.

## Architecture

The application follows a simple, scalable, agentic architecture:

1.  **Upload**: A user uploads a document via the `/upload-document/` endpoint.
2.  **Store & Queue**: The file is saved locally, a record is created in the database with `uploading` status, and a background task is scheduled.
3.  **Text Extraction**: The background task extracts the full text from the document.
4.  **Classification**: The `DocumentClassifier` agent analyzes the text and determines the domain ("Banking", "Legal", or "Other").
5.  **Specialized Processing**: Based on the classification, the system selects the appropriate set of agents (e.g., `banking_agents` or `law_agents`).
6.  **Insight Generation**: The selected summarizer and entity extractor agents process the text to generate insights.
7.  **Update Database**: The generated summary, entities, and full text are saved to the database, and the document status is updated to `completed`.
8.  **Query**: The user can then query the document via the `/document/{id}/query` endpoint, which uses the domain-specific Q&A agent for answers.

## Setup and Installation

### Prerequisites

- Python 3.9+
- A PostgreSQL database (e.g., from Supabase, Neon, or a local instance).
- A Google Gemini API Key.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Agentic-Law/backend
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

```txt
fastapi
uvicorn[standard]
python-multipart
pypdf2
python-docx
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
DATABASE_URL="your_postgresql_connection_string_here"
```

- `GEMINI_API_KEY`: Your API key from Google AI Studio.
- `DATABASE_URL`:"postgresql://postgres:Pranav@940@db.wmtgzowbnjeiyohexqng.supabase.co:5432/postgres""
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