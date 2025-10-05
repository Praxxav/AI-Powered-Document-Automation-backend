from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict
import json
import PyPDF2
import docx
import re # Added for robust JSON extraction
import asyncio
import logging
import sys

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import, specialized agents
from agent.Banking import summarizer_agent, entity_extractor_agent, qa_agent
from config import settings


from prisma import Prisma

# --- Database Client ---
# Instantiate the Prisma client
db = Prisma()

async def process_document_in_background(document_id: str, file_path: str, file_extension: str):
    """
    This function runs in the background to process the document.
    It extracts text, generates insights, and updates the database.
    """
    logging.info(f"Starting background processing for document: {document_id}")
    try:
        await db.document.update(where={"id": document_id}, data={"status": "processing"})

        # 1. Extract text from the document
        text_content = extract_text_from_file(file_path, file_extension)
        if not text_content:
            logging.error(f"Could not extract text from document {document_id}")
            await db.document.update(where={"id": document_id}, data={"status": "failed"})
            return

        # 2. Use agents to generate insights asynchronously
        summary_task = summarizer_agent.process(text_content)
        entities_task = entity_extractor_agent.process(text_content)
        summary, entities_raw = await asyncio.gather(summary_task, entities_task)

        logging.info(f"Insights generated for document: {document_id}")
        
        # Robustly extract JSON from the model's raw output
        try:
            # Find the JSON block within the raw string, which might be wrapped in markdown
            json_match = re.search(r"```(json)?\s*(\{.*?\})\s*```", entities_raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(2)
            else:
                json_str = entities_raw # Fallback to assuming the whole string is JSON
            entities = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            entities = {"error": "Failed to parse entities from model output. The format was invalid.", "raw_output": entities_raw}

        # 3. Store the final results in the database
        await db.document.update(
            where={"id": document_id},
            data={
                "status": "completed",
                "insights": json.dumps({"summary": summary, "entities": entities}),
                "fullText": text_content,
            },
        )
        logging.info(f"Successfully processed document: {document_id}")
        logging.info(f"Output for {document_id}: \nSUMMARY: {summary}\nENTITIES: {entities}")

    except Exception as e:
        logging.error(f"Error processing document {document_id}: {e}", exc_info=True)
        await db.document.update(where={"id": document_id}, data={"status": "failed"})

# --- Helper function for text extraction ---
def extract_text_from_file(file_path: str, file_extension: str) -> str:
    text = ""
    if file_extension == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif file_extension == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

app = FastAPI(
    title="Intelligent Document Analysis Agent",
    description="API for AI-powered legal document review and insight generation.",
    version="0.2.0",
)

# Configure CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded documents temporarily
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup():
    try:
        logging.info("Attempting to connect to the database...")
        await db.connect()
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error("--- DATABASE CONNECTION FAILED ---")
        logging.error("Could not connect to the database. Please check the following:")
        logging.error("1. Your `DATABASE_URL` in the .env file is correct.")
        logging.error("2. The database server is running and accessible (e.g., not paused on Supabase).")
        logging.error("3. Your network/firewall is not blocking the connection on port 5432.")
        sys.exit(1) # Exit the application if the database is not available.

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()


# @app.post("/analyze")
# async def analyze_document(file: UploadFile):
#     text = (await file.read()).decode("utf-8")
#     result = analyze_legal_document(text)
#     return JSONResponse(content=result)

@app.post("/upload-document/")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a document for intelligent analysis.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".pdf", ".docx", ".txt"]: # Simplified for now, excluding images
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, DOCX, TXT, JPG, or PNG.")

    # Create a preliminary record in the database
    new_doc = await db.document.create(data={"status": "uploading"})
    document_id = new_doc.id
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}{file_extension}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logging.info(f"File '{file.filename}' uploaded. Saved to: {file_path}")
        # Add the long-running processing task to the background
        background_tasks.add_task(
            process_document_in_background, document_id, file_path, file_extension
        )
        logging.info(f"Scheduled background processing for document_id: {document_id}")
        
        return {"message": "Document upload successful. Processing has started in the background.", "document_id": document_id}
    except Exception as e:
        # If anything fails, update the status in the DB
        await db.document.update(where={"id": document_id}, data={"status": "failed"})
        raise HTTPException(status_code=500, detail=f"Failed to upload or initiate processing: {str(e)}")
@app.get("/documents/")
async def get_all_documents():
    """
    Retrieves all documents with their details.
    """
    try:
        docs = await db.document.find_many()
        if not docs:
            return {"message": "No documents found."}
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.get("/document/{document_id}/status")
async def get_processing_status(document_id: str):
    """
    Checks the processing status of a document.
    """
    doc = await db.document.find_unique(where={"id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "status": doc.status}

@app.get("/document/{document_id}/insights")
async def get_document_insights_endpoint(document_id: str) -> Dict:
    """
    Retrieves the analysis insights for a processed document.
    """
    doc = await db.document.find_unique(where={"id": document_id})
    if not doc or doc.status != "completed":
        raise HTTPException(status_code=404, detail="Insights not found or document not processed.")
    return doc.insights or {}

@app.post("/document/{document_id}/query")
async def query_document_endpoint(document_id: str, question: Dict[str, str]):
    """
    Allows users to ask specific questions about a processed document.
    """
    doc = await db.document.find_unique(where={"id": document_id})
    if not doc or not doc.fullText:
        raise HTTPException(status_code=404, detail="Document not found or not processed.")

    user_question = question.get("question")
    if not user_question:
        raise HTTPException(status_code=400, detail="Question not provided.")
    
    # Use the agent to answer the question based on the stored full text
    answer = await qa_agent.process(user_question, context=doc.fullText)
    return {"document_id": document_id, "question": user_question, "answer": answer}
