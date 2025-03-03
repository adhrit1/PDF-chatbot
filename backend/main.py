# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import shutil
from typing import Dict, Any
from pydantic import BaseModel
import uuid
from sqlalchemy.orm import Session

from database import get_db, Document
from rag import DocumentProcessor, RAGService

# Load environment variables
load_dotenv()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

app = FastAPI(title="PDF Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
rag_service = RAGService()

class ChatRequest(BaseModel):
    query: str
    document_id: str

@app.get("/")
async def root():
    return {"message": "Welcome to PDF Chatbot API. Visit /docs for API documentation."}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Generate unique filename to prevent collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{unique_filename}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        index_id = document_processor.process_document(file_path)
        
        # Save document metadata to database
        doc = Document(
            original_filename=file.filename,
            file_path=file_path,
            index_id=index_id
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        return {
            "document_id": doc.id,
            "filename": file.filename,
            "message": "File uploaded and processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    try:
        # Get document from database
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get response using RAG
        response = rag_service.query(request.query, document.index_id)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Route to list all documents
@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
    return [{"id": doc.id, "filename": doc.original_filename, "upload_date": doc.created_at} for doc in documents]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)