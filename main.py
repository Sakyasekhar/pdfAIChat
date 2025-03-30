from fastapi import FastAPI, UploadFile, File
import uuid
from src.rag_model.rag_conversational import create_vectorstore, query_llm
from typing import List,Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.post("/upload-pdf/")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, generate embeddings, and store them in the vector database.
    - session_id: A unique identifier provided by the frontend to track the session.
    - file: The uploaded PDF file.
    """
   

    # Call function to process and store embeddings
    response = await create_vectorstore(file, session_id)

    return {"message": "PDF processed successfully", "session_id": session_id}



class QueryRequest(BaseModel):
    session_id: str
    query: str
    chat_history: List[Dict[str, str]] 

@app.post("/query/")
async def query_pdf(request: QueryRequest):
    """
    Query the LLM using the stored embeddings from a specific session's PDF.
    - session_id: The unique identifier for the uploaded PDF session.
    - query: The user's question.
    - chat_history: A list of previous messages in the conversation for context.
    """
    response = query_llm(request.query, request.chat_history, request.session_id)
    return {"response": response}
