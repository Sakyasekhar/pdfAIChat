from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from src.rag_model.rag_conversational import create_vectorstore, query_llm
from typing import List,Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

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
    stream_fn = await query_llm(request.query,
                                request.chat_history,
                                request.session_id)

    async def event_stream():
        async for token in stream_fn():          # consume generator
            # Send { "chunk": "x" }\n
            yield json.dumps({"chunk": token}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/stream-test")
async def test_stream():
    async def generate():
        for i in range(10):
            await asyncio.sleep(1)  # 1 second delay
            yield json.dumps({"chunk": f"Chunk {i} "}) + "\n"
    
    return StreamingResponse(           
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )