import os
from io import BytesIO
import time
import asyncio
#imports
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import  create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
import fitz  # PyMuPDF
from .summary.graph import app as summary_graph
from .summary.states import OverallState


# Load environment variables from .env
load_dotenv()


#vectorDB
from .vector_store import index

#embedding model and Chat model
from .config import embeddings,llm





from .vector_store import batch_embed_queries,async_upsert_batches

async def create_vectorstore(pdf, session_id):
    start_time = time.time()
    
    # Delete existing vectors for this session_id
    index.delete(filter={"session_id": session_id})
    
    # Read the PDF file into memory
    file_bytes = pdf.file.read()
    pdf_buffer = BytesIO(file_bytes)

    # Use PyMuPDF to open the PDF and extract text
    pdf_document = fitz.open(stream=pdf_buffer, filetype="pdf")
    
    # Extract text from all pages
    pdf_text = ""
    for page in pdf_document:
        pdf_text += page.get_text() + "\n" 
    print(f"PDF read and text extracted in {time.time() - start_time:.2f} seconds")
    step1_time = time.time()

    # Create document chunks
    document = Document(page_content=pdf_text, metadata={"session_id": session_id})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents([document])

    # Optimize embedding by using controlled batching
    embeddings_list = await batch_embed_queries(doc_chunks, batch_size=10)  # Smaller batch size improves efficiency
    print(f"Embeddings generated for {len(doc_chunks)} chunks in {time.time() - step1_time:.2f} seconds")
    step2_time = time.time()

    # Create vectors with simple counter since we've deleted previous ones
    vectors = [(f"{session_id}-{i}", embedding, {"text": doc.page_content,"session_id": session_id}) 
              for i, (doc, embedding) in enumerate(zip(doc_chunks, embeddings_list))]

    # Optimize upserts by running them concurrently
    await async_upsert_batches(vectors, batch_size=100)
    print(f"Upserted {len(vectors)} vectors in {time.time() - step2_time:.2f} seconds")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return {"message": "Embeddings stored successfully"}



from .prompts import contextualize_q_prompt,qa_prompt

# ------------------------------------------------------------------
# Helper that turns any LangChain chain into a token generator
# ------------------------------------------------------------------
def make_token_stream(chain, chain_input: dict, chunk_size: int = 1):
    async def _generator():
        buf = ""
        async for chunk in chain.astream(chain_input):
            # LangChain returns dicts like {"answer": "partial text ..."}
            if not chunk.get("answer"):
                continue
            buf += chunk["answer"]
            # Yield fixed-size pieces (1 token, 3 chars, etc.)
            while len(buf) >= chunk_size:
                yield buf[:chunk_size]
                buf = buf[chunk_size:]
        # Flush the remainder
        if buf:
            yield buf
    return _generator


async def is_summary_request(query):
    intent_prompt = PromptTemplate(
        template="""You are a classifier that determines if a user query is asking for a summary.
        Consider the following types of summary requests:
        - Direct requests for summaries
        - Requests to summarize the document
        - Requests for an overview
        - Requests for a brief explanation
        - Requests to explain the main points
        
        Query: {query}
        
        Respond only with "yes" or "no".""",
        input_variables=["query"]
    )

    chain = LLMChain(llm=llm,prompt=intent_prompt)
    result =await chain.arun(query=query)
    result = result.strip().lower()

    return "yes" in result

async def get_summary(session_id):
    results = index.query(
            vector=[0.0]*768, 
            top_k=100,
            include_metadata=True,
            filter={"session_id": session_id}
        )
    matches = results["matches"]

    # Sort by the numeric part of the ID (after the last '-')
    sorted_matches = sorted(matches, key=lambda m: int(m["id"].rsplit("-", 1)[-1]))

    contents = [match["metadata"]["text"]for match in sorted_matches]

    #Initial state for the summary graph
    initial_state: OverallState = {
        "contents":contents,
        "summaries":[],
        "collapsed_summaries":[],
        "final_summary":""
    }

    response = await summary_graph.ainvoke(initial_state)
    return response["final_summary"]

async def get_summary_streaming(session_id):
    results = index.query(
            vector=[0.0]*768, 
            top_k=100,
            include_metadata=True,
            filter={"session_id": session_id}
        )
    matches = results["matches"]

    # Sort by the numeric part of the ID (after the last '-')
    sorted_matches = sorted(matches, key=lambda m: int(m["id"].rsplit("-", 1)[-1]))
    contents = [match["metadata"]["text"] for match in sorted_matches]

    # Combine all content into one document for streaming summary
    combined_content = "\n\n".join(contents)
    
    # Create a streaming summary using the LLM directly
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes documents. Write a comprehensive but concise summary of the following document. Focus on the main points and key insights."},
        {"role": "user", "content": combined_content}
    ]
    
    # Use the streaming version of the LLM
    async def stream_summary():
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    
    return stream_summary

async def query_llm(query, chat_historyjson, session_id):
    chat_history = []

    for turn in chat_historyjson:
        # Safely build chat history without KeyErrors
        if "human" in turn:
            chat_history.append(HumanMessage(content=turn["human"]))
        elif "AI" in turn:
            chat_history.append(AIMessage(content=turn["AI"]))
        # ignore any malformed turn
    
    if await is_summary_request(query):
        return await get_summary_streaming(session_id)
    
    # Create a retriever for querying the Pinecone vector store
    retriever = PineconeVectorStore(index, embeddings).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3,"score_threshold": 0.1, "filter": {"session_id": session_id}}
    )
    

    # Create a history-aware retriever that takes the user's query and chat history, reformulates 
    # the query into a standalone question if necessary, and retrieves relevant document chunks 
    # from the vector store. This ensures the query has enough context from previous interactions.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
   
    # `create_stuff_documents_chain` takes all retrieved documents (from history_aware_retriever)  
    # and feeds them into the LLM to generate a final response based on the given query.
    stuff_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval-augmented generation (RAG) chain that connects the history-aware retriever 
    # with the document combination step. This chain ensures that relevant document chunks are retrieved 
    # and used effectively for generating an accurate answer.
    rag_chain = create_retrieval_chain(history_aware_retriever, stuff_chain)

    # #Process the user's query through the retrieval chain
    # response = rag_chain.invoke({"input": query, "chat_history": chat_history})

    # #return the AI's response to frontend to there we can append it to chat_history
    # return response["answer"]

    token_stream = make_token_stream(
        rag_chain,
        {"input": query, "chat_history": chat_history},
        chunk_size=1,              # 1 character ≈ "typewriter" effect
    )
    return token_stream