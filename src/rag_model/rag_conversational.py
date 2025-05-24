import os
from io import BytesIO
import time
#imports
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import  create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from pinecone import  Pinecone,ServerlessSpec
import asyncio
import fitz  # PyMuPDF
from .summary.graph import app as summary_graph
from .summary.states import OverallState


# Load environment variables from .env
load_dotenv()


#vectorDB
pc = Pinecone()

# Define the embedding model
from .config import embeddings
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a ChatOpenAI model
from .config import llm

index_name = "pdfchatbot"
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

# Ensure the index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
index = pc.Index(index_name)
time.sleep(1)

index = pc.Index(index_name)



async def batch_embed_queries(chunks, batch_size=10):
    """Batch embed multiple text chunks while respecting API limits."""
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(None, embeddings.embed_documents, [doc.page_content for doc in chunks[i:i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ]

    embeddings_list_batches = await asyncio.gather(*tasks)
    embeddings_list = [embedding for batch in embeddings_list_batches for embedding in batch]
    
    
    return embeddings_list

def chunker(seq, batch_size):
    return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

async def async_upsert_batches(vectors, batch_size=200):
    chunks = list(chunker(vectors, batch_size))
    async def upsert_chunk(chunk):
        return await asyncio.to_thread(index.upsert, vectors=chunk, async_req=True)

    tasks = [upsert_chunk(chunk) for chunk in chunks]
    await asyncio.gather(*tasks)

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



#Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)




#qa_system prompt (context consists of retrieved document chunks from the vector database, 
# which are used along with the query to generate an answer)
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

#Create a prompt template for answering questions  (input nothing but query)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])





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

    #Run the summary graph
    summary_graph_state =await summary_graph.ainvoke(initial_state)

    return summary_graph_state["final_summary"]


async def query_llm(query, chat_historyjson, session_id):
    chat_history = []

    for dir in chat_historyjson:
        if "human" in dir:
            chat_history.append(HumanMessage(content=dir["human"]))
        else:
            chat_history.append(SystemMessage(content=dir["AI"]))
    
    if await is_summary_request(query):
        return await get_summary(session_id)
    
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

    #Process the user's query through the retrieval chain
    response = rag_chain.invoke({"input": query, "chat_history": chat_history})

    #return the AI's response to frontend to there we can append it to chat_history
    return response["answer"]