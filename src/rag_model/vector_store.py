from pinecone import  Pinecone,ServerlessSpec
import time
import asyncio
from .config import embeddings

pc = Pinecone()

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