from typing import List
from langchain_core.documents import Document
from ..config import llm

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

async def _reduce(input: dict) -> str:
    from .prompts import reduce_prompt
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content