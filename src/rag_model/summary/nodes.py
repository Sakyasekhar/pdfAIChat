from .prompts import map_prompt, reduce_prompt
from .states import OverallState, SummaryState
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from .utils import length_function,_reduce
from ..config import llm, token_max

# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text. Write a concise summary of the following text."},
            {"role": "user", "content": state["content"]}
        ]
    response = await llm.ainvoke(messages)
    return {"summaries": [response.content]}


async def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}

# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}