from typing import TypedDict, List, Annotated
import operator
from langchain_core.documents import Document

class OverallState(TypedDict):
    """
    Main state for the summary generation process.
    Contains all the necessary data for the graph to operate.
    """
    contents: List[str]  # Input documents to summarize
    summaries: Annotated[List[str], operator.add]  # Individual summaries
    collapsed_summaries: List[Document]  # Summaries after collapsing
    final_summary: str  # The final output summary


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    """
    State for individual summary generation nodes.
    Contains the content to be summarized.
    """
    content: str  # The content to be summarized