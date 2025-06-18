import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Load environment variables from .env
load_dotenv()


# Define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GEMINI_API_KEY"))
# Create a ChatOpenAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Constants
token_max = 10000  # Maximum tokens before collapsing summaries