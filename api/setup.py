# setup llm, embeddings and any other things that need to be setup

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from utils.streaming_callback import StreamingCallback

# load api keys
load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# initialize an embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# return an LLM with a handler attached
def get_streaming_llm(publish, stage: str):
    """Return a Gemini LLM with token streaming callbacks."""
    handler = StreamingCallback(publish, stage)
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True,
        callbacks=[handler],
    )