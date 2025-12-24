"""Google embeddings integration using text-embedding-004."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get the Google embeddings instance."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in environment variables")
    
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
