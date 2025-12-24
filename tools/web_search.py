"""Tavily web search tool integration."""

from tavily import TavilyClient
from langchain_core.documents import Document
from typing import List

from app.config import TAVILY_API_KEY


def get_tavily_client() -> TavilyClient:
    """Get Tavily client instance."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in environment variables")
    return TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> List[Document]:
    """
    Perform web search using Tavily API.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        List of Document objects with web search results
    """
    client = get_tavily_client()
    
    # Add automotive context to query for better results
    enhanced_query = f"car dealership automotive {query}"
    
    try:
        response = client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )
        
        documents = []
        
        # Add the AI-generated answer as first document if available
        if response.get("answer"):
            documents.append(Document(
                page_content=response["answer"],
                metadata={
                    "source": "Tavily AI Summary",
                    "category": "web_search",
                    "url": "AI-generated summary"
                }
            ))
        
        # Add individual search results
        for result in response.get("results", []):
            documents.append(Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("title", "Web Result"),
                    "category": "web_search",
                    "url": result.get("url", "")
                }
            ))
        
        return documents
        
    except Exception as e:
        return [Document(
            page_content=f"Web search failed: {str(e)}",
            metadata={"source": "error", "category": "web_search"}
        )]


def format_web_results(documents: List[Document]) -> str:
    """Format web search results into context string."""
    if not documents:
        return "No web search results found."
    
    parts = []
    for doc in documents:
        url = doc.metadata.get("url", "")
        source = doc.metadata.get("source", "Web")
        url_text = f" ({url})" if url and url != "AI-generated summary" else ""
        parts.append(f"[{source}{url_text}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(parts)
