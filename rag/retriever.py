"""Hybrid retriever with reranking for RAG."""

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import List

from app.config import (
    TOP_K_RESULTS,
    RELEVANCE_THRESHOLD,
    COLLECTION_INVENTORY,
    COLLECTION_KNOWLEDGE,
    COLLECTION_POLICIES,
)
from rag.vectorstore import get_vectorstore


def get_retriever(collection_name: str = None):
    """
    Get a retriever for the specified collection or all collections.
    
    Args:
        collection_name: Specific collection to search, or None for all
    """
    if collection_name:
        vectorstore = get_vectorstore(collection_name)
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": TOP_K_RESULTS,
                "score_threshold": RELEVANCE_THRESHOLD
            }
        )
    
    # Create ensemble retriever across all collections
    retrievers = []
    for name in [COLLECTION_INVENTORY, COLLECTION_KNOWLEDGE, COLLECTION_POLICIES]:
        try:
            vs = get_vectorstore(name)
            retrievers.append(
                vs.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": TOP_K_RESULTS}
                )
            )
        except Exception:
            continue
    
    if not retrievers:
        raise ValueError("No collections available for retrieval")
    
    # Equal weights across all collections
    weights = [1.0 / len(retrievers)] * len(retrievers)
    
    return EnsembleRetriever(
        retrievers=retrievers,
        weights=weights
    )


def retrieve_with_scores(query: str, collection_name: str = None) -> List[tuple]:
    """
    Retrieve documents with relevance scores.
    
    Returns:
        List of (Document, score) tuples
    """
    collections = [collection_name] if collection_name else [
        COLLECTION_INVENTORY, COLLECTION_KNOWLEDGE, COLLECTION_POLICIES
    ]
    
    results = []
    for name in collections:
        try:
            vs = get_vectorstore(name)
            docs_with_scores = vs.similarity_search_with_score(query, k=TOP_K_RESULTS)
            results.extend(docs_with_scores)
        except Exception:
            continue
    
    # Sort by score (higher is better) and return top k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:TOP_K_RESULTS]


def format_retrieved_context(documents: List[Document]) -> str:
    """Format retrieved documents into context string for LLM."""
    if not documents:
        return "No relevant information found in the knowledge base."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        category = doc.metadata.get("category", "General")
        context_parts.append(
            f"[Source {i}: {category} - {source}]\n{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(context_parts)
