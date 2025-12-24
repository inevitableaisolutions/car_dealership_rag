"""Weaviate vector store operations."""

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from langchain_weaviate import WeaviateVectorStore
from typing import Optional
import atexit

from app.config import (
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    USE_EMBEDDED_WEAVIATE,
    COLLECTION_INVENTORY,
    COLLECTION_KNOWLEDGE,
    COLLECTION_POLICIES,
    EMBEDDING_DIMENSION,
)
from rag.embeddings import get_embeddings

# Global client instance
_client: Optional[weaviate.WeaviateClient] = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Get or create Weaviate client instance."""
    global _client
    
    if _client is not None and _client.is_connected():
        return _client
    
    if USE_EMBEDDED_WEAVIATE:
        _client = weaviate.connect_to_embedded()
    else:
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
        )
    
    # Register cleanup
    atexit.register(lambda: _client.close() if _client else None)
    
    return _client


def init_collections():
    """Initialize all required collections if they don't exist."""
    client = get_weaviate_client()
    
    collections = [
        (COLLECTION_INVENTORY, "Car inventory items with vehicle details"),
        (COLLECTION_KNOWLEDGE, "Dealership FAQs and general knowledge"),
        (COLLECTION_POLICIES, "Dealership policies and terms"),
    ]
    
    for name, description in collections:
        if not client.collections.exists(name):
            client.collections.create(
                name=name,
                description=description,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ]
            )


def get_vectorstore(collection_name: str) -> WeaviateVectorStore:
    """Get a LangChain vector store for the given collection."""
    client = get_weaviate_client()
    embeddings = get_embeddings()
    
    return WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="content",
        embedding=embeddings,
    )


def add_documents(collection_name: str, documents: list, source: str = "upload"):
    """Add documents to a collection."""
    vectorstore = get_vectorstore(collection_name)
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = source
        doc.metadata["category"] = collection_name
    
    vectorstore.add_documents(documents)


def delete_collection(collection_name: str):
    """Delete a collection and all its data."""
    client = get_weaviate_client()
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)


def get_collection_count(collection_name: str) -> int:
    """Get the number of documents in a collection."""
    client = get_weaviate_client()
    if not client.collections.exists(collection_name):
        return 0
    collection = client.collections.get(collection_name)
    return collection.aggregate.over_all(total_count=True).total_count
