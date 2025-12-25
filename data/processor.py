"""Text processing and chunking utilities."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Get the text splitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects with preserved metadata
    """
    splitter = get_text_splitter()
    chunked_docs = []
    
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            ))
    
    return chunked_docs


def process_documents(documents: List[Document], category: str, source_name: str) -> List[Document]:
    """
    Process documents for ingestion: chunk and add metadata.
    
    Args:
        documents: Raw documents from loader
        category: Category name (inventory, knowledge, policies)
        source_name: Name of the source file
        
    Returns:
        Processed and chunked documents ready for embedding
    """
    # Add category and source to metadata
    for doc in documents:
        doc.metadata["category"] = category
        doc.metadata["source"] = source_name
    
    # Chunk the documents
    chunked = chunk_documents(documents)
    
    return chunked
