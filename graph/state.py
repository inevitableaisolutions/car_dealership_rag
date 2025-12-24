"""Conversation state definition for LangGraph."""

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    """State maintained throughout the conversation."""
    
    # Chat messages with automatic merging
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current user query
    query: str
    
    # Query classification
    query_type: str  # 'inventory', 'knowledge', 'policy', 'web', 'general'
    
    # Retrieved context
    retrieved_docs: List[dict]
    context: str
    
    # Confidence score from retrieval (0-1)
    retrieval_confidence: float
    
    # Whether web search was used
    used_web_search: bool
    
    # Final response
    response: str
    
    # Sources for citations
    sources: List[dict]
