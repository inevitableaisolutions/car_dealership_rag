"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END
from graph.state import ConversationState
from graph.nodes import (
    route_query,
    retrieve_documents,
    check_relevance,
    perform_web_search,
    generate_response,
)


def create_rag_workflow():
    """Create and compile the RAG workflow graph."""
    
    # Initialize the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("web_search", perform_web_search)
    workflow.add_node("generate", generate_response)
    
    # Define edges
    workflow.set_entry_point("route")
    
    # Route -> Retrieve (always try local first unless explicitly web)
    workflow.add_edge("route", "retrieve")
    
    # Retrieve -> Check relevance (conditional)
    workflow.add_conditional_edges(
        "retrieve",
        check_relevance,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    
    # Web search -> Generate
    workflow.add_edge("web_search", "generate")
    
    # Generate -> End
    workflow.add_edge("generate", END)
    
    # Compile the graph
    return workflow.compile()


# Create a singleton workflow instance
_workflow = None


def get_workflow():
    """Get or create the workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = create_rag_workflow()
    return _workflow


def run_query(query: str, messages: list = None) -> dict:
    """
    Run a query through the RAG workflow.
    
    Args:
        query: User's question
        messages: Optional conversation history
        
    Returns:
        Dictionary with response, sources, and metadata
    """
    workflow = get_workflow()
    
    # Initialize state
    initial_state = {
        "messages": messages or [],
        "query": query,
        "query_type": "",
        "retrieved_docs": [],
        "context": "",
        "retrieval_confidence": 0.0,
        "used_web_search": False,
        "response": "",
        "sources": []
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    return {
        "response": result["response"],
        "sources": result["sources"],
        "used_web_search": result["used_web_search"],
        "query_type": result["query_type"],
        "messages": result["messages"]
    }
