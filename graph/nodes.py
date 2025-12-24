"""LangGraph nodes for the RAG workflow."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal

from app.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, RELEVANCE_THRESHOLD
from graph.state import ConversationState
from rag.retriever import retrieve_with_scores, format_retrieved_context
from tools.web_search import web_search, format_web_results


def get_llm():
    """Get the Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful car dealership assistant. Your role is to help customers with:
- Finding vehicles in our inventory
- Answering questions about financing, warranties, and trade-ins
- Explaining our policies and procedures
- Providing general automotive information

IMPORTANT RULES:
1. ONLY answer based on the provided context. Do not make up information.
2. If the context doesn't contain enough information, say "I don't have that information in our records."
3. Always cite your sources when providing information.
4. For pricing or availability, remind customers that information may change and to contact us for current details.
5. Be friendly, professional, and helpful.
6. If asked about topics unrelated to car dealerships, politely redirect to vehicle-related questions.

Context from our knowledge base:
{context}

Web search results (if available):
{web_context}
"""


def route_query(state: ConversationState) -> ConversationState:
    """Classify the query to determine routing."""
    query = state["query"].lower()
    
    # Simple keyword-based routing
    if any(word in query for word in ["car", "vehicle", "suv", "sedan", "truck", "price", "mileage", "inventory", "available"]):
        query_type = "inventory"
    elif any(word in query for word in ["financing", "loan", "payment", "warranty", "trade", "faq"]):
        query_type = "knowledge"
    elif any(word in query for word in ["policy", "return", "refund", "terms", "service"]):
        query_type = "policy"
    elif any(word in query for word in ["latest", "news", "current", "2024", "2025", "market", "trend"]):
        query_type = "web"
    else:
        query_type = "general"
    
    return {**state, "query_type": query_type}


def retrieve_documents(state: ConversationState) -> ConversationState:
    """Retrieve documents from vector store."""
    query = state["query"]
    
    # Get documents with scores
    results = retrieve_with_scores(query)
    
    if results:
        # Calculate average confidence
        avg_score = sum(score for _, score in results) / len(results)
        docs = [doc for doc, _ in results]
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "General")
            }
            for doc in docs
        ]
        
        context = format_retrieved_context(docs)
        
        return {
            **state,
            "retrieved_docs": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
            "context": context,
            "retrieval_confidence": avg_score,
            "sources": sources
        }
    
    return {
        **state,
        "retrieved_docs": [],
        "context": "",
        "retrieval_confidence": 0.0,
        "sources": []
    }


def check_relevance(state: ConversationState) -> Literal["generate", "web_search"]:
    """Decide whether to generate response or fall back to web search."""
    # If query type is explicitly web or confidence is low, use web search
    if state["query_type"] == "web":
        return "web_search"
    
    if state["retrieval_confidence"] < RELEVANCE_THRESHOLD and not state["retrieved_docs"]:
        return "web_search"
    
    return "generate"


def perform_web_search(state: ConversationState) -> ConversationState:
    """Perform web search for additional context."""
    query = state["query"]
    
    results = web_search(query)
    web_context = format_web_results(results)
    
    # Add web sources
    web_sources = [
        {
            "content": doc.page_content[:200] + "...",
            "source": doc.metadata.get("source", "Web"),
            "url": doc.metadata.get("url", "")
        }
        for doc in results
    ]
    
    return {
        **state,
        "context": state.get("context", "") + "\n\n" + web_context if state.get("context") else web_context,
        "used_web_search": True,
        "sources": state.get("sources", []) + web_sources
    }


def generate_response(state: ConversationState) -> ConversationState:
    """Generate response using the LLM."""
    llm = get_llm()
    
    context = state.get("context", "No relevant information found.")
    web_context = "Web search was used for this query." if state.get("used_web_search") else "No web search performed."
    
    # Build the prompt
    system_message = SYSTEM_PROMPT.format(
        context=context,
        web_context=web_context
    )
    
    # Get conversation history
    messages = state.get("messages", [])
    
    # Create prompt with system context
    full_prompt = f"{system_message}\n\nUser question: {state['query']}"
    
    # Generate response
    response = llm.invoke(full_prompt)
    
    return {
        **state,
        "response": response.content,
        "messages": messages + [
            HumanMessage(content=state["query"]),
            AIMessage(content=response.content)
        ]
    }
