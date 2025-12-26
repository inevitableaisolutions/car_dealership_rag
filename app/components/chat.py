"""Chat UI component for Streamlit."""

import streamlit as st


def init_chat_state():
    """Initialize chat session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


def render_chat():
    """Render the chat interface."""
    init_chat_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"**{i}. {source.get('source', 'Unknown')}**")
                        st.caption(source.get("content", "")[:150] + "...")
                        if source.get("url"):
                            st.caption(f"[Link]({source['url']})")
            
            # Show if web search was used
            if message.get("used_web_search"):
                st.caption("🌐 *Web search was used for this response*")
    
    # Chat input
    if prompt := st.chat_input("Ask about our vehicles, financing, or policies..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Import here to avoid circular imports
                    from graph.workflow import run_query
                    
                    result = run_query(
                        query=prompt,
                        messages=st.session_state.conversation_history
                    )
                    
                    response = result.get("response", "No response generated")
                    sources = result.get("sources", [])
                    used_web = result.get("used_web_search", False)
                    
                    # Update conversation history
                    st.session_state.conversation_history = result.get("messages", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.caption(f"**{i}. {source.get('source', 'Unknown')}**")
                                st.caption(source.get("content", "")[:150] + "...")
                                if source.get("url"):
                                    st.caption(f"[Link]({source['url']})")
                    
                    # Show web search indicator
                    if used_web:
                        st.caption("🌐 *Web search was used for this response*")
                    
                    # Save to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "used_web_search": used_web
                    })
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.code(traceback.format_exc())
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []
    st.session_state.conversation_history = []
