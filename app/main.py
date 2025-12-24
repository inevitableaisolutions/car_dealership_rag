"""Main Streamlit application entry point."""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.chat import render_chat, clear_chat
from app.components.data_manager import render_data_manager


# Page configuration
st.set_page_config(
    page_title="Car Dealership Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/car--v1.png", width=80)
        st.title("🚗 Car Dealership")
        st.caption("AI-Powered Assistant")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            options=["💬 Chat", "📁 Data Manager"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick actions
        if page == "💬 Chat":
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat()
                st.rerun()
            
            st.divider()
            
            # Tips
            st.subheader("💡 Try asking:")
            st.caption("• What SUVs do you have under $40,000?")
            st.caption("• Tell me about financing options")
            st.caption("• What's your return policy?")
            st.caption("• Do you have any electric vehicles?")
        
        st.divider()
        
        # Footer
        st.caption("Powered by LangGraph + Gemini")
        st.caption("Vector DB: Weaviate")
    
    # Main content
    if page == "💬 Chat":
        st.markdown("<h1 class='main-header'>🚗 Car Dealership Assistant</h1>", unsafe_allow_html=True)
        st.caption("Ask me anything about our vehicles, financing, warranties, or policies!")
        st.divider()
        render_chat()
    
    elif page == "📁 Data Manager":
        render_data_manager()


if __name__ == "__main__":
    main()
