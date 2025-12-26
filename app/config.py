"""Application configuration and settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for Streamlit secrets (for Streamlit Cloud deployment)
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "sample_data"

# Weaviate settings
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
USE_EMBEDDED_WEAVIATE = not WEAVIATE_URL

# Embedding settings
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768

# LLM settings
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 8192

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
RELEVANCE_THRESHOLD = 0.7

# Collection names
COLLECTION_INVENTORY = "CarInventory"
COLLECTION_KNOWLEDGE = "DealershipKnowledge"
COLLECTION_POLICIES = "DealershipPolicies"
