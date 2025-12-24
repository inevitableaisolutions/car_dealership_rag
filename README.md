# Car Dealership RAG Application

AI-powered assistant for car dealerships using LangGraph, LangChain, Google Gemini, and Weaviate.

## Features

- 💬 **Conversational AI**: Chat with customers about vehicles, financing, and policies
- 🔍 **RAG Pipeline**: Retrieves relevant information from your knowledge base
- 🌐 **Web Search**: Falls back to web search for real-time information
- 📁 **Document Upload**: Support for PDF, DOCX, JSON, CSV, TXT files
- 🧠 **Anti-Hallucination**: Uses Gemini 2.0 Flash Thinking for grounded responses

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/tharunkshathriya/Desktop/RAG
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `GOOGLE_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com)
- `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com) (optional, for web search)

### 3. Ingest Sample Data

```bash
python scripts/ingest.py
```

### 4. Run the App

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
RAG/
├── app/              # Streamlit UI
├── rag/              # RAG components (embeddings, vectorstore, retriever)
├── graph/            # LangGraph workflow
├── tools/            # Web search integration
├── data/             # Document loaders and sample data
└── scripts/          # CLI utilities
```

## Usage

### Chat Tab
Ask questions like:
- "What SUVs do you have under $40,000?"
- "Tell me about financing options"
- "What's your return policy?"

### Data Manager Tab
- Upload your own documents (PDF, DOCX, JSON, CSV, TXT)
- View collection statistics
- Manage your knowledge base

## Tech Stack

- **LangGraph**: Conversation workflow orchestration
- **LangChain**: RAG pipeline components
- **Google Gemini 2.0 Flash Thinking**: LLM with reasoning
- **Weaviate**: Vector database (embedded mode)
- **Tavily**: Web search fallback
- **Streamlit**: Web UI
