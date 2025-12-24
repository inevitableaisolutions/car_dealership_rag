"""Data manager UI component for document upload and management."""

import streamlit as st
from pathlib import Path

from app.config import (
    COLLECTION_INVENTORY,
    COLLECTION_KNOWLEDGE,
    COLLECTION_POLICIES,
)
from data.loader import load_file
from data.processor import process_documents
from rag.vectorstore import (
    init_collections,
    add_documents,
    get_collection_count,
    delete_collection,
)


CATEGORY_MAP = {
    "Vehicle Inventory": COLLECTION_INVENTORY,
    "FAQs & Knowledge": COLLECTION_KNOWLEDGE,
    "Policies & Terms": COLLECTION_POLICIES,
}


def render_data_manager():
    """Render the data management interface."""
    st.header("📁 Data Manager")
    st.caption("Upload and manage your dealership data")
    
    # Initialize collections button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Initialize Collections", use_container_width=True):
            with st.spinner("Initializing..."):
                init_collections()
            st.success("Collections initialized!")
    
    with col2:
        if st.button("📊 Run Sample Ingestion", use_container_width=True):
            with st.spinner("Ingesting sample data..."):
                try:
                    from scripts.ingest import ingest_sample_data
                    ingest_sample_data()
                    st.success("Sample data ingested!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.divider()
    
    # Collection stats
    st.subheader("📊 Collection Statistics")
    cols = st.columns(3)
    
    for i, (name, collection) in enumerate(CATEGORY_MAP.items()):
        with cols[i]:
            try:
                count = get_collection_count(collection)
                st.metric(name, f"{count} docs")
            except Exception:
                st.metric(name, "N/A")
    
    st.divider()
    
    # File upload section
    st.subheader("📤 Upload Documents")
    
    category = st.selectbox(
        "Select Category",
        options=list(CATEGORY_MAP.keys()),
        help="Choose where to store the uploaded document"
    )
    
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "doc", "json", "csv", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, JSON, CSV, TXT"
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Upload", type="primary", use_container_width=True):
            collection_name = CATEGORY_MAP[category]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_docs = 0
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                try:
                    # Load file
                    file_bytes = file.read()
                    docs = load_file(file_bytes=file_bytes, file_name=file.name)
                    
                    # Process and add
                    processed = process_documents(docs, collection_name, file.name)
                    add_documents(collection_name, processed)
                    total_docs += len(processed)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Uploaded {total_docs} document chunks to {category}")
            st.rerun()
    
    st.divider()
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    
    with st.expander("Delete Collection Data", expanded=False):
        st.warning("This action cannot be undone!")
        
        delete_category = st.selectbox(
            "Select collection to delete",
            options=list(CATEGORY_MAP.keys()),
            key="delete_category"
        )
        
        confirm = st.text_input(
            f"Type '{delete_category}' to confirm deletion",
            key="delete_confirm"
        )
        
        if st.button("🗑️ Delete Collection", type="secondary"):
            if confirm == delete_category:
                collection_name = CATEGORY_MAP[delete_category]
                delete_collection(collection_name)
                st.success(f"Deleted {delete_category} collection")
                st.rerun()
            else:
                st.error("Confirmation text doesn't match")
