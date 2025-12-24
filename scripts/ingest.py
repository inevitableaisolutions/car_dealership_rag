"""Data ingestion script for populating the vector store."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    DATA_DIR,
    COLLECTION_INVENTORY,
    COLLECTION_KNOWLEDGE,
    COLLECTION_POLICIES,
)
from data.loader import load_json
from data.processor import process_documents
from rag.vectorstore import init_collections, add_documents, get_collection_count


def ingest_sample_data():
    """Ingest sample data into the vector store."""
    print("Initializing collections...")
    init_collections()
    
    # Load inventory
    print("\nLoading inventory data...")
    inventory_path = DATA_DIR / "inventory.json"
    if inventory_path.exists():
        docs = load_json(str(inventory_path))
        processed = process_documents(docs, COLLECTION_INVENTORY, "inventory.json")
        add_documents(COLLECTION_INVENTORY, processed)
        print(f"  Added {len(processed)} inventory documents")
    
    # Load knowledge (FAQs + Policies)
    print("\nLoading knowledge base...")
    knowledge_path = DATA_DIR / "knowledge.json"
    if knowledge_path.exists():
        import json
        with open(knowledge_path, "r") as f:
            data = json.load(f)
        
        # Process FAQs
        faq_docs = []
        for faq in data.get("faqs", []):
            from langchain_core.documents import Document
            faq_docs.append(Document(
                page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
                metadata={"source": "faqs", "type": "faq"}
            ))
        processed_faqs = process_documents(faq_docs, COLLECTION_KNOWLEDGE, "knowledge.json")
        add_documents(COLLECTION_KNOWLEDGE, processed_faqs)
        print(f"  Added {len(processed_faqs)} FAQ documents")
        
        # Process Policies
        policy_docs = []
        for policy in data.get("policies", []):
            from langchain_core.documents import Document
            policy_docs.append(Document(
                page_content=f"{policy['title']}\n\n{policy['content']}",
                metadata={"source": "policies", "type": "policy"}
            ))
        processed_policies = process_documents(policy_docs, COLLECTION_POLICIES, "knowledge.json")
        add_documents(COLLECTION_POLICIES, processed_policies)
        print(f"  Added {len(processed_policies)} policy documents")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Ingestion Complete!")
    print("=" * 50)
    print(f"  Inventory: {get_collection_count(COLLECTION_INVENTORY)} documents")
    print(f"  Knowledge: {get_collection_count(COLLECTION_KNOWLEDGE)} documents")
    print(f"  Policies: {get_collection_count(COLLECTION_POLICIES)} documents")


if __name__ == "__main__":
    ingest_sample_data()
