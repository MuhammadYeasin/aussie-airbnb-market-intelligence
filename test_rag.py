#!/usr/bin/env python3
"""
Test script for RAG functionality

This script tests the RAG system without running the full notebook.
"""

import pandas as pd
import os
from dotenv import load_dotenv

def test_rag_setup():
    """Test if RAG components can be imported and basic functionality works."""
    
    print("🧪 Testing RAG Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found in environment")
        return False
    
    print("✅ OpenAI API key found")
    
    # Test imports
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        print("✅ All LangChain imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data loading
    try:
        data_path = os.path.join("data", "processed", "airbnb_unified_data.csv")
        if not os.path.exists(data_path):
            print("❌ Processed data file not found")
            return False
        
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df):,} listings")
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    # Test text splitting
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create a sample document
        sample_text = "This is a sample property description with amenities and features."
        sample_doc = Document(page_content=sample_text, metadata={"test": True})
        
        chunks = text_splitter.split_documents([sample_doc])
        print(f"✅ Text splitting works: {len(chunks)} chunks created")
    except Exception as e:
        print(f"❌ Text splitting error: {e}")
        return False
    
    # Test embeddings (without actually creating them to save API costs)
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("✅ OpenAI embeddings initialized")
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False
    
    print("\n🎉 All RAG components are ready!")
    print("You can now run the notebook: jupyter notebook notebooks/02_rag_development.ipynb")
    return True

if __name__ == "__main__":
    test_rag_setup()
