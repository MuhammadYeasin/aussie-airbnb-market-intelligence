#!/usr/bin/env python3
"""
Test script for the RAG tool functionality

This script demonstrates how to use the create_rag_tool() function.
Note: This will take some time to run as it creates embeddings for all property data.
"""

import sys
import os
sys.path.append('src')

def test_rag_tool():
    """Test the RAG tool creation and usage."""
    
    print("🧪 Testing RAG Tool Creation...")
    
    try:
        from agent import create_rag_tool
        
        print("✅ Successfully imported create_rag_tool")
        
        # Create the RAG tool (this will take time due to embeddings)
        print("Creating RAG tool (this may take a few minutes)...")
        rag_tool = create_rag_tool()
        
        print("✅ RAG tool created successfully!")
        print(f"Tool name: {rag_tool.name}")
        print(f"Tool description length: {len(rag_tool.description)} characters")
        
        # Test queries
        test_queries = [
            "find properties with dedicated workspace",
            "show me pet-friendly listings",
            "properties with ocean views"
        ]
        
        print("\n🔍 Testing RAG tool with sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 50)
            
            try:
                result = rag_tool.func(query)
                print(f"Result preview: {result[:300]}...")
                print("✅ Query successful")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        print("\n🎉 RAG tool testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_tool()
