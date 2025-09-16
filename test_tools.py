#!/usr/bin/env python3
"""
Test script for both pandas and RAG tools

This script tests the create_pandas_tool() and create_rag_tool() functions
to ensure they're ready for LangGraph integration.
"""

import sys
import os
sys.path.append('src')

def test_pandas_tool():
    """Test the pandas tool creation and functionality."""
    
    print("🧪 Testing Pandas Tool...")
    
    try:
        from agent import create_pandas_tool
        
        # Create the pandas tool
        pandas_tool = create_pandas_tool()
        
        print("✅ Pandas tool created successfully!")
        print(f"Tool name: {pandas_tool.name}")
        print(f"Description length: {len(pandas_tool.description)} characters")
        
        # Test with a simple query
        test_query = "What is the average price in Sydney?"
        result = pandas_tool.func(test_query)
        
        print(f"✅ Pandas tool test successful!")
        print(f"Query: {test_query}")
        print(f"Result preview: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Pandas tool error: {e}")
        return False

def test_rag_tool():
    """Test the RAG tool creation (without full execution)."""
    
    print("\n🧪 Testing RAG Tool Import...")
    
    try:
        from agent import create_rag_tool
        
        print("✅ RAG tool function imported successfully!")
        print("Note: Full RAG tool creation will take time due to embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG tool import error: {e}")
        return False

def test_langgraph_imports():
    """Test LangGraph imports."""
    
    print("\n🧪 Testing LangGraph Imports...")
    
    try:
        from typing import TypedDict
        from langgraph.graph import StateGraph
        
        print("✅ TypedDict imported successfully!")
        print("✅ StateGraph imported successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ LangGraph import error: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Testing Multi-Tool Setup for LangGraph Integration")
    print("=" * 60)
    
    # Test imports
    langgraph_ok = test_langgraph_imports()
    
    # Test pandas tool
    pandas_ok = test_pandas_tool()
    
    # Test RAG tool import
    rag_ok = test_rag_tool()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"LangGraph imports: {'✅' if langgraph_ok else '❌'}")
    print(f"Pandas tool: {'✅' if pandas_ok else '❌'}")
    print(f"RAG tool import: {'✅' if rag_ok else '❌'}")
    
    if all([langgraph_ok, pandas_ok, rag_ok]):
        print("\n🎉 All tools are ready for LangGraph integration!")
        print("You can now proceed with creating the StateGraph orchestrator.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()
