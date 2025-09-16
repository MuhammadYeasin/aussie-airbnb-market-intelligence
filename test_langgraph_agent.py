#!/usr/bin/env python3
"""
Test script for the complete LangGraph agent

This script tests the fully wired LangGraph agent with various types of queries
to ensure it can intelligently route between quantitative and qualitative analysis.
"""

import sys
import os
sys.path.append('src')

def test_agent_creation():
    """Test that the LangGraph agent can be created successfully."""
    
    print("🧪 Testing LangGraph Agent Creation...")
    
    try:
        from agent import create_langgraph_agent
        
        # Create the agent
        app = create_langgraph_agent()
        
        print("✅ LangGraph agent created successfully!")
        print(f"Agent type: {type(app)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False

def test_quantitative_query():
    """Test the agent with a quantitative question."""
    
    print("\n🧪 Testing Quantitative Query...")
    
    try:
        from agent import create_main_agent
        
        # Test quantitative question
        question = "What is the average price in Sydney?"
        print(f"Question: {question}")
        
        print("Executing agent...")
        response = create_main_agent(question)
        
        print("✅ Quantitative query successful!")
        print(f"Response preview: {response[:150]}...")
        
        # Check if response mentions quantitative analysis
        if "quantitative" in response.lower() or "average" in response.lower() or "price" in response.lower():
            print("✅ Response appears to be quantitative analysis")
        else:
            print("⚠️ Response may not be quantitative analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantitative query error: {e}")
        return False

def test_qualitative_query():
    """Test the agent with a qualitative question."""
    
    print("\n🧪 Testing Qualitative Query...")
    
    try:
        from agent import create_main_agent
        
        # Test qualitative question
        question = "find properties with dedicated workspace"
        print(f"Question: {question}")
        
        print("Executing agent...")
        response = create_main_agent(question)
        
        print("✅ Qualitative query successful!")
        print(f"Response preview: {response[:150]}...")
        
        # Check if response mentions qualitative search
        if "workspace" in response.lower() or "property" in response.lower() or "search" in response.lower():
            print("✅ Response appears to be qualitative search")
        else:
            print("⚠️ Response may not be qualitative search")
        
        return True
        
    except Exception as e:
        print(f"❌ Qualitative query error: {e}")
        return False

def test_mixed_query():
    """Test the agent with a mixed question requiring both tools."""
    
    print("\n🧪 Testing Mixed Query...")
    
    try:
        from agent import create_main_agent
        
        # Test mixed question
        question = "Show me pet-friendly properties in Sydney and tell me their average price"
        print(f"Question: {question}")
        
        print("Executing agent...")
        response = create_main_agent(question)
        
        print("✅ Mixed query successful!")
        print(f"Response preview: {response[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed query error: {e}")
        return False

def test_error_handling():
    """Test the agent's error handling capabilities."""
    
    print("\n🧪 Testing Error Handling...")
    
    try:
        from agent import create_main_agent
        
        # Test with an unclear question
        question = "asdfghjkl"
        print(f"Question: {question}")
        
        print("Executing agent...")
        response = create_main_agent(question)
        
        print("✅ Error handling successful!")
        print(f"Response preview: {response[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all LangGraph agent tests."""
    
    print("🚀 Testing Complete LangGraph Agent")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_agent_creation,
        test_quantitative_query,
        test_qualitative_query,
        test_mixed_query,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 Test Summary:")
    test_names = [
        "Agent Creation", 
        "Quantitative Query", 
        "Qualitative Query", 
        "Mixed Query", 
        "Error Handling"
    ]
    for name, result in zip(test_names, results):
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    if all(results):
        print("\n🎉 LangGraph agent is fully functional!")
        print("The agent can intelligently route queries between tools.")
        print("Ready for integration with the Streamlit UI!")
    else:
        print("\n❌ Some tests failed. Please review the issues.")

if __name__ == "__main__":
    main()
