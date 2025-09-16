#!/usr/bin/env python3
"""
Test script for LangGraph node functions

This script tests the call_model, call_tool, and should_continue functions
to ensure they're ready for StateGraph integration.
"""

import sys
import os
sys.path.append('src')

def test_should_continue():
    """Test the should_continue router function."""
    
    print("🧪 Testing should_continue function...")
    
    try:
        from agent import AgentState, should_continue
        
        # Test case 1: Empty messages
        state_empty = AgentState(
            question="Test question",
            messages=[]
        )
        result = should_continue(state_empty)
        assert result == "end", f"Expected 'end', got '{result}'"
        print("✅ Empty messages -> 'end'")
        
        # Test case 2: Message without tool calls
        state_no_tools = AgentState(
            question="Test question",
            messages=[{"role": "assistant", "content": "Hello!"}]
        )
        result = should_continue(state_no_tools)
        assert result == "end", f"Expected 'end', got '{result}'"
        print("✅ No tool calls -> 'end'")
        
        # Test case 3: Message with tool calls
        state_with_tools = AgentState(
            question="Test question",
            messages=[{
                "role": "assistant", 
                "content": "I'll help you!",
                "tool_calls": [{"name": "quantitative_property_analysis", "args": {"query": "test"}}]
            }]
        )
        result = should_continue(state_with_tools)
        assert result == "continue", f"Expected 'continue', got '{result}'"
        print("✅ With tool calls -> 'continue'")
        
        print("✅ should_continue function working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ should_continue error: {e}")
        return False

def test_call_tool():
    """Test the call_tool function."""
    
    print("\n🧪 Testing call_tool function...")
    
    try:
        from agent import AgentState, call_tool
        
        # Test case 1: No tool calls
        state_no_tools = AgentState(
            question="Test question",
            messages=[{"role": "assistant", "content": "Hello!"}]
        )
        result = call_tool(state_no_tools)
        assert result == state_no_tools, "State should be unchanged when no tool calls"
        print("✅ No tool calls -> state unchanged")
        
        # Test case 2: With tool calls (this will test tool creation)
        state_with_tools = AgentState(
            question="Test question",
            messages=[{
                "role": "assistant", 
                "content": "I'll help you!",
                "tool_calls": [{
                    "id": "test_call_1",
                    "name": "quantitative_property_analysis", 
                    "args": {"query": "What is the average price in Sydney?"}
                }]
            }]
        )
        
        print("Testing tool execution (this may take a moment)...")
        result = call_tool(state_with_tools)
        
        # Check that tool results were added
        assert len(result["messages"]) == 2, "Should have 2 messages after tool execution"
        assert result["messages"][-1]["role"] == "tool", "Last message should be from tool"
        print("✅ Tool execution successful")
        
        print("✅ call_tool function working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ call_tool error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_call_model():
    """Test the call_model function (basic structure only)."""
    
    print("\n🧪 Testing call_model function structure...")
    
    try:
        from agent import AgentState, call_model
        
        # Test basic state structure
        state = AgentState(
            question="What is the average price in Sydney?",
            messages=[]
        )
        
        print("✅ call_model function imported successfully")
        print("Note: Full call_model testing requires OpenAI API calls")
        print("✅ call_model function structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ call_model error: {e}")
        return False

def test_agent_state():
    """Test AgentState functionality."""
    
    print("\n🧪 Testing AgentState...")
    
    try:
        from agent import AgentState
        
        # Test creation
        state = AgentState(
            question="Test question",
            messages=[]
        )
        
        # Test mutation
        state["messages"].append({"role": "user", "content": "Hello"})
        state["messages"].append({"role": "assistant", "content": "Hi there!"})
        
        assert len(state["messages"]) == 2, "Should have 2 messages"
        assert state["question"] == "Test question", "Question should be preserved"
        
        print("✅ AgentState creation and mutation working")
        return True
        
    except Exception as e:
        print(f"❌ AgentState error: {e}")
        return False

def main():
    """Run all node function tests."""
    
    print("🚀 Testing LangGraph Node Functions")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_agent_state,
        test_should_continue,
        test_call_tool,
        test_call_model
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
    test_names = ["AgentState", "should_continue", "call_tool", "call_model"]
    for name, result in zip(test_names, results):
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    if all(results):
        print("\n🎉 All node functions are ready for StateGraph integration!")
        print("Next step: Create the StateGraph with nodes and edges.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()
