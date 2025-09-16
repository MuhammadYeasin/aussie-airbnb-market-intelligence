"""
Core Agent Logic for Aussie Airbnb Market Intelligence Agent

This module contains the main agent function that creates and manages
the pandas dataframe agent for analyzing Airbnb data.
"""

import pandas as pd
import os
from typing import Optional, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    """
    State definition for the LangGraph agent.
    
    This defines the structure of the state that flows through the graph,
    containing the user's question and the conversation messages.
    """
    question: str  # The user's question/query
    messages: list  # List of messages in the conversation


def call_model(state: AgentState) -> AgentState:
    """
    Node function that invokes the ChatOpenAI model with the current state.
    
    This node processes the user's question and generates a response,
    potentially including tool calls for quantitative or qualitative analysis.
    
    Args:
        state (AgentState): The current state containing question and messages
        
    Returns:
        AgentState: Updated state with model response added to messages
    """
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )
    
    # Create system message with tool descriptions
    system_message = """You are an Airbnb Market Intelligence Agent. You have access to two specialized tools:

1. quantitative_property_analysis: Use this for statistical analysis, prices, counts, averages, and numerical metrics. Examples:
   - "What is the average price in Sydney?"
   - "How many listings have more than 3 bedrooms?"
   - "Which city has the most expensive listings?"

2. qualitative_property_search: Use this for searching property descriptions, amenities, vibes, and qualitative features. Examples:
   - "find properties with dedicated workspace"
   - "show me pet-friendly listings"
   - "properties with ocean views"

Analyze the user's question and determine which tool(s) to use. If the question involves numbers, statistics, or comparisons, use quantitative_property_analysis. If it involves amenities, descriptions, or qualitative features, use qualitative_property_search. You can use both tools if the question requires both types of analysis.

Always provide helpful, detailed responses based on the tool results."""
    
    # Prepare messages for the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["question"]}
    ]
    
    # Add conversation history if available
    if state["messages"]:
        for msg in state["messages"]:
            if isinstance(msg, dict) and "role" in msg:
                messages.append(msg)
    
    # Get response from the model
    response = llm.invoke(messages)
    
    # Update state with the model's response
    updated_messages = state["messages"].copy()
    updated_messages.append({"role": "assistant", "content": response.content})
    
    return {
        "question": state["question"],
        "messages": updated_messages
    }


def call_tool(state: AgentState) -> AgentState:
    """
    Node function that executes a tool based on the last message.
    
    This node checks the last message for tool calls and executes them,
    then adds the tool results to the conversation.
    
    Args:
        state (AgentState): The current state containing question and messages
        
    Returns:
        AgentState: Updated state with tool results added to messages
    """
    # Get the last message
    last_message = state["messages"][-1]
    
    # Check if the message contains tool calls
    if "tool_calls" in last_message and last_message["tool_calls"]:
        tool_results = []
        
        for tool_call in last_message["tool_calls"]:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            try:
                # Execute the appropriate tool
                if tool_name == "quantitative_property_analysis":
                    pandas_tool = create_pandas_tool()
                    result = pandas_tool.func(tool_args["query"])
                elif tool_name == "qualitative_property_search":
                    rag_tool = create_rag_tool()
                    result = rag_tool.func(tool_args["query"])
                else:
                    result = f"Unknown tool: {tool_name}"
                
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "result": result
                })
                
            except Exception as e:
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "result": f"Error executing tool: {str(e)}"
                })
        
        # Add tool results to messages
        updated_messages = state["messages"].copy()
        updated_messages.append({
            "role": "tool",
            "tool_results": tool_results
        })
        
        return {
            "question": state["question"],
            "messages": updated_messages
        }
    
    # No tool calls found, return state unchanged
    return state


def should_continue(state: AgentState) -> str:
    """
    Router function that acts as a conditional edge, checking the last message for tool calls.
    
    This function determines whether to continue to the tool execution node
    or end the conversation based on the presence of tool calls.
    
    Args:
        state (AgentState): The current state containing question and messages
        
    Returns:
        str: "continue" if tool calls are present, "end" otherwise
    """
    # Check if there are any messages
    if not state["messages"]:
        return "end"
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # Check if the message contains tool calls
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "continue"
    else:
        return "end"


def create_langgraph_agent():
    """
    Create and compile the LangGraph agent with multi-tool orchestration.
    
    This function wires together all the components:
    1. Creates StateGraph with AgentState
    2. Adds nodes (call_model, call_tool)
    3. Sets entry point
    4. Adds conditional edges
    5. Compiles the graph
    
    Returns:
        CompiledStateGraph: The compiled LangGraph agent ready for execution
    """
    # Create the StateGraph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)
    
    # Set the entry point
    workflow.set_entry_point("call_model")
    
    # Add conditional edge from call_model
    # This routes to call_tool if should_continue returns "continue"
    # or to END if should_continue returns "end"
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "call_tool",
            "end": "__end__"
        }
    )
    
    # Add edge from call_tool back to call_model
    # This allows the model to process tool results and generate final response
    workflow.add_edge("call_tool", "call_model")
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def analyze_dataframe(query: str, df: pd.DataFrame) -> str:
    """
    Analyze the Airbnb dataframe based on the user's query.
    
    Args:
        query (str): The user's question
        df (pd.DataFrame): The Airbnb data
        
    Returns:
        str: Analysis results
    """
    try:
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Basic data analysis based on common questions
        if "average price" in query_lower:
            if "sydney" in query_lower:
                avg_price = df[df['city'] == 'Sydney']['price'].mean()
                return f"The average price of Airbnb listings in Sydney is ${avg_price:.2f}"
            elif "melbourne" in query_lower:
                avg_price = df[df['city'] == 'Melbourne']['price'].mean()
                return f"The average price of Airbnb listings in Melbourne is ${avg_price:.2f}"
            elif "brisbane" in query_lower:
                avg_price = df[df['city'] == 'Brisbane']['price'].mean()
                return f"The average price of Airbnb listings in Brisbane is ${avg_price:.2f}"
            else:
                avg_price = df['price'].mean()
                return f"The average price of Airbnb listings across all cities is ${avg_price:.2f}"
        
        elif "most expensive" in query_lower or "expensive" in query_lower:
            city_avg = df.groupby('city')['price'].mean().sort_values(ascending=False)
            most_expensive = city_avg.index[0]
            price = city_avg.iloc[0]
            return f"{most_expensive} has the most expensive listings with an average price of ${price:.2f}"
        
        elif "bedrooms" in query_lower:
            if "more than" in query_lower:
                # Extract number from query
                import re
                numbers = re.findall(r'\d+', query)
                if numbers:
                    threshold = int(numbers[0])
                    count = len(df[df['bedrooms'] > threshold])
                    return f"There are {count} listings with more than {threshold} bedrooms"
            else:
                avg_bedrooms = df['bedrooms'].mean()
                return f"The average number of bedrooms per listing is {avg_bedrooms:.2f}"
        
        elif "beds" in query_lower:
            avg_beds = df['beds'].mean()
            return f"The average number of beds per listing is {avg_beds:.2f}"
        
        elif "distribution" in query_lower and "price" in query_lower:
            city_stats = df.groupby('city')['price'].agg(['mean', 'median', 'count'])
            result = "Price distribution across cities:\n"
            for city in city_stats.index:
                mean_price = city_stats.loc[city, 'mean']
                median_price = city_stats.loc[city, 'median']
                count = city_stats.loc[city, 'count']
                result += f"- {city}: Mean ${mean_price:.2f}, Median ${median_price:.2f} ({count} listings)\n"
            return result
        
        elif "neighborhood" in query_lower or "neighbourhood" in query_lower:
            if "melbourne" in query_lower:
                melb_data = df[df['city'] == 'Melbourne']
                top_neighborhoods = melb_data.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(5)
                result = "Top 5 neighborhoods by average price in Melbourne:\n"
                for i, (neighborhood, price) in enumerate(top_neighborhoods.items(), 1):
                    result += f"{i}. {neighborhood}: ${price:.2f}\n"
                return result
            else:
                # General neighborhood analysis
                top_neighborhoods = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=False).head(5)
                result = "Top 5 neighborhoods by average price:\n"
                for i, (neighborhood, price) in enumerate(top_neighborhoods.items(), 1):
                    result += f"{i}. {neighborhood}: ${price:.2f}\n"
                return result
        
        else:
            # General statistics
            total_listings = len(df)
            avg_price = df['price'].mean()
            median_price = df['price'].median()
            city_counts = df['city'].value_counts()
            
            result = f"""General Airbnb Market Statistics:
            
Total listings: {total_listings:,}
Average price: ${avg_price:.2f}
Median price: ${median_price:.2f}

Listings by city:
"""
            for city, count in city_counts.items():
                result += f"- {city}: {count:,} listings\n"
            
            return result
            
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


def create_main_agent(user_question: str) -> str:
    """
    Create and invoke the main LangGraph agent for Airbnb data analysis.
    
    This function creates a multi-tool LangGraph agent that can intelligently
    route queries between quantitative analysis and qualitative property search.
    
    Args:
        user_question (str): The user's question about the Airbnb data
        
    Returns:
        str: The agent's response to the user's question
        
    Raises:
        FileNotFoundError: If the processed data file is not found
        ValueError: If the OpenAI API key is not found in environment variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("""
OPENAI_API_KEY not found in environment variables.

To fix this:
1. Create a .env file in the project root directory
2. Add your OpenAI API key: OPENAI_API_KEY=your_actual_api_key_here
3. Make sure the .env file is in the same directory as this script

Example .env file content:
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

You can copy the config_template.txt file to .env and fill in your key.
        """)
    
    # Define the path to the processed data file
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "processed", "airbnb_unified_data.csv")
    
    # Check if the processed data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
    
    try:
        # Create the LangGraph agent
        app = create_langgraph_agent()
        
        # Create initial state
        initial_state = AgentState(
            question=user_question,
            messages=[]
        )
        
        # Invoke the graph
        result = app.invoke(initial_state)
        
        # Extract the final response from the last assistant message
        final_response = ""
        for message in reversed(result["messages"]):
            if message.get("role") == "assistant" and "content" in message:
                final_response = message["content"]
                break
        
        # If no assistant message found, return a fallback
        if not final_response:
            final_response = "I apologize, but I wasn't able to generate a response. Please try rephrasing your question."
        
        return final_response
        
    except Exception as e:
        # Fallback to direct analysis if LangGraph fails
        df = pd.read_csv(data_path)
        return analyze_dataframe(user_question, df)


def create_pandas_tool() -> Tool:
    """
    Create a pandas analysis tool for quantitative property analysis.
    
    This function creates a LangChain Tool that performs statistical analysis
    on Airbnb data including prices, counts, averages, and other numerical metrics.
    
    Returns:
        Tool: A LangChain tool for quantitative property analysis
        
    Raises:
        FileNotFoundError: If the processed data file is not found
    """
    # Define path to processed data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "processed", "airbnb_unified_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
    
    # Load Airbnb data
    df = pd.read_csv(data_path)
    
    def pandas_analysis(query: str) -> str:
        """
        Perform quantitative analysis on Airbnb data.
        
        Args:
            query (str): The analysis query for statistics, prices, and counts
            
        Returns:
            str: Formatted response with quantitative analysis results
        """
        try:
            # Use the existing analyze_dataframe function
            result = analyze_dataframe(query, df)
            
            # Format the response for the tool
            response = f"Quantitative Analysis Results:\n\n"
            response += f"Query: {query}\n\n"
            response += f"Analysis: {result}\n\n"
            
            # Add some additional context about the dataset
            response += f"Dataset Info:\n"
            response += f"- Total listings: {len(df):,}\n"
            response += f"- Cities: {', '.join([str(city) for city in df['city'].unique()])}\n"
            response += f"- Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}\n"
            
            return response
            
        except Exception as e:
            return f"Error performing quantitative analysis: {str(e)}"
    
    # Create and return the LangChain Tool
    pandas_tool = Tool(
        name="quantitative_property_analysis",
        description="""Perform quantitative analysis on Airbnb property data including statistics, prices, counts, and numerical metrics.
        Use this tool when users ask about:
        - Statistical analysis (averages, medians, counts, distributions)
        - Price analysis and comparisons
        - Numerical aggregations and summaries
        - Data counts and percentages
        - City-wise comparisons
        - Property type statistics
        - Bedroom/bed counts and distributions
        
        Examples of good queries:
        - "What is the average price in Sydney?"
        - "How many listings have more than 3 bedrooms?"
        - "Which city has the most expensive listings?"
        - "What is the price distribution across all cities?"
        - "Show me statistics by property type"
        - "Compare average prices between cities"
        """,
        func=pandas_analysis
    )
    
    return pandas_tool


def create_rag_tool() -> Tool:
    """
    Create a RAG (Retrieval-Augmented Generation) tool for qualitative property search.
    
    This function encapsulates the entire RAG pipeline:
    1. Loads Airbnb data and creates text documents
    2. Splits documents into chunks
    3. Creates OpenAI embeddings
    4. Builds an in-memory Chroma vector store
    5. Returns a LangChain Tool for semantic property search
    
    Returns:
        Tool: A LangChain tool for qualitative property search
        
    Raises:
        FileNotFoundError: If the processed data file is not found
        ValueError: If the OpenAI API key is not found
    """
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Define path to processed data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "processed", "airbnb_unified_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")
    
    # Load Airbnb data
    df = pd.read_csv(data_path)
    
    # Create text documents from property data
    documents = []
    for idx, row in df.iterrows():
        # Combine text fields into a single document
        text_parts = []
        
        # Add property name
        if pd.notna(row['name']):
            text_parts.append(f"Property Name: {row['name']}")
        
        # Add description
        if pd.notna(row['description']):
            text_parts.append(f"Description: {row['description']}")
        
        # Add neighborhood overview
        if pd.notna(row['neighborhood_overview']):
            text_parts.append(f"Neighborhood: {row['neighborhood_overview']}")
        
        # Add metadata
        metadata = {
            'listing_id': row['id'],
            'city': row['city'],
            'price': row['price'],
            'property_type': row['property_type'],
            'room_type': row['room_type'],
            'bedrooms': row['bedrooms'],
            'beds': row['beds']
        }
        
        # Create document if we have text content
        if text_parts:
            combined_text = "\n\n".join(text_parts)
            doc = Document(
                page_content=combined_text,
                metadata=metadata
            )
            documents.append(doc)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    text_chunks = text_splitter.split_documents(documents)
    
    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )
    
    # Create in-memory Chroma vector store
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    
    # Create RetrievalQA chain
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        return_source_documents=True
    )
    
    def rag_search(query: str) -> str:
        """
        Perform semantic search on property descriptions.
        
        Args:
            query (str): The search query for qualitative property features
            
        Returns:
            str: Formatted response with property recommendations
        """
        try:
            # Query the RAG system
            result = qa_chain.invoke({"query": query})
            
            # Format the response
            response = f"Search Results for: '{query}'\n\n"
            response += f"Answer: {result['result']}\n\n"
            
            # Add source information
            response += "Recommended Properties:\n"
            for i, doc in enumerate(result['source_documents'][:3], 1):
                response += f"\n{i}. Property Details:\n"
                response += f"   Listing ID: {doc.metadata.get('listing_id', 'N/A')}\n"
                response += f"   City: {doc.metadata.get('city', 'N/A')}\n"
                response += f"   Price: ${doc.metadata.get('price', 'N/A')}\n"
                response += f"   Type: {doc.metadata.get('property_type', 'N/A')} - {doc.metadata.get('room_type', 'N/A')}\n"
                response += f"   Bedrooms: {doc.metadata.get('bedrooms', 'N/A')}, Beds: {doc.metadata.get('beds', 'N/A')}\n"
                response += f"   Description: {doc.page_content[:200]}...\n"
            
            return response
            
        except Exception as e:
            return f"Error performing semantic search: {str(e)}"
    
    # Create and return the LangChain Tool
    rag_tool = Tool(
        name="qualitative_property_search",
        description="""Search for Airbnb properties based on qualitative features, amenities, vibes, and descriptions. 
        Use this tool when users ask about:
        - Specific amenities (workspace, pool, pet-friendly, etc.)
        - Property vibes and atmosphere
        - Location features (ocean views, beach access, etc.)
        - Neighborhood characteristics
        - Property descriptions and qualitative features
        
        Examples of good queries:
        - "find properties with dedicated workspace"
        - "show me pet-friendly listings"
        - "properties with ocean views"
        - "places near public transportation"
        - "modern apartments with good amenities"
        """,
        func=rag_search
    )
    
    return rag_tool


if __name__ == "__main__":
    # Example usage
    sample_question = "What is the average price of Airbnb listings in Sydney?"
    try:
        result = create_main_agent(sample_question)
        print("Agent Response:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
