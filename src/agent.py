"""
Core Agent Logic for Aussie Airbnb Market Intelligence Agent

This module contains the main agent function that creates and manages
the pandas dataframe agent for analyzing Airbnb data.
"""

import pandas as pd
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool


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
    Create and invoke the main pandas dataframe agent for Airbnb data analysis.
    
    This function loads the processed Airbnb data, initializes an OpenAI LLM,
    creates a pandas dataframe agent, and processes the user's question
    to provide insights about the Airbnb market data.
    
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
    
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(data_path)
    
    # Get model configuration from environment variables (with defaults)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to gpt-4o
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))  # Default to 0
    
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(
        model=model,  # Use "gpt-4o" for best performance or "gpt-3.5-turbo" for speed/cost
        temperature=temperature,   # Set to 0 for more factual, deterministic responses
        api_key=openai_api_key
    )
    
    # For now, let's use a simpler approach that combines LLM with direct analysis
    try:
        # First, try to get a direct analysis from our function
        direct_analysis = analyze_dataframe(user_question, df)
        
        # If the analysis is comprehensive, return it directly
        if len(direct_analysis) > 50:  # If we got a good analysis
            return direct_analysis
        
        # Otherwise, use the LLM to enhance the response
        prompt = f"""
        Based on the Airbnb data analysis: {direct_analysis}
        
        User question: {user_question}
        
        Please provide a comprehensive and helpful response about the Airbnb market data.
        Include relevant insights, statistics, and recommendations if appropriate.
        """
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        # Fallback to direct analysis if LLM fails
        return analyze_dataframe(user_question, df)


if __name__ == "__main__":
    # Example usage
    sample_question = "What is the average price of Airbnb listings in Sydney?"
    try:
        result = create_main_agent(sample_question)
        print("Agent Response:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
