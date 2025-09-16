"""
Streamlit UI for Aussie Airbnb Market Intelligence Agent

This module provides a web-based user interface for interacting with
the Airbnb market intelligence agent.
"""

import streamlit as st
from src.agent import create_main_agent


def main():
    """Main Streamlit application function."""
    
    # Set page configuration
    st.set_page_config(
        page_title="Aussie Airbnb Market Intelligence Agent",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title and header
    st.title("🏠 Aussie Airbnb Market Intelligence Agent")
    
    # Introductory text
    st.markdown("""
    Welcome to the **Aussie Airbnb Market Intelligence Agent**! 
    
    This intelligent agent analyzes Airbnb market data across **Sydney**, **Melbourne**, and **Brisbane** 
    to provide insights about pricing, trends, and market opportunities.
    
    Simply ask a question about the Airbnb market data, and the agent will provide you with 
    detailed analysis and insights.
    """)
    
    # Add some spacing
    st.markdown("---")
    
    # Main input section
    st.subheader("💬 Ask a Question")
    
    # Initialize session state for user input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Text input for user questions
    user_question = st.text_input(
        "What would you like to know about the Airbnb market?",
        placeholder="e.g., What is the average price of Airbnb listings in Sydney?",
        key="user_input",
        value=st.session_state.user_input
    )
    
    # Check if we have a question from the sidebar
    sidebar_question = None
    if "example_select" in st.session_state and st.session_state.example_select != "Select a question...":
        sidebar_question = st.session_state.example_select
    
    # Use sidebar question if available, otherwise use text input
    final_question = sidebar_question if sidebar_question else user_question
    
    # Process the question if provided
    if final_question:
        # Display spinner while processing
        with st.spinner("🤔 Thinking..."):
            try:
                # Call the agent with the user's question
                response = create_main_agent(final_question)
                
                # Display the agent's response
                st.subheader("📊 Analysis Results")
                st.write(response)
                
            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                st.info("💡 Please make sure you have set up your OpenAI API key in the .env file.")
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Example Questions")
        st.markdown("Try asking these questions:")
        
        example_questions = [
            "What is the average price of Airbnb listings in Sydney?",
            "Which city has the most expensive listings?",
            "What are the top 5 neighborhoods by average price in Melbourne?",
            "How many listings have more than 3 bedrooms?",
            "What is the price distribution across all cities?",
            "Which property types are most common in Brisbane?",
            "What is the average number of beds per listing?",
            "How do prices vary by room type?",
            "What are the most expensive neighborhoods in each city?",
            "How many listings are available for instant booking?"
        ]
        
        # Use a different approach for example questions
        selected_question = st.selectbox(
            "Or select an example question:",
            ["Select a question..."] + example_questions,
            key="example_select"
        )
        
        # Display selected question info
        if selected_question != "Select a question...":
            st.info(f"Selected: {selected_question}")
        
        # Add some additional information in the sidebar
        st.markdown("---")
        st.markdown("### 📈 Data Overview")
        st.markdown("""
        **Total Listings:** 49,766
        - Sydney: 18,187 listings
        - Melbourne: 25,801 listings  
        - Brisbane: 5,774 listings
        
        **Key Stats:**
        - Average price: $280.74
        - Median price: $179.00
        - Average bedrooms: 1.84
        - Average beds: 2.19
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. Enter your OpenAI API key in the `.env` file
        2. Install dependencies: `pip install -r requirements.txt`
        3. Run the app: `streamlit run src/app.py`
        4. Ask questions about the Airbnb market!
        """)


if __name__ == "__main__":
    main()
