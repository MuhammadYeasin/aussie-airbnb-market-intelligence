# Aussie Airbnb Market Intelligence Agent

A Python-based agent that analyzes Airbnb market data across Sydney, Melbourne, and Brisbane using OpenAI's GPT models.

## Project Structure

```
├── data/
│   ├── raw/                    # Original CSV files
│   │   ├── sydney_listings.csv
│   │   ├── melbourne_listings.csv
│   │   └── brisbane_listings.csv
│   └── processed/              # Cleaned and unified data
│       └── airbnb_unified_data.csv
├── notebooks/
│   └── 01_data_processing.ipynb # Data cleaning and processing
├── src/
│   └── agent.py               # Core agent logic
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   
   **Option 1: Use the setup script (Recommended)**
   ```bash
   python setup_env.py
   ```
   
   **Option 2: Manual setup**
   ```bash
   # Copy the template
   cp config_template.txt .env
   
   # Edit .env file and add your OpenAI API key
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

3. **Run Data Processing** (if not already done)
   ```bash
   jupyter notebook notebooks/01_data_processing.ipynb
   ```

## Usage

### Running the Web Application

1. **Start the Streamlit App**
   ```bash
   # Option 1: Using the run script
   python run_app.py
   
   # Option 2: Direct Streamlit command
   streamlit run src/app.py
   ```

2. **Access the Web Interface**
   - The app will automatically open in your browser
   - If not, go to: http://localhost:8501
   - Use the sidebar to see example questions
   - Type your questions in the main input box

### Using the Agent Programmatically

```python
from src.agent import create_main_agent, create_rag_tool

# Example quantitative questions
quantitative_questions = [
    "What is the average price of Airbnb listings in Sydney?",
    "Which city has the most expensive listings?",
    "What are the top 5 neighborhoods by average price in Melbourne?",
    "How many listings have more than 3 bedrooms?",
    "What is the price distribution across all cities?"
]

# Get quantitative insights
for question in quantitative_questions:
    response = create_main_agent(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")

# Example qualitative questions using RAG
rag_tool = create_rag_tool()
qualitative_questions = [
    "find properties with dedicated workspace",
    "show me pet-friendly listings",
    "properties with ocean views",
    "places near public transportation"
]

# Get qualitative insights
for question in qualitative_questions:
    response = rag_tool.func(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

### Data Overview

The processed dataset contains **49,766 Airbnb listings** across three Australian cities:

- **Sydney**: 18,187 listings
- **Melbourne**: 25,801 listings  
- **Brisbane**: 5,774 listings

**Key Statistics:**
- Average price: $280.74
- Median price: $179.00
- Average bedrooms: 1.84
- Average beds: 2.19

## Features

- **Data Processing**: Automated cleaning and merging of Airbnb data
- **AI-Powered Analysis**: Uses OpenAI's GPT-4o for intelligent data insights
- **Pandas Integration**: Leverages pandas dataframe agent for complex queries
- **Multi-City Support**: Analyzes data across Sydney, Melbourne, and Brisbane
- **Web Interface**: User-friendly Streamlit UI with interactive sidebar
- **Example Questions**: Pre-built question templates for easy exploration
- **Real-time Analysis**: Instant responses with loading indicators
- **RAG System**: Semantic search for qualitative property features and amenities
- **Vector Search**: OpenAI embeddings with Chroma vector store for intelligent property discovery

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## Environment Variables

The application uses the following environment variables (configured in `.env` file):

### Required
- `OPENAI_API_KEY`: Your OpenAI API key (get from https://platform.openai.com/api-keys)

### Optional
- `OPENAI_MODEL`: Model to use (default: "gpt-4o", alternative: "gpt-3.5-turbo")
- `OPENAI_TEMPERATURE`: Model temperature (default: 0 for factual responses)

### Example .env file:
```bash
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0
```

## Next Steps

This is Sprint 1 of the MVP. Future sprints will include:
- Web interface for the agent
- Advanced analytics and visualizations
- Market trend analysis
- Investment recommendations
