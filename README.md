# Equity News Research Tool

A RAG (Retrieval-Augmented Generation) application for analyzing equity news and providing investment insights using LangChain and Groq LLM.

## Features

- **Web Scraping**: Automatically scrape equity news from multiple sources
- **Intelligent Chunking**: Break down large documents into manageable chunks
- **Vector Search**: Use FAISS for efficient similarity search
- **RAG Pipeline**: Combine retrieval with generation for accurate answers
- **Source Tracking**: Always know where your information comes from
- **Persistent Storage**: Save and load vector indexes for faster access

## Requirements

- Python 3.8+
- Groq API key

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd EquityNewsResearchTool
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Basic Usage

```python
from equity_news_research_tool import EquityNewsResearchTool

# Initialize the tool
tool = EquityNewsResearchTool()

# Load news from URLs
urls = [
    "https://www.investing.com/analysis/some-equity-news",
    "https://www.bloomberg.com/news/equity-analysis"
]
tool.load_news_from_urls(urls)

# Ask questions
result = tool.ask_question("What are the key investment opportunities mentioned?")
print(result['answer'])
```

### Command Line Usage

```bash
python equity_news_research_tool.py
```

### Advanced Features

```python
# Save index for later use
tool.save_index("my_equity_index")

# Load previously saved index
tool.load_index("my_equity_index")

# Ask questions with source tracking
result = tool.ask_question(
    "What are the main risks mentioned?", 
    show_sources=True
)
print(f"Sources: {result['sources']}")
```

## Project Structure

```
EquityNewsResearchTool/
├── equity_news_research_tool.py  # Main application
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (not in git)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
└── equity_news_index/           # Saved FAISS indexes (not in git)
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLM access

### Customization

You can customize the tool by modifying:

- **Chunk size**: Adjust `chunk_size` in `RecursiveCharacterTextSplitter`
- **Retrieval count**: Change `k` value in `search_kwargs`
- **LLM parameters**: Modify temperature, max_tokens, etc.
- **Prompt template**: Customize the prompt in `_create_rag_chain`

## How It Works

1. **Data Ingestion**: Scrape news articles from provided URLs
2. **Text Processing**: Clean and chunk the content
3. **Embedding**: Convert text chunks to vector embeddings
4. **Indexing**: Store embeddings in FAISS vector database
5. **Retrieval**: Find relevant chunks for user questions
6. **Generation**: Use LLM to generate answers based on retrieved context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors before making investment choices.
