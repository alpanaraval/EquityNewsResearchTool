# Equity News Research Tool

A Streamlit application that allows users to research equity news by processing URLs and asking questions about the content using RAG (Retrieval-Augmented Generation).

## Features

- **URL Processing**: Load and process up to 3 news article URLs
- **Document Chunking**: Split articles into manageable chunks for better processing
- **Vector Search**: Use FAISS for efficient document retrieval
- **AI-Powered Q&A**: Ask questions about the processed articles using Groq's LLM
- **Source Attribution**: See which sources were used to generate answers

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Process URLs**: Enter up to 3 news article URLs in the sidebar and click "Process URLs"
2. **Ask Questions**: Once processing is complete, ask questions about the articles
3. **View Sources**: The app will show which sources were used to answer your questions

## Deployment

This app is designed to be deployed on **Streamlit Community Cloud**:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Community Cloud
3. Set your `GROQ_API_KEY` as a secret in the Streamlit Community Cloud dashboard
4. Deploy!

## Requirements

- Python 3.8+
- Groq API key
- Internet connection for web scraping

## Dependencies

- Streamlit for the web interface
- LangChain for RAG implementation
- FAISS for vector search
- Groq for LLM integration
- BeautifulSoup for web scraping