#!/usr/bin/env python3
"""
Equity News Research Tool
A RAG-based application for analyzing equity news and providing investment insights.
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set USER_AGENT environment variable to avoid warnings
os.environ["USER_AGENT"] = "EquityNewsResearchTool/1.0"

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper class for SentenceTransformer to work with LangChain."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


class EquityNewsResearchTool:
    """Main class for the Equity News Research Tool."""
    
    def __init__(self, groq_api_key: str = None):
        """Initialize the tool with API key."""
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file or pass it directly.")
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.rag_chain = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all the required components."""
        print("🚀 Initializing Equity News Research Tool...")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1000,
            api_key=self.groq_api_key
        )
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
        
        print("✅ Components initialized successfully!")
    
    def load_news_from_urls(self, urls: List[str]) -> None:
        """Load and process news from URLs."""
        print(f"📰 Loading news from {len(urls)} URLs...")
        
        # Load documents
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        # Clean the documents
        for doc in documents:
            doc.page_content = '\n'.join([
                line.strip() for line in doc.page_content.split('\n') 
                if line.strip()
            ])
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"📊 Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create RAG chain
        self._create_rag_chain()
        
        print("✅ News data loaded and indexed successfully!")
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering."""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst specializing in equity research. 
Answer the following question based on the provided news context.

Context: {context}

Question: {question}

Provide a comprehensive, well-structured answer with:
1. Key insights from the news
2. Investment implications
3. Risk considerations
4. Supporting evidence from the context

Answer:
""")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_question(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """Ask a question about the loaded news."""
        if not self.rag_chain:
            raise ValueError("No news data loaded. Please load news first.")
        
        print(f"🤔 Question: {question}")
        
        # Get relevant chunks
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        
        # Show sources if requested
        if show_sources:
            print(f"\n📚 Found {len(retrieved_docs)} relevant sources:")
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get('source', 'Unknown')
                print(f"  {i+1}. {source}")
        
        # Get answer
        answer = self.rag_chain.invoke({"question": question})
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs] if show_sources else []
        }
    
    def save_index(self, path: str = "equity_news_index") -> None:
        """Save the FAISS index for later use."""
        if not self.vectorstore:
            raise ValueError("No vector store to save.")
        
        self.vectorstore.save_local(path)
        print(f"💾 Index saved to {path}")
    
    def load_index(self, path: str = "equity_news_index") -> None:
        """Load a previously saved FAISS index."""
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self._create_rag_chain()
        print(f"📂 Index loaded from {path}")


def main():
    """Main function for command-line usage."""
    print("=" * 60)
    print("🔍 EQUITY NEWS RESEARCH TOOL")
    print("=" * 60)
    
    try:
        # Initialize the tool
        tool = EquityNewsResearchTool()
        
        # Example URLs (you can modify these)
        urls = [
            "https://www.investing.com/analysis/heres-the-best-buffett-stock--and-its-not-apple-or-amazon-200668829",
            "https://www.blackrock.com/us/individual/investment-ideas/fundamental-equities"
        ]
        
        # Load news data
        tool.load_news_from_urls(urls)
        
        # Save the index for future use
        tool.save_index()
        
        # Example questions
        questions = [
            "What are the key investment themes in the current market?",
            "Which stocks are mentioned as good investment opportunities?",
            "What are the main risks mentioned in the news?"
        ]
        
        print("\n" + "=" * 60)
        print("💬 SAMPLE QUESTIONS & ANSWERS")
        print("=" * 60)
        
        for question in questions:
            result = tool.ask_question(question)
            print(f"\n❓ {result['question']}")
            print(f"💡 {result['answer']}")
            print("-" * 40)
        
        print("\n✅ Tool is ready! You can now ask questions about equity news.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
