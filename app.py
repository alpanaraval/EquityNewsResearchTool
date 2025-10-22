import streamlit as st
import os
import time
import pickle
file_path = "faiss_index.pkl"

from langchain_groq import ChatGroq
from dotenv import load_dotenv  
load_dotenv()

st.title("Equity News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
if process_url_clicked:
    # Load the URLs into a loader
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(urls)
    main_placeholder.text("Data Loading.....Started.....")
    data = loader.load()

    # Split the data into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", ","], chunk_size=1000)
    main_placeholder.text("Data Splitting.....Started.....")
    chunks = text_splitter.split_documents(data)

    # Embed the chunks using SentenceTransformerEmbeddings from langchain_community
    #from langchain_community.embeddings import SentenceTransformerEmbeddings
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")'''
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
  
    # Store them in a pickle file
    import pickle
    from langchain_community.vectorstores import FAISS
    faiss_index = FAISS.from_documents(chunks, embeddings)
    main_placeholder.text("Data Embedding.....Vector Store Started.....")
    time.sleep(2)

    # Store the vector store in a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(faiss_index, f)

# Creating a question box for user to ask questions
query = st.text_input("Ask a question about the news articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
                       
        # Initialize Groq LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.9
        )
        
        # Simple RAG approach - retrieve documents and generate answer
        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Define format_docs function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the RAG chain using LCEL
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(template="Based on the following context, answer the question. If you don't know the answer based on the context, say so.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:", input_variables=["context", "question"])
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        result = rag_chain.invoke(query)
        
        st.write("**Answer:**")
        st.write(result)

        # Only display sources if the answer indicates information was found
        no_info_indicators = [
            "i don't see any information",
            "i don't know",
            "no information",
            "not found",
            "cannot find",
            "unable to find",
            "no relevant information"
        ]
        
        # Check if the answer indicates no information was found
        answer_lower = result.lower()
        has_relevant_info = not any(indicator in answer_lower for indicator in no_info_indicators)
        
        if has_relevant_info:
            # Display sources from retrieved documents
            st.write("**Sources:**")
            retrieved_docs = retriever.invoke(query)
            if retrieved_docs:
                # Get unique sources to avoid duplicates
                unique_sources = []
                seen_sources = set()
                for doc in retrieved_docs:
                    source = doc.metadata.get('source', 'Unknown source')
                    if source not in seen_sources:
                        unique_sources.append(source)
                        seen_sources.add(source)
                
                for i, source in enumerate(unique_sources, 1):
                    st.write(f"{i}. {source}")
            else:
                st.write("No sources found")
    else:
        st.error("Vector store not found. Please process URLs first.")
