import streamlit as st
import os

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

if process_url_clicked:
    # Load the URLs into a loader
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(urls)
    data = loader.load()

    # Split the data into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", ","], chunk_size=1000)
    chunks = text_splitter.split_documents(data)

    # Embed the chunks using SentenceTransformerEmbeddings from langchain_community
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store them in a pickle file
    import pickle
    from langchain_community.vectorstores import FAISS
    faiss_index = FAISS.from_documents(chunks, embeddings)
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(faiss_index, f)
   

    st.success(f"Successfully created vectorstore with {len(chunks)} chunks!")
    st.write(f"Number of chunks: {len(chunks)}")
    st.write(f"Vectorstore ready for similarity search!")
