import streamlit as st
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

import qdrant_client

# Set up environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Constants
REPO_ID = "codellama/CodeLlama-7b-hf"
QDRANT_PATH = "./local_qdrant"
QDRANT_COLLECTION_NAME = "my_documents"

# Streamlit UI setup
st.set_page_config(page_title="CodeLlama RAG System", layout="wide")

st.title("CodeLlama RAG System")

# Sidebar for inputs
st.sidebar.header("Settings")

# Directory input
root_dir = st.sidebar.text_input("Enter the root directory path:", "codebase")


# Function to load documents
def load_documents(root_dir: str) -> List:
    loader = DirectoryLoader(
        root_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()
    st.sidebar.success(f"Loaded {len(documents)} documents")
    return documents

# Load documents when the directory is provided
if root_dir:
    documents = load_documents(root_dir)

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    st.sidebar.success(f"Split into {len(texts)} chunks")

    # Initialize embeddings
    
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    embeddings = get_embeddings()

    # Set up Qdrant vector store

    if not os.path.exists(QDRANT_PATH):
        os.makedirs(QDRANT_PATH)


    def get_vector_store(texts, embeddings):
        client = qdrant_client.QdrantClient(location=":memory:")  # In-memory for testing
        return Qdrant.from_documents(
            texts, embeddings, collection_name=QDRANT_COLLECTION_NAME
        )


    vector_store = get_vector_store(texts, embeddings)
    st.sidebar.success("Vector store initialized")

    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # Set up LLM
    
    def get_llm():
        return HuggingFaceHub(
            repo_id=REPO_ID, model_kwargs={"temperature": 0.5, "max_length": 500}
        )

    llm = get_llm()

    # Set up QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Main query interface
    st.header("Ask a question about your codebase")
    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("Processing query..."):
            result = qa_chain.run(query)
        st.subheader("Answer:")
        st.write(result)

    # Display relevant documents
    if st.checkbox("Show relevant documents"):
        st.subheader("Relevant Documents:")
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs):
            st.markdown(f"**Document {i+1}:**")
            st.text(doc.page_content)
            st.markdown("---")

else:
    st.warning("Please enter a valid directory path in the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CodeLlama RAG System - Powered by Streamlit, LangChain, and Qdrant")