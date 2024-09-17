import streamlit as st
from constants import QDRANT_PATH, QDRANT_COLLECTION_NAME, REPO_ID
from utils import load_documents, split_text, initialize_vector_store, get_embeddings
from llm_utils import get_llm, setup_qa_chain
import os
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit UI setup
st.set_page_config(page_title="Enhanced CodeLlama RAG System", layout="wide")
st.title("Enhanced CodeLlama RAG System")
st.sidebar.header("Settings")

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Set up environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["huggingfacehub_api_token"]

# Constants
REPO_ID = config["repo_id"]
QDRANT_PATH = config["qdrant_path"]
QDRANT_COLLECTION_NAME = config["qdrant_collection_name"]

# Directory input
root_dir = st.sidebar.text_input("Enter the root directory path:", "codebase")

if root_dir:
    documents = load_documents(root_dir)
    st.sidebar.success(f"Loaded {len(documents)} documents")

    texts = split_text(documents)
    st.sidebar.success(f"Split into {len(texts)} chunks")

    embeddings = get_embeddings()

    vector_store = initialize_vector_store(texts, embeddings, QDRANT_PATH, QDRANT_COLLECTION_NAME)
    st.sidebar.success("Vector store initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    llm = get_llm(REPO_ID)
    qa_chain = setup_qa_chain(llm, retriever)

    # Create an explanation chain
    explanation_template = """
    Analyze and explain the following result:
    {result}

    Please provide:
    1. A summary of the main points
    2. Any technical concepts mentioned and their explanations
    3. Potential implications or applications of this information
    """
    explanation_prompt = PromptTemplate(template=explanation_template, input_variables=["result"])
    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

    # Main query interface
    st.header("Ask a question about your codebase")
    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("Processing query..."):
            result = qa_chain.run(query)
            explanation = explanation_chain.run(result=result)

        st.subheader("Answer:")
        st.write(result)

        st.subheader("Explanation:")
        st.write(explanation)

    # Display and analyze relevant documents
    if st.checkbox("Show and analyze relevant documents"):
        st.subheader("Relevant Documents:")
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs):
            st.markdown(f"**Document {i + 1}:**")
            st.text(doc.page_content)

            # Analyze document content
            doc_analysis = explanation_chain.run(result=doc.page_content)
            st.subheader(f"  Analysis of Document {i + 1}:")
            st.write(doc_analysis)

            st.markdown("---")

else:
    st.warning("Please enter a valid directory path in the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Enhanced CodeLlama RAG System - Powered by Streamlit, LangChain, and Qdrant")