import streamlit as st
from codellama_agent import run_codellama_agent
from constants import QDRANT_PATH, QDRANT_COLLECTION_NAME, REPO_ID
from utils import load_documents, split_text, initialize_vector_store, get_embeddings
from llm_utils import get_llm, setup_qa_chain
import os
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Streamlit UI setup
st.set_page_config(page_title="Code RAG Using CodeLlama And Qdrant", layout="wide")
st.title("Code RAG Using CodeLlama And Qdrant")
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
            agent_result = run_codellama_agent(result)  # Run our new agent on the result

        st.subheader("Answer:")
        st.write(result)

        st.subheader("AI Agent Analysis:")
        st.write(agent_result['analysis'])

        st.subheader("AI Agent Explanation:")
        st.write(agent_result['explanation'])

        st.subheader("AI Agent Suggested Improvements:")
        st.write(agent_result['improvements'])

    # Display and analyze relevant documents
    if st.checkbox("Show and analyze relevant documents"):
        st.subheader("Relevant Documents:")
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs):
            st.markdown(f"**Document {i + 1}:**")
            st.text(doc.page_content)

            # Analyze document content using our new agent
            doc_analysis = run_codellama_agent(doc.page_content)
            st.subheader(f"AI Agent Analysis of Document {i + 1}:")
            st.write(doc_analysis['analysis'])
            st.write(doc_analysis['explanation'])
            st.write(doc_analysis['improvements'])

            st.markdown("---")

else:
    st.warning("Please enter a valid directory path in the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Enhanced CodeLlama RAG System - Powered by Streamlit, LangChain, and Qdrant")