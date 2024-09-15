from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def get_llm(repo_id):
    """Retrieves the HuggingFace LLM."""
    return HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 500}
    )

def setup_qa_chain(llm, retriever):
    """Sets up the QA chain with the retriever and LLM."""
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
