# File: codellama_agent.py

from typing import Dict, TypedDict, Any
from langgraph.graph import StateGraph, Graph
from langchain_core.messages import HumanMessage, AIMessage
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import runnable

# Define the state of our agent
class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]
    next_step: str

# Initialize our language model using Ollama with Llama 3.1
llm = Ollama(model="llama3.1")

# Define our agent's steps
def analyze_code(state: AgentState) -> AgentState:
    messages = state['messages']
    code_analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code analysis expert. Analyze the following code and provide insights."),
        ("human", "{input}")
    ])
    code_analysis_chain = code_analysis_prompt | llm
    response = code_analysis_chain.invoke({"input": messages[-1].content})
    state['messages'].append(AIMessage(content=response))  # response is already a string
    state['next_step'] = 'explain_result'
    return state

def explain_result(state: AgentState) -> AgentState:
    messages = state['messages']
    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at explaining technical concepts. Explain the following analysis in simpler terms."),
        ("human", "{input}")
    ])
    explanation_chain = explanation_prompt | llm
    response = explanation_chain.invoke({"input": messages[-1].content})
    state['messages'].append(AIMessage(content=response))  # response is already a string
    state['next_step'] = 'suggest_improvements'
    return state

def suggest_improvements(state: AgentState) -> AgentState:
    messages = state['messages']
    improvement_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a software optimization expert. Suggest improvements for the following code and analysis."),
        ("human", "{input}")
    ])
    improvement_chain = improvement_prompt | llm
    response = improvement_chain.invoke({"input": "\n".join([m.content for m in messages])})
    state['messages'].append(AIMessage(content=response))  # response is already a string
    state['next_step'] = 'end'
    return state

# Define our workflow
workflow = StateGraph(AgentState)

# Add nodes to our graph
workflow.add_node("analyze_code", analyze_code)
workflow.add_node("explain_result", explain_result)
workflow.add_node("suggest_improvements", suggest_improvements)

# Add edges to our graph
workflow.add_edge('analyze_code', 'explain_result')
workflow.add_edge('explain_result', 'suggest_improvements')
workflow.set_entry_point("analyze_code")

# Compile the graph
graph = workflow.compile()

# Function to run our agent
def run_codellama_agent(code: str) -> Dict[str, Any]:
    result = graph.invoke({
        "messages": [HumanMessage(content=code)],
        "next_step": "analyze_code"
    })
    return {
        "analysis": result['messages'][1].content,
        "explanation": result['messages'][2].content,
        "improvements": result['messages'][3].content
    }
