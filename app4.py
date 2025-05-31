import streamlit as st
import os
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import TypedDict, Annotated, List

# Load environment variables
load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, verbose=False)

# Tavily Search Tool
search_tool = TavilySearchResults(k=5)
tools = [search_tool]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    symptoms: str
    diagnosis_complete: bool

# System Prompt
MEDICAL_SYSTEM_PROMPT = """
You are an experienced and knowledgeable medical AI assistant acting as a licensed doctor.

Given the patient's symptoms, perform the following steps carefully:

1. Identify the most likely medical condition(s).
2. Provide a clear diagnosis.
3. Recommend specific medicines:
   - Name, strength, dosage, frequency, and duration
4. Include precautions and side effects
5. Suggest when to see a doctor
6. Provide self-care advice

Format:

**Condition:** [Name]

**Medicines:**
- [Medicine]: [Dosage] - [Frequency] - [Duration]

**Precautions:** [Details]
"""

# LangGraph Logic
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "tools" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else "end"

def call_model(state: AgentState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=MEDICAL_SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    if hasattr(last_message, 'tool_calls'):
        for tool_call in last_message.tool_calls:
            try:
                result = tools_dict[tool_call["name"]].run(tool_call["args"])
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            except Exception as e:
                tool_messages.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

def create_medical_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()

medical_agent = create_medical_agent()

def diagnose_symptoms(symptoms: str) -> str:
    user_prompt = f"Patient symptoms: {symptoms}"
    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "symptoms": symptoms,
        "diagnosis_complete": False
    }
    try:
        result = medical_agent.invoke(initial_state)
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- Streamlit UI ----------------
st.set_page_config("🩺 HealthMate AI", layout="centered", page_icon="🩺")

st.title("🩺 HealthMate AI")
st.caption("Your private AI health assistant. Enter your symptoms below to get a medical diagnosis.")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Describe your symptoms..."):
    # Show user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms..."):
            response = diagnose_symptoms(prompt)
            st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
