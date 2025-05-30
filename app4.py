import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing import TypedDict, Annotated, List

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, verbose=False)

# Tavily tool for web search
search_tool = TavilySearchResults(k=5)
tools = [search_tool]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Agent state
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    symptoms: str
    diagnosis_complete: bool

# System prompt
MEDICAL_SYSTEM_PROMPT = """
You are an experienced and knowledgeable medical AI assistant acting as a licensed doctor.

Given the patient's symptoms, perform the following steps carefully:

1. Identify the most likely medical condition(s) based on symptoms.
2. Provide a clear diagnosis name.
3. Recommend specific medicines with:
   - Medicine name and exact strength
   - Dosage quantity per intake
   - Frequency
   - Duration
4. Include precautions, potential side effects, and warnings.
5. Suggest when to seek immediate medical attention.
6. Suggest self-care/lifestyle advice.

Format:

**Condition:** [Name]

**Medicines:**
- [Medicine]: [Dosage] - [Frequency] - [Duration]

**Precautions:** [Precautions]
"""

# LangGraph agent
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
        return f"Error in diagnosis: {str(e)}"

# --------------------- Streamlit UI -----------------------

st.set_page_config("🩺 HealthMate AI", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 🩺 HealthMate AI")
    st.info("Enter your symptoms below to receive medical advice powered by AI.")
    st.caption("💡 Built with Gemini, Tavily & LangGraph")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main title
st.markdown("<h2 style='text-align:center;'>💬 Describe Your Symptoms</h2>", unsafe_allow_html=True)

# Input text area
with st.container():
    symptoms = st.text_area(
        "Enter symptoms here...",
        placeholder="Ex: High fever, body ache, sore throat for 2 days...",
        height=150,
        label_visibility="collapsed"
    )

    if st.button("🔍 Diagnose"):
        if symptoms.strip():
            with st.spinner("🤖 Analyzing your symptoms..."):
                reply = diagnose_symptoms(symptoms)
                st.session_state.chat_history.insert(0, {"role": "assistant", "content": reply})
                st.session_state.chat_history.insert(0, {"role": "user", "content": symptoms})
        else:
            st.warning("Please enter symptoms before diagnosis.")

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧾 Diagnosis Results")
    for chat in st.session_state.chat_history:
        with st.chat_message(name=chat["role"], avatar="🧍" if chat["role"] == "user" else "🩺"):
            st.markdown(chat["content"], unsafe_allow_html=True)
