import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Add root to path so we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.basic_memory import FullMemory, SlidingWindowMemory
from core.advanced_memory import RelevanceMemory, SummaryMemory
from core.external_memory import VectorMemory, GraphMemory
from core.hybrid_memory import HierarchicalMemory, OSMemory
from langchain_openai import ChatOpenAI

# Load Env
load_dotenv()

st.set_page_config(page_title="LLM Memory Patterns", layout="wide")

st.title("🧠 LLM Memory Patterns Playground")
st.markdown("""
Explore how different memory strategies manipulate the Context Window. 
This is **NOT** just a chatbot; it's an inspector for the AI's brain.
""")

# Sidebar
st.sidebar.header("Configuration")
strategy_name = st.sidebar.selectbox(
    "Choose Memory Strategy",
    [
        "1. Full Memory",
        "2. Sliding Window",
        "3. Relevance Filtering",
        "4. Summary Memory",
        "5. Vector Memory (RAG)",
        "6. Knowledge Graph",
        "7. Hierarchical (Hybrid)",
        "8. OS-Like (MemGPT)"
    ]
)

# Strategy Factory
def get_strategy(name):
    if "Full" in name: return FullMemory()
    if "Sliding" in name: return SlidingWindowMemory()
    if "Relevance" in name: return RelevanceMemory()
    if "Summary" in name: return SummaryMemory()
    if "Vector" in name: return VectorMemory()
    if "Graph" in name: return GraphMemory()
    if "Hierarchical" in name: return HierarchicalMemory()
    if "OS-Like" in name: return OSMemory()
    return FullMemory()

# Initialize Session State
if "memory_strategy" not in st.session_state or st.session_state.current_strategy_name != strategy_name:
    st.session_state.memory_strategy = get_strategy(strategy_name)
    st.session_state.current_strategy_name = strategy_name
    st.session_state.messages = [] # Clear chat on switch
    st.session_state.debug_logs = [] # Clear logs
    
    # Re-init LLM
    if os.getenv("OPENAI_API_KEY"):
        st.session_state.llm = ChatOpenAI(temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        st.error("Please set OPENAI_API_KEY in .env")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Show debug info if available
        if "debug_info" in msg:
            with st.expander("🔍 Internal Monologue & Memory State"):
                st.json(msg["debug_info"])


# User Input
prompt = st.chat_input("Say something...")
if prompt:
    # 1. Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Add to Memory Strategy
    st.session_state.memory_strategy.add_message("user", prompt)
    
    # 3. Retrieve Context
    context = st.session_state.memory_strategy.get_context(prompt)
    final_system_prompt = context['final_prompt']
    debug_info = context['debug_info']
    
    # 4. Generate Response (Mock or Real)
    with st.chat_message("assistant"):
        if "llm" in st.session_state:
            with st.spinner("Thinking..."):
                # Construct the full prompt for the LLM
                # We use a simple convention: System Context + User Question
                full_messages = [
                    ("system", f"You are a helpful AI. Base your answer on the following memory context:\n\n{final_system_prompt}"),
                    ("user", prompt)
                ]
                response = st.session_state.llm.invoke(full_messages)
                response_text = response.content
                st.write(response_text)
        else:
            response_text = "LLM not initialized. Check API Key."
            st.error(response_text)
            
        # Show Debug Info immediately
        with st.expander("🔍 Internal Monologue & Memory State", expanded=True):
            st.json(debug_info)
            
    # 5. Save Assistant Message to State & Memory
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    # Store debug info aligned with message index. 
    # Messages list has [User, Assistant, User, Assistant]. 
    # Logs should align with Assistant indices.
    # We just append to a separate list and map by index in the display loop.
    # Actually, easier to just store it in the message dict for future, but user asked for specific structure.
    # Let's append to a parallel list.
    # Wait, the display loop relies on indices. 
    # Let's just append placeholders to keep length matching or use a smarter way.
    # Simpler: Add 'debug_info' to the message dict itself.
    st.session_state.messages[-1]['debug_info'] = debug_info
    
    st.session_state.memory_strategy.add_message("assistant", response_text)

# Fix display loop to use the new dict key
# Rerun to update UI properly? Streamlit auto-reruns on interaction, so the next loop will catch it.
# But for immediate display we handled it above. 
# The loop at the top needs to be updated to read from msg['debug_info'] instead of a separate list.

# Let's quick-fix the top loop logic in next edit if needed, but I'll write the file correctly now.

