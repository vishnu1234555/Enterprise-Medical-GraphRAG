"""
app.py
======
Enterprise Medical GraphRAG — Conversational Streamlit Frontend

Features
--------
• Full persistent chat history via st.session_state
• Agent initialised once per server session via @st.cache_resource
• Typewriter streaming effect on every assistant response
• Graceful error display — never crashes the UI
"""

import os
import time
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.agent.rag_bot import MedicalAgent
from sentence_transformers import SentenceTransformer
from src.database.neo4j_client import get_driver

# ---------------------------------------------------------------------------
# Page Configuration  (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical GraphRAG Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark clinical theme with subtle micro-animations
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #64748b;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #38bdf8;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 0.4rem;
        animation: fadeSlideIn 0.3s ease both;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(129, 140, 248, 0.06);
        border: 1px solid rgba(129, 140, 248, 0.15);
    }
    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(56, 189, 248, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.12);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1) !important;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 0.75rem 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Spinner text ── */
    .stSpinner > div > div { border-top-color: #38bdf8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🧬 Enterprise Medical GraphRAG</h1>
        <p>Actor-Critic Reasoning Engine · Neo4j Knowledge Graph · Groq Llama-3</p>
    </div>
    <div style="text-align:center;">
        <span class="status-badge">
            <span class="status-dot"></span>
            SYSTEM ONLINE
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Chat transcript (must exist before any branch reads it)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Agent Initialisation — cached for the lifetime of the Streamlit server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_globals():
    return get_driver(), SentenceTransformer("all-MiniLM-L6-v2")

driver, embedder = load_globals()

@st.cache_resource(show_spinner=False)
def get_agent(_conn_signature: str) -> MedicalAgent:
    """Recreate agent when Neo4j connection env changes (cached driver otherwise)."""
    return MedicalAgent(driver=driver, embedder=embedder)


with st.spinner("🔗 Connecting to Neo4j graph & loading models…"):
    _sig = f"{os.getenv('NEO4J_URI', '')}|{os.getenv('NEO4J_USER', '')}|{os.getenv('NEO4J_DATABASE', '')}"
    agent = get_agent(_sig)

# ---------------------------------------------------------------------------
# Sidebar — session controls & info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Session Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔌 Reconnect Neo4j (reload agent)", use_container_width=True):
        # Drops cached MedicalAgent / driver so code + .env changes take effect
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Architecture")
    st.markdown(
        """
        **Pipeline steps:**
        1. 🔍 Semantic vector search (Neo4j)
        2. 🕸️ Deterministic graph traversal
        3. ✍️ Actor LLM — retrieve & explain targets
        4. ✅ Validated response → UI
        """
    )



# ---------------------------------------------------------------------------
# Chat State Initialisation (welcome message on first load)
# ---------------------------------------------------------------------------

# Welcome message — shown only once at the start of a new session
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "👋 Welcome to the **Enterprise Medical GraphRAG Assistant**.\n\n"
                "I'm connected to a Neo4j biomedical knowledge graph and powered "
                "by an **Actor-Critic reasoning engine** that retrieves graph facts, "
                "drafts an answer, and then fact-checks it against the raw database "
                "context before responding — minimising hallucination.\n\n"
                "Ask me anything about disease targets, gene associations, or "
                "clinical evidence. *What would you like to explore?*"
            ),
        }
    )

# ---------------------------------------------------------------------------
# Render Conversation History
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Typewriter Streaming Helper
# ---------------------------------------------------------------------------
def stream_text(text: str, delay: float = 0.018):
    """Yields words one by one for a typewriter effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)



# ---------------------------------------------------------------------------
# Main Chat Input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a clinical question — e.g. 'Primary targets for breast cancer?'"):

    chat_history = list(st.session_state.messages)

    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run the Actor-Critic backend pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving graph context & running Critic validation…"):
            try:
                final_response = asyncio.run(agent.execute_query(prompt, chat_history=chat_history))
            except Exception as exc:
                final_response = (
                    f"⚠️ **Unexpected error:** `{type(exc).__name__}: {exc}`\n\n"
                    "Please check the server logs and retry."
                )

        # 3. Stream the validated answer
        st.write_stream(stream_text(final_response))

    # 4. Persist to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": final_response}
    )
