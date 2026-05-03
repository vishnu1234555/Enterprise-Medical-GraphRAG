import streamlit as st
import time
from rag_bot import MedicalGraphRAG

# --- ADD THIS HELPER FUNCTION AT THE TOP OF app.py (under your imports) ---
def stream_text(text):
    """Simulates a typewriter effect for the bot 'talking back'"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04) # Speed of the typing

# Configure Page
st.set_page_config(page_title="Medical GraphRAG", page_icon="🧬", layout="wide")

# Custom Dark Theme & UI Tweaks
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-box { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Enterprise Medical GraphRAG")
st.caption("Deterministic Clinical Extraction powered by Neo4j & Groq Llama-3 70B")

# Initialize Bot in Session State (so it doesn't reload every time)
if "bot" not in st.session_state:
    with st.spinner("Initializing Graph Connection..."):
        st.session_state.bot = MedicalGraphRAG()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- REPLACE THE USER INPUT BLOCK WITH THIS ---
if prompt := st.chat_input("Ask a clinical question (e.g., 'Primary targets for breast cancer?'):"):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Bot "Talking Back"
    with st.chat_message("assistant"):
        # We use a status container so the backend thinking doesn't clutter the final chat
        with st.status("🧠 Searching 3.8M Node Medical Graph...", expanded=True) as status:
            try:
                st.write("Phase 1: Semantic Vector Search...")
                d_id, d_name, entity = st.session_state.bot.semantic_vector_search(prompt)
                
                if not d_id:
                    status.update(label="Query Rejected", state="error", expanded=False)
                    final_response = "I couldn't find any relevant medical entities for that query in my graph. I only extract targeted clinical evidence."
                    st.write_stream(stream_text(final_response))
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.stop()

                st.write(f"Phase 2: Extracting topological evidence for {d_name}...")
                context = st.session_state.bot.deterministic_traversal(d_id)
                
                if not context:
                    status.update(label="No Evidence Found", state="error", expanded=False)
                    final_response = f"I found the disease '{d_name}', but there are zero clinical evidence links attached to it in the database."
                    st.write_stream(stream_text(final_response))
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.stop()

                st.write("Phase 3: Generator Agent drafting response...")
                draft = st.session_state.bot.generate_draft_response(prompt, context)
                
                st.write("Phase 4: 70B Critic auditing RAG Triad...")
                metrics = st.session_state.bot.evaluate_rag_triad(prompt, draft, context)
                
                f_val = float(metrics.get('faithfulness', 0.0))
                if f_val < 1.0:
                    status.update(label="Hallucination Blocked", state="error", expanded=False)
                    final_response = "My Critic Agent detected a hallucination in the draft. I have discarded the response for your safety."
                    st.write_stream(stream_text(final_response))
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.stop()

                # If everything passes, collapse the status window
                status.update(label=f"Validated Evidence Extracted for: {d_name}", state="complete", expanded=False)
                
                # THIS IS WHERE IT "TALKS BACK"
                final_response = draft
                
                # Render the telemetry sidebar
                with st.sidebar:
                    st.subheader("Last Query Telemetry")
                    st.markdown(f"**Entity Matched:** `{d_name}`")
                    st.metric(label="Faithfulness", value=f"{float(metrics.get('faithfulness', 0)):.2f}")
                    st.metric(label="Relevance", value=f"{float(metrics.get('answer_relevance', 0)):.2f}")
                    st.metric(label="Precision", value=f"{float(metrics.get('context_precision', 0)):.2f}")

            except Exception as e:
                status.update(label="Pipeline Crash", state="error", expanded=False)
                final_response = f"A critical error occurred: {str(e)}"
                st.write_stream(stream_text(final_response))
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.stop()

        # After the status window collapses, stream the text into the chat bubble
        st.write_stream(stream_text(final_response))
        st.session_state.messages.append({"role": "assistant", "content": final_response})
