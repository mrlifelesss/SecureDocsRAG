# app_streamlit.py
"""
Streamlit front-end for a RAG (Retrieval-Augmented Generation) chatbot.

This application provides a user interface for interacting with the RAG graph defined
in `graph.py`. It manages the conversation state, displays chat history, handles
user input, and streams the generated responses from the backend.

This version uses in-line citation parsing to verify sources.

To run the app:
streamlit run app_streamlit.py
"""
import os
import re
from pathlib import Path
import streamlit as st
from graph import prepare_context, stream_generate, DOMAIN, MIN_CONTEXT_CHARS

# --------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Chatbot · LangGraph + Gemini",
    page_icon="🔒",
    layout="wide",
)

# --- Example questions for the sidebar ---
EXAMPLES = [
    "How does the SRA use delegated administration with AWS Organizations, and what benefits does it provide",
    "What are three recommendations to protect the AWS root user?",
    "Explain least privilege in IAM and give two concrete implementation methods.",
    "Which S3 encryption options exist and when to use each?",
]

# --------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------

# Initialize session state for chat history and conversational memory
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # Stores {role, content, sources, confidence}
if "prior_docs" not in st.session_state:
    st.session_state["prior_docs"] = [] # Stores 'kept_docs' for conversational memory

# --------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------------------------------

def render_sources(sources: list):
    """
    Renders a list of source documents with metadata and the chunk text.
    """
    if not sources:
        return

    # st.subheader("Cited Content") # Optional: Add a header

    for i, s in enumerate(sources):
        # --- Part 1: Render the metadata chip ---
        idx = s.get("index", "?")
        src = os.path.basename(str(s.get("source", "unknown")))
        page_raw = s.get("page", "n/a")
        page_disp = f"p. {page_raw + 1}" if isinstance(page_raw, int) else str(page_raw)
        doc_id = s.get("doc_id")
        id_html = f'<span class="dim" title="{doc_id}"> ID:{doc_id[:8]}…</span>' if doc_id else ""

        chip_html = f'<div style="margin-bottom: 4px;"><span class="src-chip"><span class="idx">{idx}</span> {src}<span class="dim">:{page_disp}</span>{id_html}</span></div>'
        st.markdown(chip_html, unsafe_allow_html=True)

        # --- Part 2: Render the chunk text in an info box ---
        text = s.get("text", "Content not available.")
        st.info(text, icon="📄")

        # Add a separator if it's not the last item
        if i < len(sources) - 1:
            st.markdown("---")
# --- NEW HELPER FUNCTION FOR PARSING CITATIONS ---
def parse_and_filter_sources(answer: str, sources: list) -> list:
    """
    Parses citation numbers like [1], [2] from the answer text and filters
    the source list to only include those cited.
    """
    if not answer or not sources:
        return []
    try:
        # Find all unique numbers inside square brackets
        cited_indices = {int(i) for i in re.findall(r'\[(\d+)\]', answer)}
        if not cited_indices:
            # If the LLM failed to cite, return nothing.
            return []
        # Filter the original source list
        return [s for s in sources if s.get("index") in cited_indices]
    except Exception as e:
        print(f"Error parsing sources: {e}")
        return [] # Return empty list on failure

# --------------------------------------------------------------------------
# 4. UI & PAGE LAYOUT
# --------------------------------------------------------------------------

# --- Minimal CSS for a polished look ---
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .app-header {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%); color: white;
        padding: 18px 24px; border-radius: 16px; margin-bottom: 8px;
        display: flex; align-items: center; gap: 12px;
    }
    .app-badge {
        font-size: 12px; background-color: rgba(255,255,255,0.18);
        padding: 4px 10px; border-radius: 999px; margin-left: 8px;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .stChatMessage { font-size: 16px; }
    .stChatMessage div[data-testid="chat-message-content"] { border-radius: 14px; padding: 10px 12px; }
    .stChatMessage[data-testid="chat-message-user"] div[data-testid="chat-message-content"] {
        background: #f0f9ff; border: 1px solid #e0f2fe;
    }
    .stChatMessage[data-testid="chat-message-assistant"] div[data-testid="chat-message-content"] {
        background: #f8fafc; border: 1px solid #e5e7eb;
    }
    .src-chip {
        display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
        margin: 4px 6px 0 0; border-radius: 999px; border: 1px solid #e5e7eb;
        background: #ffffff; font-size: 12px;
    }
    .src-chip .idx {
        background:#eef2ff; color:#4338ca; font-weight:600;
        padding: 2px 6px; border-radius: 999px;
    }
    .dim { color: #6b7280; }
    .pill {
        display:inline-block; padding:4px 8px; border-radius:999px;
        background:#f1f5f9; border:1px solid #e2e8f0; font-size:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <div class="app-header">
      <div style="font-size:22px; font-weight:700;">🔒 RAG Chatbot — LangGraph + Gemini</div>
      <span class="app-badge">AWS Security / IAM</span>
      <span class="app-badge">Hybrid Retrieval</span>
      <span class="app-badge">Reranked Context</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Controls")
    if st.button("🧹 Clear chat", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    st.subheader("📄 Knowledge Base")
    kb_dir = Path("storage")
    has_kb = kb_dir.exists() and any(kb_dir.iterdir())
    data_dir_exists = Path("data").exists()
    st.markdown(
        f"- Vector DB: **{'Ready' if has_kb else 'Missing'}** {'✅' if has_kb else '⚠️'}\n"
        f"- Data Folder: **{'Exists' if data_dir_exists else 'Missing'}** {'✅' if data_dir_exists else '⚠️'}"
    )
    if not has_kb:
        st.caption("Run `python ingest.py` after adding files to `data/`.")
    st.markdown("---")
    st.subheader("💡 Examples")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["_prefill"] = ex
            st.rerun()
    st.markdown("---")
    st.caption("Built with LangGraph, Gemini, and Streamlit.")


# --------------------------------------------------------------------------
# 5. MAIN CHAT INTERFACE
# --------------------------------------------------------------------------

# --- Display previous chat messages ---
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("confidence"):
                st.markdown(f"<span class='pill'>Confidence: {msg['confidence']}</span>", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("Sources", expanded=False):
                    render_sources(msg["sources"])

# --- Handle new user input ---
prefill = st.session_state.pop("_prefill", None)
user_q = st.chat_input("Ask a question about AWS security...", key="chat_input", max_chars=4000)
if prefill and not user_q:
    user_q = prefill

if user_q:
    st.session_state["chat"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            ctx_state = prepare_context(user_q, chat_history=st.session_state["chat"], prior_docs=st.session_state.get("prior_docs", []))
        
        ctx = ctx_state.get("context", "")
        sources = ctx_state.get("sources", [])
        kept_docs = ctx_state.get("kept_docs", [])
        
        if len(ctx) < MIN_CONTEXT_CHARS:
            clarify_msg = ("I don’t have enough information to answer confidently.\n\nCould you clarify one of:\n- AWS service (IAM, S3, etc.)\n- Scope (account/organization)\n- A specific keyword")
            st.markdown(clarify_msg)
            st.session_state["chat"].append({"role": "assistant", "content": clarify_msg, "sources": [], "confidence": "Low"})
        else:
            # Stream the generated response
            streamed_response = st.write_stream(stream_generate(user_q, ctx, domain=DOMAIN))

            # === NEW PARSING STEP ===
            # Parse the streamed response to find which sources the LLM cited
            verified_sources = parse_and_filter_sources(streamed_response, sources)
            
            # Extract confidence from the answer itself if present, otherwise default
            confidence_match = re.search(r"Confidence:\s*(High|Medium|Low)", streamed_response, re.IGNORECASE)
            confidence = confidence_match.group(1) if confidence_match else "Medium"
            
            # Display confidence pill and the PARSED/VERIFIED sources
            st.markdown(f"<span class='pill'>Confidence: {confidence}</span>", unsafe_allow_html=True)
            if verified_sources:
                with st.expander("Sources", expanded=False):
                    render_sources(verified_sources)
            
            # Store the response and PARSED/VERIFIED sources in chat history
            st.session_state["chat"].append({
                "role": "assistant",
                "content": streamed_response,
                "sources": verified_sources,
                "confidence": confidence,
            })

        st.session_state["prior_docs"] = kept_docs

# --- Footer ---
st.markdown(
    "<div style='text-align:center; margin-top:24px; color:#9ca3af;'>"
    "<span class='pill'>Rewrite → Retrieve → Rerank → Generate</span> "
    "<span class='pill'>Chroma + BM25</span> "
    "<span class='pill'>Gemini</span>"
    "</div>",
    unsafe_allow_html=True,
)