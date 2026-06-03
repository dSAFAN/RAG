import streamlit as st
import os
import tempfile
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Background */
.stApp {
    background-color: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99, 60, 255, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0, 210, 180, 0.08) 0%, transparent 50%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(14, 14, 22, 0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Sidebar title */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Main header */
.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.main-header span {
    background: linear-gradient(135deg, #7c5cfc, #00d2b4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-tagline {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.35);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}

/* Chat messages */
.chat-bubble-user {
    background: rgba(124, 92, 252, 0.15);
    border: 1px solid rgba(124, 92, 252, 0.3);
    border-radius: 2px 16px 16px 16px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    color: #e8e8f0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-bubble-ai {
    background: rgba(0, 210, 180, 0.06);
    border: 1px solid rgba(0, 210, 180, 0.18);
    border-radius: 16px 2px 16px 16px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    color: #d4f5f0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.5;
}
.chat-label-user { color: #a08cff; }
.chat-label-ai   { color: #00d2b4; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
}
.badge-ready  { background: rgba(0,210,100,0.15); color: #00d264; border: 1px solid rgba(0,210,100,0.3); }
.badge-idle   { background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.4); border: 1px solid rgba(255,255,255,0.1); }
.badge-working{ background: rgba(124,92,252,0.15); color: #a08cff; border: 1px solid rgba(124,92,252,0.3); }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.2rem 0;
}

/* Source chunks */
.source-chunk {
    background: rgba(255,255,255,0.03);
    border-left: 2px solid rgba(124,92,252,0.4);
    border-radius: 0 4px 4px 0;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    line-height: 1.5;
}

/* File uploader override */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Input */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(124,92,252,0.5) !important;
    box-shadow: 0 0 0 2px rgba(124,92,252,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c5cfc, #5c3cdc) !important;
    border: none !important;
    border-radius: 6px !important;
    color: white !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Scrollable chat area */
.chat-container {
    max-height: 56vh;
    overflow-y: auto;
    padding-right: 0.3rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(124,92,252,0.3) transparent;
}

/* Remove Streamlit branding clutter */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "db_source" not in st.session_state:
    st.session_state.db_source = None


# ─── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_default_vectorstore(_embedding):
    if os.path.exists("chromaDB"):
        return Chroma(persist_directory="chromaDB", embedding_function=_embedding)
    return None


def build_vectorstore_from_pdf(pdf_bytes, filename, embedding):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(documents=chunks, embedding=embedding)
    return vs, len(chunks)


def get_answer(query, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatMistralAI(model="mistral-small-2603")
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Helpful AI Assistant. "
         "Use ONLY the provided context to answer the question. "
         'If the answer is not present in the context, say "I could not find the answer in the document."'),
        ("human", "Context : {context}\n\nQuestion : {question}"),
    ])
    final_prompt = template.invoke({"question": query, "context": context})
    response = llm.invoke(final_prompt)
    return response.content, docs


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">RAG — Retrieval Assistant</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Load embedding model
    with st.spinner("Loading model..."):
        embedding = load_embedding_model()

    # PDF Upload
    st.markdown("**📄 Upload a PDF**", unsafe_allow_html=False)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.db_source != uploaded_file.name:
            with st.spinner(f"Indexing `{uploaded_file.name}`…"):
                vs, n_chunks = build_vectorstore_from_pdf(
                    uploaded_file.read(), uploaded_file.name, embedding
                )
            st.session_state.vectorstore = vs
            st.session_state.db_source = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"✓ {n_chunks} chunks indexed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Default DB fallback
    st.markdown("**🗄 Default Knowledge Base**")
    if st.button("Load default DB", use_container_width=True):
        with st.spinner("Loading chromaDB…"):
            default_vs = load_default_vectorstore(embedding)
        if default_vs:
            st.session_state.vectorstore = default_vs
            st.session_state.db_source = "chromaDB (default)"
            st.session_state.chat_history = []
            st.success("✓ Default DB loaded")
        else:
            st.error("No chromaDB found. Run create_db.py first.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Status
    if st.session_state.vectorstore:
        st.markdown(
            f'<span class="badge badge-ready">● ready</span>&nbsp; '
            f'<span style="font-size:0.7rem;color:rgba(255,255,255,0.35);">{st.session_state.db_source}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-idle">○ no document loaded</span>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Show sources toggle
    show_sources = st.toggle("Show source chunks", value=False)


# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Ask your <span>documents</span>.</div>', unsafe_allow_html=True)
st.markdown('<div class="main-tagline">Semantic retrieval · Mistral · ChromaDB</div>', unsafe_allow_html=True)

# Chat history display
chat_html = '<div class="chat-container">'
for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        chat_html += f'''
        <div class="chat-bubble-user">
            <div class="chat-label chat-label-user">You</div>
            {turn["content"]}
        </div>'''
    else:
        chat_html += f'''
        <div class="chat-bubble-ai">
            <div class="chat-label chat-label-ai">DocMind</div>
            {turn["content"]}
        </div>'''
        if show_sources and "sources" in turn:
            for i, chunk in enumerate(turn["sources"], 1):
                preview = chunk.page_content[:200].replace("\n", " ")
                chat_html += f'<div class="source-chunk">⟨{i}⟩ {preview}…</div>'
chat_html += "</div>"

st.markdown(chat_html, unsafe_allow_html=True)

# Input row
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "", placeholder="Ask a question about your document…", label_visibility="collapsed", key="query_input"
    )
with col_btn:
    send = st.button("Send →", use_container_width=True)

# Handle send
if send and query.strip():
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload a PDF or load the default knowledge base first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        with st.spinner(""):
            answer, source_docs = get_answer(query.strip(), st.session_state.vectorstore)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs,
        })
        st.rerun()