"""
RAG Lite - Interactive Document QA & Semantic Search Cockpit (Streamlit).
Explore vector retrieval, test similarity thresholds, and inspect chunk attribution.
"""

import streamlit as st
from pathlib import Path

from rag_lite.engine import RAGPipeline
from rag_lite.providers import GeminiGenerator, OpenAIGenerator, MockGenerator
from rag_lite.models import RAGResponse

st.set_page_config(
    page_title="RAG Lite | Grounded Document QA",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RAG Lite: Grounded Document QA & Semantic Search")
st.markdown(
    "Query technical documentation with dense vector retrieval, source chunk attribution, "
    "and **automated abstention guardrails** to prevent hallucinations."
)

# Sidebar Configuration
st.sidebar.header("⚙️ RAG Engine Parameters")
top_k = st.sidebar.slider("Top-k Context Chunks", min_value=1, max_value=5, value=3)
threshold = st.sidebar.slider(
    "Abstention Threshold (Cosine Similarity)",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05,
    help="Queries with similarity below this threshold trigger automatic abstention to eliminate hallucination."
)

provider_choice = st.sidebar.selectbox(
    "Generation Backend",
    ["Mock (Offline Demo)", "Gemini 2.5 Flash", "OpenAI (GPT-4o-mini)"],
    index=0
)

api_key = ""
if "Gemini" in provider_choice:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get free key at aistudio.google.com")
elif "OpenAI" in provider_choice:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.divider()
st.sidebar.subheader("📂 Custom Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload .md or .txt document", type=["md", "txt"])

# Initialize RAG Pipeline in session state
@st.cache_resource
def get_pipeline():
    p = RAGPipeline(default_score_threshold=0.40)
    kb_path = Path(__file__).resolve().parent / "knowledge_base"
    if kb_path.exists():
        p.ingest_directory(kb_path)
    return p

pipeline = get_pipeline()

# Handle custom document upload
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    title = uploaded_file.name.replace(".md", "").replace(".txt", "").replace("_", " ").title()
    chunks_added = pipeline.ingest_text(title=title, text=content, doc_id=uploaded_file.name)
    st.sidebar.success(f"✓ Ingested '{title}' ({chunks_added} chunks)")

# Knowledge Base Overview
st.sidebar.divider()
docs = pipeline.vector_store.list_documents()
st.sidebar.caption(f"**Indexed Documents ({len(docs)}):**")
for d in docs:
    st.sidebar.markdown(f"- `{d}`")
st.sidebar.caption(f"**Total Chunks in Vector Store:** `{pipeline.vector_store.count()}`")

# Sample questions
st.subheader("💡 Try Sample Questions")
col1, col2, col3, col4 = st.columns(4)

selected_question = None
with col1:
    if st.button("💳 Stripe Payment Failures", use_container_width=True):
        selected_question = "What happens when an invoice payment fails in Stripe?"
with col2:
    if st.button("🚀 Canary Rollback Threshold", use_container_width=True):
        selected_question = "What is our automated rollback threshold for Kubernetes canary deployments?"
with col3:
    if st.button("🚨 P0 Outage SLA", use_container_width=True):
        selected_question = "What is the response SLA and escalation protocol for a P0 critical outage?"
with col4:
    if st.button("🐕 Office Pet Policy (Out of Domain)", use_container_width=True):
        selected_question = "What is our company policy on bringing pet dogs to the office?"

# Input box
query_input = st.text_input(
    "Ask a question based on the technical documentation:",
    value=selected_question or "",
    placeholder="e.g. How does Stripe handle mid-cycle subscription downgrades?"
)

if st.button("🔍 Search & Answer", type="primary") or selected_question:
    if not query_input.strip():
        st.warning("Please enter a question.")
    else:
        # Configure generator
        if "Gemini" in provider_choice and api_key:
            pipeline.generator = GeminiGenerator(api_key=api_key)
        elif "OpenAI" in provider_choice and api_key:
            pipeline.generator = OpenAIGenerator(api_key=api_key)
        else:
            pipeline.generator = MockGenerator()

        with st.spinner("Embedding query & searching vector space..."):
            response: RAGResponse = pipeline.query(
                question=query_input,
                top_k=top_k,
                score_threshold=threshold
            )
            st.session_state["rag_response"] = response

# Render Response
if "rag_response" in st.session_state:
    resp: RAGResponse = st.session_state["rag_response"]
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Pipeline Latency", f"{resp.latency_ms} ms")
    m2.metric("Top Cosine Similarity", f"{resp.confidence_score:.4f}")
    m3.metric("Status", "Abstained (Guardrail)" if resp.is_abstained else "Answered")

    if resp.is_abstained:
        st.error(f"**🛡️ Abstention Guardrail Triggered:**\n\n{resp.answer}")
        st.info("💡 **Why did this happen?** The query had no semantic match above your selected threshold in the knowledge base. Rather than hallucinating an answer, RAG Lite gracefully declined.")
    else:
        st.success(f"### 💬 Grounded Answer\n\n{resp.answer}")

        st.subheader("📚 Source Attribution & Retrieved Chunks")
        for idx, src in enumerate(resp.sources, 1):
            score_badge = f"🎯 Similarity: **{src.score:.4f}**"
            with st.expander(f"Chunk #{idx}: {src.chunk.document_title} ({src.chunk.chunk_id}) | {score_badge}", expanded=True):
                st.caption(f"Document: `{src.chunk.document_title}` | Character Range: `{src.chunk.start_char}` to `{src.chunk.end_char}`")
                st.markdown(f"> \"{src.chunk.text}\"")
