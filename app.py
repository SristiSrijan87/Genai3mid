"""
app.py - Streamlit Web UI for Project 7: AI Research Paper & Architecture Literature Assistant
Corpus: transformer_rag_research_paper_50p.pdf (50 pages)

Features:
- Academic dark/light compatible styling.
- Corpus and Vector DB persistence status sidebar with Rebuild Index trigger.
- Interactive Query Assistant with quick-select buttons for Q1–Q5.
- Structured Answer Card featuring a high-visibility badge for [not found].
- Collapsible source inspection cards detailing Page number, Chunk ID, Similarity Distance, and Raw Text.
- One-click Full Assessment Suite (Q1–Q5) runner with evaluation table for submission screenshots.
"""

import os
import time
import json
import streamlit as st
import pandas as pd
from pathlib import Path

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Project 7: AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import RAG Engine functions
from rag_engine import (
    answer,
    retrieve_context,
    build_or_load_index,
    get_status,
    get_gemini_api_key,
    DEFAULT_PDF_PATH,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_MODEL
)
from run_assessment import ASSESSMENT_QUESTIONS

# Custom CSS for Academic Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: rgba(30, 144, 255, 0.08);
        border-left: 4px solid #1E90FF;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .badge-not-found {
        display: inline-block;
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.0rem;
        margin-bottom: 10px;
    }
    .badge-supported {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.0rem;
        margin-bottom: 10px;
    }
    .answer-box {
        background-color: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-tag-chip {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        border-radius: 12px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .metric-badge {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "current_response" not in st.session_state:
    st.session_state["current_response"] = None
if "current_sources_detail" not in st.session_state:
    st.session_state["current_sources_detail"] = []
if "assessment_results" not in st.session_state:
    st.session_state["assessment_results"] = None


# Sidebar Configuration & Corpus Status
with st.sidebar:
    st.image("https://ssl.gstatic.com/docs/doclist/images/drive_favicon_2026_32dp.png", width=36)
    st.title("Project 7: RAG System")
    st.caption("AI Research Paper & Architecture Literature Assistant")
    st.divider()

    st.subheader("📁 Corpus & Persistence Status")
    status = get_status()

    st.write(f"**Document:** `{os.path.basename(DEFAULT_PDF_PATH)}`")
    if status["pdf_exists"]:
        st.success(f"✓ PDF Verified ({status['page_count']} Pages)")
    else:
        st.error("✗ PDF Not Found!")

    st.write(f"**Vector Store:** `{CHROMA_PERSIST_DIR}`")
    st.write(f"**Collection:** `{COLLECTION_NAME}`")
    st.write(f"**Indexed Chunks:** `{status['indexed_chunks']}`")

    # API Key check
    api_key = get_gemini_api_key()
    if api_key:
        st.success("✓ `GEMINI_API_KEY` Configured")
    else:
        st.warning("⚠️ `GEMINI_API_KEY` Missing in .env")
        key_input = st.text_input("Enter GEMINI_API_KEY", type="password")
        if key_input:
            os.environ["GEMINI_API_KEY"] = key_input
            st.rerun()

    st.divider()
    st.subheader("⚙️ System Architecture")
    st.write(f"• **LLM:** `{GENERATION_MODEL}` (temp=0.0)")
    st.write(f"• **Embeddings:** `{EMBEDDING_MODEL}`")
    st.write("• **Top-K Retrieval:** 5 Chunks")
    st.write("• **Grounding:** Strict (Zero Hallucination)")

    st.divider()
    if st.button("🔄 Rebuild ChromaDB Index", use_container_width=True):
        if not get_gemini_api_key():
            st.error("Cannot rebuild index: GEMINI_API_KEY is required.")
        else:
            with st.spinner("Re-indexing PDF corpus into ChromaDB..."):
                try:
                    col = build_or_load_index(force_rebuild=True)
                    st.success(f"✓ Re-indexed {col.count()} chunks successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error rebuilding index: {e}")


# Main Workspace Header
st.markdown('<div class="main-title">📚 AI Research Paper & Architecture Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Querying <i>Hybrid Contextual Retrieval & Transformer Scaling Laws: Technical Monograph</i> (50 Pages) with zero-hallucination guarantee.</div>',
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Interactive Query Assistant",
    "📊 Full Assessment Suite (Q1–Q5)",
    "📄 Corpus & Architecture Overview"
])


# TAB 1: INTERACTIVE QUERY ASSISTANT
with tab1:
    st.write("### Ask a Question")
    st.write("Type your research or architecture question below, or select a preset assessment question:")

    # Quick Select Buttons
    st.markdown("**Quick-Select Presets:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Q1: Hyperparameters", use_container_width=True):
            st.session_state["query_input"] = ASSESSMENT_QUESTIONS[0]["question"]
    with col2:
        if st.button("Q2: CEO OpenAI", use_container_width=True):
            st.session_state["query_input"] = ASSESSMENT_QUESTIONS[1]["question"]
    with col3:
        if st.button("Q3: EM Accuracy", use_container_width=True):
            st.session_state["query_input"] = ASSESSMENT_QUESTIONS[2]["question"]
    with col4:
        if st.button("Q4: GPU Cluster", use_container_width=True):
            st.session_state["query_input"] = ASSESSMENT_QUESTIONS[3]["question"]
    with col5:
        if st.button("Q5: Oppenheimer", use_container_width=True):
            st.session_state["query_input"] = ASSESSMENT_QUESTIONS[4]["question"]

    user_query = st.text_area(
        "Question Input:",
        value=st.session_state["query_input"],
        height=90,
        placeholder="e.g. What GPU cluster configuration was used to train the 13B model?"
    )

    query_button = st.button("🚀 Query RAG Assistant", type="primary", use_container_width=False)

    if query_button and user_query.strip():
        if not get_gemini_api_key():
            st.error("Please configure your `GEMINI_API_KEY` in `.env` to execute queries.")
        else:
            with st.spinner("Searching persistent vector index & generating strictly grounded response..."):
                t_start = time.time()
                try:
                    res = answer(user_query.strip())
                    # Also fetch detailed chunks for UI inspection
                    detailed_chunks = retrieve_context(user_query.strip(), top_k=5)
                    st.session_state["current_response"] = res
                    st.session_state["current_sources_detail"] = detailed_chunks
                    st.session_state["current_latency"] = round(time.time() - t_start, 2)
                except Exception as ex:
                    st.error(f"Error executing RAG query: {ex}")

    # Display Answer Card
    if st.session_state.get("current_response"):
        resp = st.session_state["current_response"]
        ans_text = resp.get("answer", "")
        sources = resp.get("sources", [])
        latency = st.session_state.get("current_latency", 0.0)

        is_not_found = ans_text.strip().lower() == "[not found]"

        st.markdown("---")
        st.subheader("💡 Generated Response")

        if is_not_found:
            st.markdown(
                '<div class="badge-not-found">🚫 [not found] &bull; Strictly Refused Out-of-Domain Query</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div class="answer-box" style="border-left: 5px solid #c62828;">
                <b>Answer:</b> <code>[not found]</code><br/><br/>
                <small style="color: #666;">
                <b>Strict Grounding Enforcement:</b> The retrieved excerpts from <code>transformer_rag_research_paper_50p.pdf</code> do not contain facts to support this question. In accordance with zero-hallucination specifications, the model strictly abstained from using external training memory.
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="badge-supported">✓ Context Supported &bull; Grounded in Research Corpus</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div class="answer-box" style="border-left: 5px solid #2e7d32;">
                <b>Answer:</b><br/>
                {ans_text}
            </div>
            """, unsafe_allow_html=True)

            if sources:
                st.markdown("**Cited Sources:**")
                source_html = "".join([f'<span class="source-tag-chip">📄 {s}</span>' for s in sources])
                st.markdown(source_html, unsafe_allow_html=True)

        st.caption(f"⚡ Retrieval + Inference Latency: {latency}s | Model: gemini-2.5-flash (temperature=0.0)")

        # Expandable Source Chunks Inspector
        st.markdown("### 🔎 Retrieved Context Inspection (Top-5 Chunks)")
        detailed = st.session_state.get("current_sources_detail", [])
        if detailed:
            for i, chunk in enumerate(detailed, start=1):
                with st.expander(f"Chunk {i}: Page {chunk['page']} ({chunk['chunk_id']}) — Cosine Distance: {chunk['distance']:.4f}"):
                    st.markdown(f"**Metadata:** Page `{chunk['page']}` | Chunk ID: `{chunk['chunk_id']}` | Distance: `{chunk['distance']:.4f}`")
                    st.text_area("Extracted Chunk Text:", value=chunk["text"], height=160, key=f"chunk_disp_{i}")
        else:
            st.info("No chunk details available.")


# TAB 2: FULL ASSESSMENT SUITE
with tab2:
    st.subheader("📋 Q1–Q5 Evaluation Benchmark Suite")
    st.write(
        "This evaluation suite runs all 5 mandatory assessment questions to verify both in-domain fact extraction "
        "and strict `[not found]` refusal for out-of-domain queries."
    )

    run_suite = st.button("▶ Run Full Assessment (Q1–Q5)", type="primary")

    if run_suite:
        if not get_gemini_api_key():
            st.error("Please configure your `GEMINI_API_KEY` in `.env` before running the assessment suite.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            suite_results = []
            total = len(ASSESSMENT_QUESTIONS)

            for i, q_item in enumerate(ASSESSMENT_QUESTIONS, start=1):
                status_text.text(f"Evaluating {q_item['id']}: {q_item['question'][:60]}...")
                t0 = time.time()
                res = answer(q_item["question"])
                latency = round(time.time() - t0, 2)

                actual_ans = res.get("answer", "")
                sources = res.get("sources", [])
                is_nf = actual_ans.strip().lower() == "[not found]"

                if q_item["expected_grounding"] == "[not found]":
                    passed = is_nf and len(sources) == 0
                    stat = "✅ PASS ([not found])" if passed else "❌ FAIL (Hallucinated)"
                else:
                    passed = not is_nf and len(sources) > 0
                    stat = "✅ PASS (Supported)" if passed else "❌ FAIL (Missed info)"

                suite_results.append({
                    "ID": q_item["id"],
                    "Category": q_item["type"],
                    "Question": q_item["question"],
                    "Grounding Status": stat,
                    "Actual Answer": actual_ans,
                    "Sources": ", ".join(sources) if sources else "None",
                    "Latency": f"{latency}s",
                    "Passed": passed
                })
                progress_bar.progress(i / total)

            status_text.success("✓ Assessment suite completed!")
            st.session_state["assessment_results"] = suite_results

    # Display Assessment Results Table
    if st.session_state.get("assessment_results"):
        data = st.session_state["assessment_results"]
        pass_count = sum(1 for r in data if r["Passed"])
        
        st.markdown(f"### Assessment Score: **{pass_count}/5 ({pass_count * 20}%)**")
        
        # Display Table
        df = pd.DataFrame(data)[["ID", "Category", "Question", "Grounding Status", "Actual Answer", "Sources", "Latency"]]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### Detailed Question Breakdown")
        for r in data:
            with st.expander(f"{r['ID']}: {r['Question']} — {r['Grounding Status']}"):
                st.write(f"**Question:** {r['Question']}")
                st.write(f"**Category:** {r['Category']}")
                st.write(f"**Answer:** {r['Actual Answer']}")
                st.write(f"**Sources:** `{r['Sources']}`")
                st.write(f"**Latency:** {r['Latency']}")


# TAB 3: CORPUS & ARCHITECTURE
with tab3:
    st.subheader("📖 Monograph & System Documentation")
    st.markdown("""
    ### Corpus Details
    - **Title:** *Hybrid Contextual Retrieval & Transformer Scaling Laws: Technical Monograph*
    - **Total Extracted Pages:** 50 pages
    - **Format:** PDF document (Version 1.4)
    - **Chunking Parameters:** Window size ~1000 characters with ~200 characters overlap.
    - **Index Engine:** ChromaDB persistent storage at `./chroma_db` (collection: `project7_transformer_rag`).

    ### Grounding & Zero-Hallucination Guardrails
    1. **Strict Context Boundary:** External model knowledge is forbidden through strict system prompts and temperature=0.0.
    2. **Exact Fallback:** Any query without explicit, verifiable proof in the retrieved top-5 chunks returns:
       ```json
       {"answer": "[not found]", "sources": []}
       ```
    3. **Grounding Checks (Q2 & Q5):** OpenAI leadership and film box office receipts are completely absent from the monograph and properly trigger `[not found]`.
    """)

