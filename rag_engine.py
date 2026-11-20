"""
rag_engine.py - Core Retrieval-Augmented Generation (RAG) Engine
Project 7: AI Research Paper & Architecture Literature Assistant
Corpus: transformer_rag_research_paper_50p.pdf (50 pages)

Technical Responsibilities:
1. Ingestion: Page-by-page PDF extraction using pypdf (1-indexed).
2. Chunking: Boundary-aware splitting (~1000 chars, ~200 chars overlap) with metadata.
3. Embeddings: google-genai using text-embedding-004.
4. Persistence: Local ChromaDB persistence at ./chroma_db.
5. Strict Grounding: gemini-2.5-flash at temperature=0.0.
6. Exact [not found] Rule: Returns {"answer": "[not found]", "sources": []}
   whenever the question is unsupported by the retrieved corpus excerpts.
"""

import os
import sys
import time
import json
import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore")

# Suppress chromadb telemetry warnings and noisy logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
try:
    import chromadb.telemetry.product.posthog
    chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import pypdf
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Constants
DEFAULT_PDF_PATH = "./transformer_rag_research_paper_50p.pdf"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "project7_transformer_rag"
# Model configurations with backward & forward compatibility cascade
PRIMARY_EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_MODEL = PRIMARY_EMBEDDING_MODEL
FALLBACK_EMBEDDING_MODELS = ["gemini-embedding-001", "gemini-embedding-2"]

PRIMARY_GENERATION_MODEL = "gemini-2.5-flash"
GENERATION_MODEL = PRIMARY_GENERATION_MODEL
FALLBACK_GENERATION_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]

_ACTIVE_EMBEDDING_MODEL = None
_ACTIVE_GENERATION_MODEL = None

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def get_effective_embedding_model(client: genai.Client) -> str:
    """Detects and returns the active embedding model supported by the endpoint."""
    global _ACTIVE_EMBEDDING_MODEL
    if _ACTIVE_EMBEDDING_MODEL:
        return _ACTIVE_EMBEDDING_MODEL

    for m in [PRIMARY_EMBEDDING_MODEL] + FALLBACK_EMBEDDING_MODELS:
        try:
            client.models.embed_content(model=m, contents="probe")
            _ACTIVE_EMBEDDING_MODEL = m
            return m
        except Exception:
            continue
    _ACTIVE_EMBEDDING_MODEL = "gemini-embedding-001"
    return _ACTIVE_EMBEDDING_MODEL


def get_effective_generation_model(client: genai.Client) -> str:
    """Detects and returns the active generation model supported by the endpoint."""
    global _ACTIVE_GENERATION_MODEL
    if _ACTIVE_GENERATION_MODEL:
        return _ACTIVE_GENERATION_MODEL

    for m in [PRIMARY_GENERATION_MODEL] + FALLBACK_GENERATION_MODELS:
        try:
            client.models.generate_content(
                model=m,
                contents="probe",
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            _ACTIVE_GENERATION_MODEL = m
            return m
        except Exception:
            continue
    _ACTIVE_GENERATION_MODEL = "gemini-3.5-flash-lite"
    return _ACTIVE_GENERATION_MODEL

SYSTEM_INSTRUCTION = """You are an AI research paper & architecture literature assistant strictly grounded in the provided document excerpts from "transformer_rag_research_paper_50p.pdf".

CRITICAL GROUNDING RULES:
1. STRICT GROUNDING: You must answer the user's question using ONLY the provided document excerpts. You are strictly forbidden from using external knowledge, prior training data, assumptions, or general internet facts.
2. EXACT [not found] RULE: If the answer is NOT explicitly stated and directly supported by the provided excerpts, you MUST return EXACTLY:
{"answer": "[not found]", "sources": []}
Do NOT include any additional words, explanations, excuses, or citations when returning [not found].
3. FACTUAL CITATIONS: If the question IS explicitly answered in the excerpts:
   - Provide a factual, concise, and complete answer based strictly on the retrieved text.
   - In the "sources" list, include every excerpt that directly provided evidence for the answer, formatted exactly as "Page <page> (<chunk_id>)" (for example: ["Page 16 (p16_c1)"]).
4. FORMAT: Return your response strictly as valid JSON matching this schema:
{
  "answer": "<supported answer string or [not found]>",
  "sources": ["Page <page> (<chunk_id>)", ...]
}
Do NOT wrap the output in markdown codeblocks (no ```json). Output raw JSON only."""


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Check standard GOOGLE_API_KEY fallback
        api_key = os.getenv("GOOGLE_API_KEY")
    return api_key or ""


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialize Google GenAI client."""
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your .env file."
        )
    return genai.Client(api_key=key)


def extract_pdf_pages(pdf_path: str = DEFAULT_PDF_PATH) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from the PDF using 1-based indexing.
    Returns list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    resolved_path = Path(pdf_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Corpus PDF file not found at: {resolved_path}")

    reader = pypdf.PdfReader(str(resolved_path))
    total_pages = len(reader.pages)
    pages_data = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-indexed
        text = page.extract_text() or ""
        pages_data.append({
            "page": page_num,
            "text": text.strip()
        })

    return pages_data


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into meaningful overlapping chunks (~1000 chars, ~200 chars overlap)
    preserving sentence and paragraph boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)
    half_chunk = chunk_size // 2

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Look for natural boundaries backwards within [start + half_chunk, end]
        split_pos = -1

        # 1. Double newline (paragraph break)
        pos = text.rfind("\n\n", start + half_chunk, end)
        if pos != -1:
            split_pos = pos + 2
        else:
            # 2. Sentence terminators (. , ? , ! followed by whitespace/newline)
            for punct in [".\n", ". ", "?\n", "? ", "!\n", "! "]:
                p = text.rfind(punct, start + half_chunk, end)
                if p != -1:
                    split_pos = p + len(punct)
                    break

            if split_pos == -1:
                # 3. Single newline
                p = text.rfind("\n", start + half_chunk, end)
                if p != -1:
                    split_pos = p + 1
                else:
                    # 4. Space
                    p = text.rfind(" ", start + half_chunk, end)
                    if p != -1:
                        split_pos = p + 1
                    else:
                        # 5. Hard boundary
                        split_pos = end

        chunk_content = text[start:split_pos].strip()
        if chunk_content:
            chunks.append(chunk_content)

        # Advance window with overlap
        start = max(start + 1, split_pos - overlap)

    return chunks


def ingest_and_chunk_pdf(pdf_path: str = DEFAULT_PDF_PATH) -> List[Dict[str, Any]]:
    """
    Ingests the PDF and generates structured chunks with required metadata:
    {"page": <int>, "chunk_id": "p<page>_c<index>"}
    """
    pages_data = extract_pdf_pages(pdf_path)
    all_chunks = []

    for page_item in pages_data:
        page_num = page_item["page"]
        raw_text = page_item["text"]
        chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for c_idx, content in enumerate(chunks, start=1):
            chunk_id = f"p{page_num}_c{c_idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": content,
                "metadata": {
                    "page": page_num,
                    "chunk_id": chunk_id
                }
            })

    return all_chunks


def get_chroma_collection(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    """Initializes or connects to local persistent ChromaDB collection."""
    os.makedirs(persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, is_persistent=True)
    )
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_batch_with_retry(
    client: genai.Client,
    model_name: str,
    texts: List[str],
    max_retries: int = 5
) -> List[List[float]]:
    """
    Embeds a batch of texts with automatic retry and rate-limit backoff.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=texts,
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                match = re.search(r"retry in (\d+)", err_str)
                wait_seconds = (int(match.group(1)) + 5) if match else 55
                time.sleep(wait_seconds)
            elif attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to embed batch after {max_retries} attempts: {e}")
    raise RuntimeError("Failed to embed batch.")


def build_or_load_index(
    pdf_path: str = DEFAULT_PDF_PATH,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
    client: Optional[genai.Client] = None
) -> chromadb.Collection:
    """
    Checks if collection already exists and has documents (collection.count() > 0).
    If populated and not force_rebuild, loads directly from disk without regenerating embeddings.
    Otherwise, extracts PDF, chunks, computes embeddings, and stores in ChromaDB.
    """
    collection = get_chroma_collection(persist_dir, collection_name)
    existing_count = collection.count()

    if existing_count > 0 and not force_rebuild:
        return collection

    # Needs rebuild / population
    if force_rebuild and existing_count > 0:
        all_data = collection.get()
        all_ids = all_data.get("ids", [])
        if all_ids:
            collection.delete(ids=all_ids)

    if client is None:
        client = get_genai_client()

    chunks = ingest_and_chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("No text chunks extracted from PDF.")

    model_name = get_effective_embedding_model(client)

    # Check which chunks are already stored (for incremental resume)
    existing_ids = set(collection.get().get("ids", []))
    pending_chunks = [c for c in chunks if c["id"] not in existing_ids]

    batch_size = 25
    for i in range(0, len(pending_chunks), batch_size):
        b_chunks = pending_chunks[i:i + batch_size]
        b_ids = [c["id"] for c in b_chunks]
        b_texts = [c["text"] for c in b_chunks]
        b_metadatas = [c["metadata"] for c in b_chunks]

        # Embed with retry
        b_embeddings = embed_batch_with_retry(
            client=client,
            model_name=model_name,
            texts=b_texts
        )

        # Immediately add this batch to ChromaDB
        collection.add(
            ids=b_ids,
            embeddings=b_embeddings,
            documents=b_texts,
            metadatas=b_metadatas
        )
        time.sleep(0.5)

    return collection


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
    collection: Optional[chromadb.Collection] = None,
    client: Optional[genai.Client] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant chunks from ChromaDB for a given question.
    """
    if client is None:
        client = get_genai_client()

    if collection is None:
        collection = build_or_load_index(client=client)

    model_name = get_effective_embedding_model(client)

    # Embed query
    query_res = client.models.embed_content(
        model=model_name,
        contents=question,
    )
    query_embedding = query_res.embeddings[0].values

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        chunk_ids = results["ids"][0] if results.get("ids") else [m.get("chunk_id", "") for m in metas]

        for doc, meta, dist, cid in zip(docs, metas, dists, chunk_ids):
            retrieved.append({
                "chunk_id": cid,
                "page": meta.get("page", 1),
                "text": doc,
                "distance": dist,
                "source_tag": f"Page {meta.get('page', 1)} ({cid})"
            })

    return retrieved


def clean_llm_json_output(raw_output: str) -> dict:
    """
    Parses LLM output into JSON, stripping code blocks if present.
    """
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Check if the output explicitly says [not found]
        if "[not found]" in cleaned.lower():
            return {"answer": "[not found]", "sources": []}
        return {"answer": cleaned, "sources": []}


def answer(question: str) -> dict:
    """
    MANDATORY TOP-LEVEL FUNCTION SIGNATURE:
    answer(question: str) -> dict
    Returns {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}

    Strict Grounding & Exact [not found] Rule:
    - Strictly forbids external knowledge, assumptions, or internet facts.
    - If unsupported by retrieved excerpts, returns {"answer": "[not found]", "sources": []}.
    - No extra words, explanations, or citations when returning [not found].
    - For supported answers, sources contains relevant page numbers and chunk IDs.
    """
    client = get_genai_client()
    collection = build_or_load_index(client=client)

    # Retrieve top-k relevant chunks
    retrieved_chunks = retrieve_context(
        question=question,
        top_k=TOP_K,
        collection=collection,
        client=client
    )

    if not retrieved_chunks:
        return {"answer": "[not found]", "sources": []}

    # Format retrieved context
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"--- START EXCERPT: {c['source_tag']} ---\n{c['text']}\n--- END EXCERPT ---"
        )
    formatted_context = "\n\n".join(context_blocks)

    user_prompt = f"""USER QUESTION:
{question}

RETRIEVED EXCERPTS FROM CORPUS (transformer_rag_research_paper_50p.pdf):
{formatted_context}

Provide your response in accordance with the Grounding Rules. If the question cannot be answered solely from the excerpts above, return {{"answer": "[not found]", "sources": []}}."""

    gen_model = get_effective_generation_model(client)

    try:
        response = client.models.generate_content(
            model=gen_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=SYSTEM_INSTRUCTION,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
        )
        parsed = clean_llm_json_output(response.text)
    except Exception as e:
        # If generation fails due to strict rules or client error
        return {"answer": "[not found]", "sources": []}

    ans_str = str(parsed.get("answer", "")).strip()
    sources_list = parsed.get("sources", [])

    # Strict normalization of [not found]
    if (
        ans_str.lower() == "[not found]"
        or "[not found]" in ans_str.lower()
        or not ans_str
    ):
        return {"answer": "[not found]", "sources": []}

    # Ensure sources is a valid list of formatted strings
    if not isinstance(sources_list, list):
        sources_list = [str(sources_list)]

    # If the model produced a valid answer but empty or unformatted sources,
    # match against the retrieved chunks
    cleaned_sources = []
    for s in sources_list:
        s_str = str(s).strip()
        if s_str:
            cleaned_sources.append(s_str)

    if not cleaned_sources and retrieved_chunks:
        # Use top retrieved chunk as fallback source
        cleaned_sources = [retrieved_chunks[0]["source_tag"]]

    return {
        "answer": ans_str,
        "sources": cleaned_sources
    }


def get_status() -> dict:
    """Returns database and corpus status for UI display."""
    pdf_exists = os.path.exists(DEFAULT_PDF_PATH)
    page_count = 0
    if pdf_exists:
        try:
            reader = pypdf.PdfReader(DEFAULT_PDF_PATH)
            page_count = len(reader.pages)
        except Exception:
            page_count = 0

    chroma_exists = os.path.exists(CHROMA_PERSIST_DIR)
    chunk_count = 0
    if chroma_exists:
        try:
            col = get_chroma_collection()
            chunk_count = col.count()
        except Exception:
            chunk_count = 0

    return {
        "pdf_path": DEFAULT_PDF_PATH,
        "pdf_exists": pdf_exists,
        "page_count": page_count,
        "chroma_dir": CHROMA_PERSIST_DIR,
        "chroma_exists": chroma_exists,
        "indexed_chunks": chunk_count,
        "collection_name": COLLECTION_NAME,
        "has_api_key": bool(get_gemini_api_key())
    }
