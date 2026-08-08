import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import base64
import re
import os
import tempfile
import pypdf
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import uuid

# ──────────────────────────────────────────────────────────────────────────────
# DOCLING (optional) — local document parser, no API key needed.
# Used instead of PyPDFLoader for PDF/DOCX/PPTX/HTML when installed.
# If it's missing or a conversion fails, the app falls back to the
# original PyPDFLoader / TextLoader path automatically.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

DOCLING_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #1a1a1a; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #2d2d2d; }
    .stTextInput textarea { color: #ffffff !important; }
    .think-output { color: #ffcc00; font-style: italic; }
    .actual-output { color: #00ff00; font-weight: bold; }
    .stSelectbox div[data-baseweb="select"] { color: white !important; background-color: #3d3d3d !important; }
    .stSelectbox svg { fill: white !important; }
    .stSelectbox option { background-color: #2d2d2d !important; color: white !important; }
    div[role="listbox"] div { background-color: #2d2d2d !important; color: white !important; }
    .user-image { max-width: 300px; margin-top: 10px; }
    .warning-box { background-color: #ff6b6b; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .performance-badge { background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; display: inline-block; margin: 5px 0; }
    .speed-badge { background-color: #2196F3; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; display: inline-block; margin-left: 10px; }
    .chunk-badge { background-color: #9C27B0; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; display: inline-block; margin: 5px 2px; }
    .docling-badge { background-color: #FF9800; color: white; padding: 3px 8px; border-radius: 3px; font-size: 11px; display: inline-block; margin: 5px 2px; }
</style>
""", unsafe_allow_html=True)

st.caption("🚀 Your Local Multimodal AI Pair Programmer - Optimized for Speed + Smart Chunking!")
st.markdown(
    '<span class="performance-badge">⚡ Parallel RAG + Optimized Streaming</span>'
    '<span class="chunk-badge">🧩 Semantic Chunking + BM25 Hybrid</span>'
    + (
        '<span class="docling-badge">📄 Docling Parsing (local, no API key)</span>'
        if DOCLING_AVAILABLE else
        '<span class="docling-badge">📄 Basic Parsing (Docling not installed)</span>'
    ),
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "all_chunks": [],          # flat list of all child chunks (for BM25)
    "parent_map": {},          # chunk_id → parent Document
    "documents_processed": False,
    "page_warnings": [],
    "generation_stats": {"tokens": 0, "time": 0.0, "tps": 0.0},
    "message_log": [
        {"role": "ai", "content": "Hi! I'm your Local AI Code Assistant 💻📚👁️ (Smart Chunking Enabled!)"}
    ],
    "chunking_stats": {}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    ollama_base_url = st.text_input(
        "Ollama Server URL", value="http://localhost:11434",
        help="Default local Ollama server address", key="ollama_url"
    )

    selected_model = st.selectbox(
        "Choose Vision Model",
        ["qwen3-vl:2b", "ministral-3:3b"],
        help="Q4 models are 2-3x faster with minimal quality loss",
        key="model_select"
    )

    st.divider()

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        help="Lower = more focused, Higher = more creative", key="temp_slider"
    )

    # ── Chunking Settings ──────────────────────────────────────────────────
    st.markdown("### 🧩 Chunking Strategy")

    chunking_strategy = st.selectbox(
        "Strategy",
        ["Semantic (Recommended)", "Recursive (Fast)", "Hybrid (Semantic + Recursive)"],
        index=0,
        help=(
            "Semantic: splits on topic shifts using embeddings — best retrieval quality.\n"
            "Recursive: splits on paragraph/sentence boundaries — fast.\n"
            "Hybrid: semantic for large docs, recursive for small ones."
        ),
        key="chunking_strategy"
    )

    semantic_threshold_type = st.selectbox(
        "Semantic Breakpoint Method",
        ["percentile", "standard_deviation", "interquartile"],
        index=0,
        help=(
            "percentile: split when distance > Nth percentile of all distances.\n"
            "standard_deviation: split when distance > mean + N * std.\n"
            "interquartile: robust to outliers, good for noisy docs."
        ),
        key="semantic_threshold_type"
    )

    semantic_threshold_amount = st.slider(
        "Breakpoint Threshold",
        min_value=50, max_value=99, value=90, step=1,
        help="Higher = fewer, larger chunks. Lower = more, smaller chunks.",
        key="semantic_threshold_amount"
    )

    # Parent chunk (larger context given to LLM)
    parent_chunk_size = st.number_input(
        "Parent Chunk Size (tokens)",
        min_value=200, max_value=2000, value=800, step=100,
        help="Larger chunk returned to the LLM for rich context.",
        key="parent_chunk_size"
    )

    # Child chunk (smaller unit used for embedding/search)
    child_chunk_size = st.number_input(
        "Child Chunk Size (tokens)",
        min_value=50, max_value=500, value=200, step=50,
        help="Smaller chunk embedded for precise retrieval. Must be < Parent.",
        key="child_chunk_size"
    )

    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0, max_value=200, value=40, step=10,
        help="Shared tokens between adjacent chunks to preserve boundary context.",
        key="chunk_overlap"
    )

    st.markdown("### 🔍 Retrieval Settings")

    retrieval_k = st.slider(
        "Top-K chunks to retrieve", min_value=2, max_value=12, value=5, step=1,
        help="How many chunks are fetched per query.",
        key="retrieval_k"
    )

    mmr_fetch_k = st.slider(
        "MMR Fetch-K pool", min_value=5, max_value=40, value=15, step=5,
        help="Candidate pool before MMR diversity filtering. Should be > Top-K.",
        key="mmr_fetch_k"
    )

    mmr_lambda = st.slider(
        "MMR Lambda (relevance ↔ diversity)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="1.0 = pure relevance, 0.0 = maximum diversity.",
        key="mmr_lambda"
    )

    bm25_weight = st.slider(
        "BM25 weight in hybrid retrieval",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Weight of keyword (BM25) retriever. Semantic weight = 1 - BM25 weight.",
        key="bm25_weight"
    )

    use_parent_child = st.checkbox(
        "Parent-Child Retrieval",
        value=True,
        help=(
            "Embed small child chunks for precise search, "
            "but return large parent chunks to the LLM for richer context."
        ),
        key="use_parent_child"
    )

    st.divider()

    with st.expander("🚀 Advanced Performance Settings"):
        st.markdown("**Inference Optimization**")
        num_ctx = st.number_input(
            "Context Window Size", min_value=512, max_value=32768, value=4096, step=512,
            key="num_ctx_input"
        )
        num_batch = st.number_input(
            "Batch Size", min_value=128, max_value=1024, value=512, step=128,
            key="num_batch_input"
        )
        num_thread = st.number_input(
            "CPU Threads", min_value=1, max_value=16, value=4, step=1,
            key="num_thread_input"
        )
        keep_alive_time = st.selectbox(
            "Model Keep-Alive Time",
            options=["1m", "5m", "10m", "30m", "1h"], index=2,
            key="keep_alive_select"
        )
        use_mmap = st.checkbox("Memory Mapping (mmap)", value=True, key="mmap_checkbox")
        use_mlock = st.checkbox("Lock in RAM (mlock)", value=False, key="mlock_checkbox")
        enable_parallel_rag = st.checkbox(
            "Parallel RAG Retrieval", value=True,
            help="Fetch documents while model loads (faster)", key="parallel_rag_checkbox"
        )

    st.divider()
    st.markdown("### 📚 Document Upload")

    if st.session_state.vectorstore is not None:
        st.success("✅ RAG Active")
        if st.session_state.chunking_stats:
            cs = st.session_state.chunking_stats
            st.info(
                f"📊 {cs.get('parent_chunks', 0)} parent chunks │ "
                f"{cs.get('child_chunks', 0)} child chunks indexed"
            )
        if st.button("🗑️ Clear Documents & Free Memory", key="clear_docs_btn"):
            st.session_state.documents_processed = False
            st.session_state.page_warnings = []
            st.session_state.vectorstore = None
            st.session_state.all_chunks = []
            st.session_state.parent_map = {}
            st.session_state.chunking_stats = {}
            st.rerun()
    else:
        st.info("ℹ️ No documents loaded")

    st.warning("⚠️ PDFs larger than 10 pages will be truncated!")

    # File types depend on whether Docling is installed: with Docling we can
    # also accept DOCX / PPTX / HTML, since Docling parses those locally too.
    allowed_doc_types = ["pdf", "txt"]
    uploader_label = "Upload Documents for RAG (PDF/TXT)"
    if DOCLING_AVAILABLE:
        allowed_doc_types += ["docx", "pptx", "html", "htm"]
        uploader_label = "Upload Documents for RAG (PDF/TXT/DOCX/PPTX/HTML - via Docling)"
    else:
        st.caption("ℹ️ Install `docling` to also upload DOCX, PPTX and HTML files.")

    uploaded_docs = st.file_uploader(
        uploader_label,
        type=allowed_doc_types, accept_multiple_files=True, key="doc_uploader"
    )

    st.markdown("### 🖼️ Image Upload")
    uploaded_file = st.file_uploader(
        "Upload an image (optional)",
        type=["jpg", "png", "jpeg"], key="image_uploader"
    )

    st.divider()
    st.markdown("### Quick Setup")
    st.code("""
# Install vision model
ollama pull qwen3-vl:2b

# Install embedding model (required for RAG)
ollama pull nomic-embed-text

# Optional: local document parsing for PDF/DOCX/PPTX/HTML
pip install docling

# Start server
ollama serve
    """, language="bash")

# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS (lazy-loaded, cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings(base_url):
    """Load and cache nomic-embed-text embeddings."""
    try:
        return OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        st.info("Run: ollama pull nomic-embed-text")
        return None


@st.cache_resource
def get_docling_converter():
    """
    Load and cache a single Docling DocumentConverter.
    Loading Docling's layout/table models is the slow part, so this is
    done once per Streamlit session, not on every file upload.
    """
    if not DOCLING_AVAILABLE:
        return None
    return DocumentConverter()


# ──────────────────────────────────────────────────────────────────────────────
# CHUNKING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_semantic_splitter(embeddings, threshold_type, threshold_amount):
    """
    SemanticChunker splits at topic-shift boundaries detected via
    cosine distance between adjacent sentence embeddings.
    """
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )


def build_recursive_splitter(chunk_size, overlap):
    """
    RecursiveCharacterTextSplitter tries paragraph → sentence → word
    boundaries before falling back to raw characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def create_parent_child_chunks(parent_docs, child_chunk_size, overlap):
    """
    Parent-Child strategy:
    - Parent docs are stored in a lookup map (returned to LLM for context).
    - Child chunks (smaller) are embedded and indexed for precise retrieval.
    - Each child carries a 'parent_id' metadata key linking back to its parent.
    """
    child_splitter = build_recursive_splitter(child_chunk_size, overlap)
    child_chunks = []
    parent_map = {}

    for parent_doc in parent_docs:
        parent_id = str(uuid.uuid4())
        parent_doc.metadata["doc_id"] = parent_id
        parent_map[parent_id] = parent_doc

        children = child_splitter.split_documents([parent_doc])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["source"] = parent_doc.metadata.get("source", "unknown")
            child_chunks.append(child)

    return child_chunks, parent_map


def chunk_documents(
    raw_docs,
    embeddings,
    strategy,
    threshold_type,
    threshold_amount,
    parent_chunk_size,
    child_chunk_size,
    overlap,
    use_parent_child,
):
    """
    Main chunking pipeline. Returns:
        child_chunks  : list[Document] – embedded and indexed in FAISS
        parent_map    : dict[str, Document] – id → parent doc (for LLM context)
    """
    # ── Step 1: Create parent (large) chunks ─────────────────────────────────
    if strategy == "Semantic (Recommended)":
        semantic_splitter = build_semantic_splitter(embeddings, threshold_type, threshold_amount)
        try:
            parent_docs = semantic_splitter.split_documents(raw_docs)
        except Exception as e:
            st.warning(f"Semantic chunking failed ({e}), falling back to recursive.")
            parent_docs = build_recursive_splitter(parent_chunk_size, overlap).split_documents(raw_docs)

    elif strategy == "Recursive (Fast)":
        parent_docs = build_recursive_splitter(parent_chunk_size, overlap).split_documents(raw_docs)

    else:  # Hybrid
        # Use semantic for large documents, recursive for small ones
        semantic_splitter = build_semantic_splitter(embeddings, threshold_type, threshold_amount)
        parent_docs = []
        for doc in raw_docs:
            if len(doc.page_content) > 1500:
                try:
                    parent_docs.extend(semantic_splitter.split_documents([doc]))
                except Exception:
                    parent_docs.extend(
                        build_recursive_splitter(parent_chunk_size, overlap).split_documents([doc])
                    )
            else:
                parent_docs.extend(
                    build_recursive_splitter(parent_chunk_size, overlap).split_documents([doc])
                )

    # ── Step 2: Attach metadata to each parent chunk ──────────────────────────
    for i, doc in enumerate(parent_docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["strategy"] = strategy

    # ── Step 3: Optionally create smaller child chunks for embedding ──────────
    if use_parent_child:
        child_chunks, parent_map = create_parent_child_chunks(
            parent_docs, child_chunk_size, overlap
        )
    else:
        # No parent-child: embed parent chunks directly
        parent_map = {}
        for doc in parent_docs:
            pid = str(uuid.uuid4())
            doc.metadata["doc_id"] = pid
            doc.metadata["parent_id"] = pid
            parent_map[pid] = doc
        child_chunks = parent_docs

    return child_chunks, parent_map


# ──────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_with_docling(tmp_path, original_name, suffix, warnings):
    """
    Parse one file with Docling and return a single LangChain Document
    holding the full Markdown text.

    PDFs are capped at 10 pages using Docling's built-in max_num_pages
    argument (this stops Docling from even reading pages past 10, so it
    is also faster than loading everything and slicing afterwards).

    Note: unlike PyPDFLoader, this produces ONE Document per file rather
    than one Document per page, so per-page metadata is not available
    for Docling-parsed files. The downstream chunker still splits this
    into multiple chunks as normal.
    """
    converter = get_docling_converter()

    if suffix == ".pdf":
        # Count real pages first so the truncation warning stays accurate.
        try:
            with open(tmp_path, "rb") as f:
                total_pages = len(pypdf.PdfReader(f).pages)
            if total_pages > 10:
                warnings.append(f"⚠️ '{original_name}' has {total_pages} pages. Only first 10 processed (Docling).")
        except Exception:
            pass  # page count is just for the warning message, not critical

        result = converter.convert(tmp_path, max_num_pages=10)
    else:
        result = converter.convert(tmp_path)

    markdown_text = result.document.export_to_markdown()
    return Document(page_content=markdown_text, metadata={"source": original_name})


def load_raw_documents(uploaded_docs):
    """
    Load raw documents from uploaded files.

    Uses Docling for PDF/DOCX/PPTX/HTML when it is installed, since it
    gives cleaner text and keeps table structure. Falls back to the
    original PyPDFLoader (PDF, 10-page cap) / TextLoader (TXT) path
    automatically if Docling is not installed or fails on a file.
    """
    documents = []
    warnings = []
    docling_converter = get_docling_converter()

    for uploaded_doc in uploaded_docs:
        suffix = os.path.splitext(uploaded_doc.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_doc.read())
            tmp_path = tmp.name

        try:
            used_docling = False
            if docling_converter is not None and suffix in DOCLING_SUPPORTED_EXTENSIONS:
                try:
                    documents.append(
                        load_with_docling(tmp_path, uploaded_doc.name, suffix, warnings)
                    )
                    used_docling = True
                except Exception as e:
                    st.warning(f"Docling failed on '{uploaded_doc.name}', falling back to basic parser: {e}")

            if not used_docling:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    total = len(pages)
                    if total > 10:
                        warnings.append(f"⚠️ '{uploaded_doc.name}' has {total} pages. Only first 10 processed.")
                        pages = pages[:10]
                    documents.extend(pages)
                elif suffix == ".txt":
                    loader = TextLoader(tmp_path)
                    documents.extend(loader.load())
                else:
                    warnings.append(
                        f"⚠️ '{uploaded_doc.name}' needs Docling to be parsed (install with `pip install docling`)."
                    )
        except Exception as e:
            st.error(f"Error loading {uploaded_doc.name}: {e}")
        finally:
            os.unlink(tmp_path)

    return documents, warnings


def process_documents(uploaded_docs, base_url):
    """
    Full pipeline:
    1. Load raw pages (Docling for PDF/DOCX/PPTX/HTML, else PyPDFLoader/TextLoader)
    2. Chunk with selected strategy
    3. Index child chunks in FAISS (same nomic-embed-text embeddings)
    4. Build BM25 index over child chunks
    """
    if not uploaded_docs:
        return None, [], [], {}

    embeddings = get_embeddings(base_url)
    if embeddings is None:
        return None, [], [], {}

    with st.spinner("📄 Parsing documents..."):
        raw_docs, warnings = load_raw_documents(uploaded_docs)
    if not raw_docs:
        return None, warnings, [], {}

    with st.spinner("🧩 Chunking documents..."):
        child_chunks, parent_map = chunk_documents(
            raw_docs,
            embeddings,
            strategy=st.session_state.chunking_strategy,
            threshold_type=st.session_state.semantic_threshold_type,
            threshold_amount=st.session_state.semantic_threshold_amount,
            parent_chunk_size=st.session_state.parent_chunk_size,
            child_chunk_size=st.session_state.child_chunk_size,
            overlap=st.session_state.chunk_overlap,
            use_parent_child=st.session_state.use_parent_child,
        )

    stats = {
        "parent_chunks": len(parent_map),
        "child_chunks": len(child_chunks),
        "strategy": st.session_state.chunking_strategy,
    }

    try:
        with st.spinner(f"📐 Embedding {len(child_chunks)} chunks into FAISS..."):
            vectorstore = FAISS.from_documents(child_chunks, embeddings)
        st.info(
            f"📊 FAISS index: {vectorstore.index.ntotal} vectors │ "
            f"Strategy: {stats['strategy']} │ "
            f"Parent chunks: {stats['parent_chunks']} │ "
            f"Child chunks: {stats['child_chunks']}"
        )
        return vectorstore, warnings, child_chunks, stats
    except Exception as e:
        st.error(f"❌ FAISS indexing failed: {e}")
        return None, warnings, [], stats


# ──────────────────────────────────────────────────────────────────────────────
# DOCUMENT UPLOAD TRIGGER
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_docs and not st.session_state.documents_processed:
    with st.spinner("📚 Processing documents..."):
        (
            st.session_state.vectorstore,
            st.session_state.page_warnings,
            st.session_state.all_chunks,
            st.session_state.chunking_stats,
        ) = process_documents(uploaded_docs, ollama_base_url)
        st.session_state.documents_processed = True

    for w in st.session_state.page_warnings:
        st.warning(w)

    if st.session_state.vectorstore:
        st.success("✅ Documents processed with smart chunking!")
    else:
        st.error("❌ Failed to process documents.")


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def build_hybrid_retriever(vectorstore, all_chunks):
    """
    Ensemble retriever combining:
    - FAISS MMR (semantic, diversity-aware)
    - BM25 (keyword-exact matching)

    After retrieval, if parent-child is enabled, child chunks are swapped
    for their parent documents before being passed to the LLM.
    """
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": st.session_state.retrieval_k,
            "fetch_k": st.session_state.mmr_fetch_k,
            "lambda_mult": st.session_state.mmr_lambda,
        }
    )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = st.session_state.retrieval_k

    semantic_weight = round(1.0 - st.session_state.bm25_weight, 2)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[st.session_state.bm25_weight, semantic_weight],
    )
    return ensemble


def retrieve_context(query, vectorstore, all_chunks, parent_map, use_parent_child):
    """
    Retrieve relevant context for a query.

    Flow:
    1. Run hybrid (BM25 + MMR) retrieval to get child chunks
    2. If parent-child enabled: look up parent doc for each child
       → LLM sees richer surrounding context, not just the tiny matched chunk
    3. Deduplicate by parent_id to avoid repeating the same passage
    4. Return joined context string
    """
    if vectorstore is None:
        return ""

    try:
        retriever = build_hybrid_retriever(vectorstore, all_chunks)
        retrieved_docs = retriever.invoke(query)

        if use_parent_child and parent_map:
            seen_parent_ids = set()
            context_docs = []
            for doc in retrieved_docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    # Return the full parent chunk for richer LLM context
                    parent_doc = parent_map.get(pid, doc)
                    context_docs.append(parent_doc)
        else:
            context_docs = retrieved_docs

        # Join with clear separators so the LLM can distinguish passages
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            header = f"[Chunk {i} | Source: {source}{' | Page: ' + str(page) if page else ''}]"
            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        st.warning(f"Retrieval error: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT & MESSAGE BUILDING
# ──────────────────────────────────────────────────────────────────────────────

system_prompt = """You are an AI coding assistant. Analyze code, debug issues, process images, \
and use the provided context to give clear answers in simple English.

When context is provided below, base your answer on it. If the context does not contain \
the answer, say so clearly rather than guessing.

Context:
{context}
"""


def build_message_list(context="", current_query="", image_data=None):
    messages = [SystemMessage(content=system_prompt.format(context=context))]

    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            clean = re.sub(r'<[^>]+>', '', msg["content"])
            messages.append(AIMessage(content=clean))

    if image_data:
        messages.append(HumanMessage(content=[
            {"type": "text", "text": current_query},
            {"type": "image_url", "image_url": image_data},
        ]))
    else:
        messages.append(HumanMessage(content=current_query))

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

def format_output(full_response):
    think_part = ""
    actual_part = ""

    if "<think>" in full_response and "</think>" in full_response:
        m = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        if m:
            think_part = m.group(1)
            actual_part = full_response.split("</think>", 1)[1].strip()
    elif "<think>" in full_response:
        think_part = full_response.split("<think>", 1)[1]
    else:
        actual_part = full_response

    html = ""
    if think_part:
        html += f'<div class="think-container"><span class="think-output">💭 Thinking: {think_part}</span></div>'
    if actual_part:
        html += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'
    return html


# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def optimized_generation(user_query, vectorstore, all_chunks, parent_map,
                         use_parent_child, image_data_url, response_placeholder):
    start_time = time.time()
    token_count = 0

    try:
        rag_context = ""

        if enable_parallel_rag and vectorstore is not None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                rag_future = executor.submit(
                    retrieve_context, user_query, vectorstore, all_chunks, parent_map, use_parent_child
                )
                llm = ChatOllama(
                    model=selected_model,
                    base_url=ollama_base_url,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    num_batch=num_batch,
                    num_thread=num_thread,
                    top_k=20,
                    top_p=0.9,
                    repeat_penalty=1.05,
                    keep_alive=keep_alive_time,
                    mmap=use_mmap,
                    mlock=use_mlock,
                )
                try:
                    rag_context = rag_future.result(timeout=15)
                except TimeoutError:
                    st.warning("RAG retrieval timed out — proceeding without context.")
        else:
            if vectorstore is not None:
                rag_context = retrieve_context(
                    user_query, vectorstore, all_chunks, parent_map, use_parent_child
                )
            llm = ChatOllama(
                model=selected_model,
                base_url=ollama_base_url,
                temperature=temperature,
                num_ctx=num_ctx,
                num_batch=num_batch,
                num_thread=num_thread,
                top_k=20,
                top_p=0.9,
                repeat_penalty=1.05,
                keep_alive=keep_alive_time,
                mmap=use_mmap,
                mlock=use_mlock,
            )

        messages = build_message_list(rag_context, user_query, image_data_url)
        ai_response = ""

        for chunk in llm.stream(messages):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            ai_response += token
            token_count += 1
            if token_count % 3 == 0:
                response_placeholder.markdown(format_output(ai_response), unsafe_allow_html=True)

        response_placeholder.markdown(format_output(ai_response), unsafe_allow_html=True)

        elapsed = time.time() - start_time
        st.session_state.generation_stats = {
            "tokens": token_count,
            "time": elapsed,
            "tps": token_count / elapsed if elapsed > 0 else 0,
        }
        return ai_response

    except Exception as e:
        msg = (
            f"❌ Error: {e}\n\nChecklist:\n"
            f"1. Ollama running: `ollama serve`\n"
            f"2. Model installed: `ollama pull {selected_model}`\n"
            f"3. Server URL: {ollama_base_url}"
        )
        response_placeholder.markdown(msg)
        return msg


# ──────────────────────────────────────────────────────────────────────────────
# CHAT UI
# ──────────────────────────────────────────────────────────────────────────────

chat_container = st.container()
with chat_container:
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            if msg["role"] == "ai":
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            st.markdown(item["text"])
                        elif item["type"] == "image_url":
                            st.markdown(
                                f'<img src="{item["image_url"]}" class="user-image" alt="Uploaded Image">',
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(msg["content"])

if st.session_state.generation_stats["tokens"] > 0:
    stats = st.session_state.generation_stats
    st.markdown(
        f'<span class="speed-badge">⚡ {stats["tps"]:.1f} tokens/sec | '
        f'{stats["tokens"]} tokens in {stats["time"]:.1f}s</span>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────────────────────────────────────

user_query = st.chat_input("Type your question here...")

if user_query:
    image_content = None
    image_data_url = None

    if uploaded_file is not None:
        file_type = uploaded_file.type
        image_data = uploaded_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_data_url = f"data:{file_type};base64,{image_base64}"
        content = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": image_data_url},
        ]
    else:
        content = user_query

    st.session_state.message_log.append({"role": "user", "content": content})

    with st.chat_message("ai"):
        response_placeholder = st.empty()
        ai_response = optimized_generation(
            user_query,
            st.session_state.vectorstore,
            st.session_state.all_chunks,
            st.session_state.parent_map,
            st.session_state.use_parent_child,
            image_data_url,
            response_placeholder,
        )

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
