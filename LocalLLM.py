import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import base64
import re
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

# Custom CSS styling for the application
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .think-output {
        color: #ffcc00;
        font-style: italic;
    }
    .actual-output {
        color: #00ff00;
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .user-image {
        max-width: 300px;
        margin-top: 10px;
    }
    .warning-box {
        background-color: #ff6b6b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .performance-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
        font-size: 12px;
        display: inline-block;
        margin: 5px 0;
    }
    .speed-badge {
        background-color: #2196F3;
        color: white;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 11px;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Application title and caption
st.caption("🚀 Your Local Multimodal AI Pair Programmer - Optimized for Speed!")
st.markdown('<span class="performance-badge">⚡ Parallel RAG + Optimized Streaming</span>', unsafe_allow_html=True)

# Initialize session state FIRST (before sidebar uses it)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "page_warnings" not in st.session_state:
    st.session_state.page_warnings = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = {"tokens": 0, "time": 0.0}
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm your Local AI Code Assistant 💻📚👁️ (Speed Optimized!)"}
    ]

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ollama_base_url = st.text_input(
        "Ollama Server URL",
        value="http://localhost:11434",
        help="Default local Ollama server address",
        key="ollama_url"
    )
    
    selected_model = st.selectbox(
        "Choose Vision Model",
        [
            "qwen3-vl:2b",
            # "qwen3-vl:2b-q4_K_M",
            "ministral-3:3b",
            # "ministral-3:3b-q4_0"
        ],
        help="Q4 models are 2-3x faster with minimal quality loss",
        key="model_select"
    )

    st.divider()
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative",
        key="temp_slider"
    )

    with st.expander("🚀 Advanced Performance Settings"):
        st.markdown("**Inference Optimization**")
        
        num_ctx = st.number_input(
            "Context Window Size",
            min_value=512,
            max_value=32768,
            value=4096,
            step=512,
            help="Larger = longer conversations, more memory usage",
            key="num_ctx_input"
        )
        
        num_batch = st.number_input(
            "Batch Size",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            help="Higher = faster prompt processing (30-50% boost)",
            key="num_batch_input"
        )
        
        num_thread = st.number_input(
            "CPU Threads",
            min_value=1,
            max_value=16,
            value=4,
            step=1,
            help="Number of CPU threads for parallel processing",
            key="num_thread_input"
        )
        
        keep_alive_time = st.selectbox(
            "Model Keep-Alive Time",
            options=["1m", "5m", "10m", "30m", "1h"],
            index=2,
            help="Keeps model in memory for faster responses",
            key="keep_alive_select"
        )
        
        use_mmap = st.checkbox(
            "Memory Mapping (mmap)",
            value=True,
            help="Reduces RAM usage",
            key="mmap_checkbox"
        )
        
        use_mlock = st.checkbox(
            "Lock in RAM (mlock)",
            value=False,
            help="Prevents swapping (requires sufficient RAM)",
            key="mlock_checkbox"
        )
        
        # st.markdown("**UI Performance**")
        
        enable_parallel_rag = st.checkbox(
            "Parallel RAG Retrieval",
            value=True,
            help="Fetch documents while model loads (faster)",
            key="parallel_rag_checkbox"
        )
    
    st.divider()
    st.markdown("### 📚 Document Upload")
    
    # Show RAG status
    if st.session_state.vectorstore is not None:
        st.success("✅ RAG Active")
        # Add manual clear button
        if st.button("🗑️ Clear Documents & Free Memory", key="clear_docs_btn"):
            st.session_state.documents_processed = False
            st.session_state.page_warnings = []
            st.session_state.vectorstore = None
            st.rerun()
    else:
        st.info("ℹ️ No documents loaded - embeddings will load when you upload")
    
    st.warning("⚠️ PDFs larger than 10 pages will be truncated!")
    uploaded_docs = st.file_uploader(
        "Upload Documents for RAG (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True, 
        key="doc_uploader"
    )
    
    st.markdown("### 🖼️ Image Upload")
    uploaded_file = st.file_uploader(
        "Upload an image (optional)", 
        type=["jpg", "png", "jpeg"], 
        key="image_uploader"
    )
    
    st.divider()
    st.markdown("### Quick Setup")
    st.code("""
# Install vision model (required)
ollama pull qwen3-vl:2b-q4_K_M

# Install embedding model (only if using RAG)
ollama pull nomic-embed-text

# Start server
ollama serve
    """, language="bash")
    
    st.info("💡 Tip: Embeddings load automatically when you upload documents")

# Don't load embeddings until documents are uploaded
embeddings = None

@st.cache_resource
def get_embeddings(base_url):
    """Initialize and cache Ollama embedding model (lazy-loaded)."""
    try:
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url
        )
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama embeddings: {str(e)}")
        st.info("Make sure you have: ollama pull nomic-embed-text")
        return None

# Don't load embeddings until documents are uploaded
embeddings = None

def process_documents(uploaded_docs, base_url):
    """Process documents and create FAISS vector store (10 page limit for PDFs)."""
    if not uploaded_docs:
        return None, []
    
    # Lazy-load embeddings only when documents are uploaded
    embeddings = get_embeddings(base_url)
    if embeddings is None:
        return None, []
    
    documents = []
    warnings = []
    
    for uploaded_doc in uploaded_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
            tmp_file.write(uploaded_doc.read())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_doc.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                
                total_pages = len(pages)
                if total_pages > 10:
                    warnings.append(f"⚠️ '{uploaded_doc.name}' has {total_pages} pages. Only first 10 processed.")
                    pages = pages[:10]
                
                documents.extend(pages)
            else:
                loader = TextLoader(tmp_file_path)
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error processing {uploaded_doc.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    if documents:
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore, warnings
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            return None, warnings
    return None, warnings

# Process uploaded documents (lazy-load embeddings only when needed)
if uploaded_docs and not st.session_state.documents_processed:
    with st.spinner("📚 Processing documents (loading embeddings)..."):
        st.session_state.vectorstore, st.session_state.page_warnings = process_documents(uploaded_docs, ollama_base_url)
        st.session_state.documents_processed = True
        
        if st.session_state.page_warnings:
            for warning in st.session_state.page_warnings:
                st.warning(warning)
        
        if st.session_state.vectorstore:
            st.success("✅ Documents processed successfully!")
        else:
            st.error("❌ Failed to process documents. Make sure embedding model is installed.")

# Only clear on explicit user action or page refresh (not on widget changes)
# Removed auto-clear when uploader is empty to preserve session state

system_prompt = """
You are an AI coding assistant. Analyze code, debug issues, process images, and use the provided context to give clear answers in simple English.

Context: {context}
"""

chat_container = st.container()

def retrieve_context(query, vectorstore):
    """Retrieve RAG context (thread-safe)."""
    if vectorstore is None:
        return ""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return ""

def format_output(full_response):
    """Format AI response with thinking and actual output."""
    think_part = ""
    actual_part = ""
    
    if "<think>" in full_response and "</think>" in full_response:
        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        if think_match:
            think_part = think_match.group(1)
            actual_part = full_response.split("</think>", 1)[1].strip()
    elif "<think>" in full_response:
        think_part = full_response.split("<think>", 1)[1]
    else:
        actual_part = full_response
    
    html_output = ""
    if think_part:
        html_output += f'<div class="think-container"><span class="think-output">💭 Thinking: {think_part}</span></div>'
    if actual_part:
        html_output += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'
    
    return html_output

def build_message_list(context="", current_query="", image_data=None):
    """Build message list for Ollama."""
    messages = []
    
    messages.append(SystemMessage(content=system_prompt.format(context=context)))
    
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            clean_content = re.sub(r'<[^>]+>', '', msg["content"])
            messages.append(AIMessage(content=clean_content))
    
    if image_data:
        messages.append(HumanMessage(content=[
            {"type": "text", "text": current_query},
            {"type": "image_url", "image_url": image_data}
        ]))
    else:
        messages.append(HumanMessage(content=current_query))
    
    return messages

def optimized_generation(user_query, vectorstore, image_data_url, response_placeholder):
    """
    Optimized generation with parallel RAG and efficient streaming.
    """
    start_time = time.time()
    token_count = 0
    
    try:
        rag_context = ""
        
        # Parallel RAG retrieval if enabled
        if enable_parallel_rag and vectorstore is not None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                rag_future = executor.submit(retrieve_context, user_query, vectorstore)
                
                # Initialize model while RAG is running
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
                
                # Get RAG results with timeout
                try:
                    rag_context = rag_future.result(timeout=10)
                except TimeoutError:
                    st.warning("RAG retrieval timed out, proceeding without context")
                    rag_context = ""
        else:
            # Sequential processing
            if vectorstore is not None:
                rag_context = retrieve_context(user_query, vectorstore)
            
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
        
        # Build messages
        messages = build_message_list(rag_context, user_query, image_data_url)
        
        # Stream response
        ai_response = ""
        
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                token = chunk.content
            else:
                token = str(chunk)
            
            ai_response += token
            token_count += 1
            
            # Batched UI updates (every 3 tokens for optimal performance)
            if token_count % 3 == 0:
                html_output = format_output(ai_response)
                response_placeholder.markdown(html_output, unsafe_allow_html=True)
        
        # Final update
        final_html = format_output(ai_response)
        response_placeholder.markdown(final_html, unsafe_allow_html=True)
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
        
        st.session_state.generation_stats = {
            "tokens": token_count,
            "time": elapsed_time,
            "tps": tokens_per_second
        }
        
        return ai_response
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n\nChecklist:\n1. Ollama running: `ollama serve`\n2. Model installed: `ollama pull {selected_model}`\n3. Server URL: {ollama_base_url}"
        response_placeholder.markdown(error_message)
        return error_message

# Display chat history
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

# Display generation stats if available
if st.session_state.generation_stats["tokens"] > 0:
    stats = st.session_state.generation_stats
    st.markdown(
        f'<span class="speed-badge">⚡ {stats["tps"]:.1f} tokens/sec | '
        f'{stats["tokens"]} tokens in {stats["time"]:.1f}s</span>',
        unsafe_allow_html=True
    )

user_query = st.chat_input("Type your coding question here...")

if user_query:
    image_content = None
    image_data_url = None
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        image_data = uploaded_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_data_url = f"data:{file_type};base64,{image_base64}"
        image_content = {
            "type": "image_url", 
            "image_url": image_data_url
        }
        content = [
            {"type": "text", "text": user_query},
            image_content
        ]
    else:
        content = user_query
    
    st.session_state.message_log.append({"role": "user", "content": content})
    
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        
        ai_response = optimized_generation(
            user_query, 
            st.session_state.vectorstore, 
            image_data_url,
            response_placeholder
        )
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Don't auto-rerun - let Streamlit handle it naturally
    # This prevents clearing session state on every interaction
