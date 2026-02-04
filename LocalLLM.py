import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import base64
import re
import os
import tempfile

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
        color: #ffcc00; /* Yellow color for <think> part */
        font-style: italic; /* Italic style for emphasis */
    }
    .actual-output {
        color: #00ff00; /* Green color for actual output */
        font-weight: bold; /* Bold style for emphasis */
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
</style>
""", unsafe_allow_html=True)

# Application title and caption
st.title("🧠 AI Code Companion with Vision & RAG (Ollama)")
st.caption("🚀 Your Local Multimodal AI Pair Programmer - No GPU Required!")

# Sidebar configuration for model selection and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.info("💡 Make sure Ollama is running: ollama serve")
    
    # Ollama base URL configuration
    ollama_base_url = st.text_input(
        "Ollama Server URL",
        value="http://localhost:11434",
        help="Default local Ollama server address"
    )
    
    # Model selection - choose a vision-capable Ollama model
    selected_model = st.selectbox(
        "Choose Vision Model",
        [ "qwen3-vl:2b","ministral-3:3b"
        ],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # File uploader for RAG documents with 10-page limit warning
    st.markdown("### 📚 Document Upload")
    st.warning("⚠️ PDFs larger than 10 pages will be truncated!")
    uploaded_docs = st.file_uploader(
        "Upload Documents for RAG (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True, 
        key="doc_uploader"
    )
    
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    - 👁️ Image Analysis (Vision)
    - 🤔 Reasoning & Thinking
    - 📚 RAG Document Retrieval
    - 💻 100% Local - No Cloud
    - 🖥️ CPU Only - No GPU Needed
    """)
    st.divider()
    st.markdown("### How to Install Models")
    st.code("""


# Install embedding model
ollama pull nomic-embed-text
    """, language="bash")
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize session state for vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "page_warnings" not in st.session_state:
    st.session_state.page_warnings = []

# Initialize embedding model for RAG (using Ollama embeddings)
@st.cache_resource
def get_embeddings(base_url):
    """Initialize and cache the Ollama embedding model."""
    try:
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=base_url
        )
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama embeddings: {str(e)}")
        st.info("Make sure you have pulled the embedding model: ollama pull nomic-embed-text")
        return None

embeddings = get_embeddings(ollama_base_url)

# Process uploaded documents for RAG with 10-page limit
def process_documents(uploaded_docs):
    """Process uploaded documents and create a FAISS vector store. Limit PDFs to 10 pages."""
    if not uploaded_docs or embeddings is None:
        return None, []
    
    documents = []
    warnings = []
    
    for uploaded_doc in uploaded_docs:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_doc.name)[1]) as tmp_file:
            tmp_file.write(uploaded_doc.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Load document based on file type
            if uploaded_doc.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                
                # Check page count and limit to 10 pages
                total_pages = len(pages)
                if total_pages > 10:
                    warnings.append(f"⚠️ '{uploaded_doc.name}' has {total_pages} pages. Only first 10 pages processed.")
                    pages = pages[:10]  # Limit to first 10 pages
                
                documents.extend(pages)
            else:  # txt
                loader = TextLoader(tmp_file_path)
                documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error processing {uploaded_doc.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)  # Clean up temp file
    
    # Create FAISS vector store
    if documents:
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore, warnings
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            return None, warnings
    return None, warnings

# Update vector store when new documents are uploaded
if uploaded_docs and not st.session_state.documents_processed:
    with st.spinner("📚 Processing documents..."):
        st.session_state.vectorstore, st.session_state.page_warnings = process_documents(uploaded_docs)
        st.session_state.documents_processed = True
        
        # Show warnings for large PDFs
        if st.session_state.page_warnings:
            for warning in st.session_state.page_warnings:
                st.warning(warning)
        
        if st.session_state.vectorstore:
            st.success("✅ Documents processed successfully!")

# Reset document processing flag when uploader is cleared
if not uploaded_docs and st.session_state.documents_processed:
    st.session_state.documents_processed = False
    st.session_state.page_warnings = []

# System prompt for the Ollama model
system_prompt = """
You are an AI coding assistant. Analyze code, debug issues, process images, and use the provided context to give clear answers in simple English.

Context: {context}
"""

# Manage session state for chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm your Local AI Code Assistant with Vision & RAG powered by Ollama. Upload documents, ask coding questions, or analyze images! 💻📚👁️"}
    ]

# Create a container for the chat interface
chat_container = st.container()

# Function to stream and format AI output incrementally
def stream_formatted_output(raw_stream, placeholder):
    """Stream AI response token by token and format output."""
    full_response = ""
    for chunk in raw_stream:
        if chunk:
            full_response += chunk
            
            # Parse thinking and actual output (if model uses <think> tags)
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
            
            # Format output with HTML
            html_output = ""
            if think_part:
                html_output += f'<div class="think-container"><span class="think-output">💭 Thinking: {think_part}</span></div>'
            if actual_part:
                html_output += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'
            
            placeholder.markdown(html_output, unsafe_allow_html=True)
    
    return full_response

# Display chat messages in the container
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

# User input section: text query and optional image upload
user_query = st.chat_input("Type your coding question here...")
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"], key="image_uploader")

# Function to build the message list for Ollama
def build_message_list(context="", current_query="", image_data=None):
    """Construct the message list for Ollama including chat history and context."""
    messages = []
    
    # Add system message with context
    messages.append(SystemMessage(content=system_prompt.format(context=context)))
    
    # Add chat history
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            if isinstance(msg["content"], list):
                # Handle multimodal content
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            # Clean AI response from HTML formatting
            clean_content = re.sub(r'<[^>]+>', '', msg["content"])
            messages.append(AIMessage(content=clean_content))
    
    # Add current query
    if image_data:
        messages.append(HumanMessage(content=[
            {"type": "text", "text": current_query},
            {"type": "image_url", "image_url": image_data}
        ]))
    else:
        messages.append(HumanMessage(content=current_query))
    
    return messages

# Process user input and generate streamed AI response
if user_query:
    with st.spinner("🧠 Processing your request..."):
        # Handle multimodal input (text + optional image)
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
        
        # Append user message to the chat log
        st.session_state.message_log.append({"role": "user", "content": content})
        
        # Retrieve RAG context if vectorstore exists
        rag_context = ""
        if st.session_state.vectorstore is not None:
            try:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(user_query)
                rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                st.warning(f"Could not retrieve documents: {str(e)}")
        
        # Initialize Ollama ChatModel
        try:
            llm = OllamaLLM(
                model=selected_model,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            
            # Build message list
            messages = build_message_list(rag_context, user_query, image_data_url)
            
            # Generate response
            with st.chat_message("ai"):
                response_placeholder = st.empty()
                
                try:
                    # Stream the response
                    ai_response = ""
                    for chunk in llm.stream(messages):
                        if hasattr(chunk, 'content'):
                            token = chunk.content
                        else:
                            token = str(chunk)
                        
                        ai_response += token
                        
                        # Format and display incrementally
                        think_part = ""
                        actual_part = ""
                        
                        if "<think>" in ai_response and "</think>" in ai_response:
                            think_match = re.search(r'<think>(.*?)</think>', ai_response, re.DOTALL)
                            if think_match:
                                think_part = think_match.group(1)
                                actual_part = ai_response.split("</think>", 1)[1].strip()
                        elif "<think>" in ai_response:
                            think_part = ai_response.split("<think>", 1)[1]
                        else:
                            actual_part = ai_response
                        
                        html_output = ""
                        if think_part:
                            html_output += f'<div class="think-container"><span class="think-output">💭 Thinking: {think_part}</span></div>'
                        if actual_part:
                            html_output += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'
                        
                        response_placeholder.markdown(html_output, unsafe_allow_html=True)
                    
                except Exception as e:
                    ai_response = f"❌ Error generating response: {str(e)}\n\nPlease check:\n1. Ollama is running: ollama serve\n2. Model is installed: ollama pull {selected_model}\n3. Server URL is correct: {ollama_base_url}"
                    response_placeholder.markdown(ai_response)
            
            # Append the complete AI response to the chat log
            st.session_state.message_log.append({"role": "ai", "content": ai_response})
            
        except Exception as e:
            st.error(f"❌ Failed to connect to Ollama: {str(e)}")
            st.info("Make sure Ollama is running: ollama serve")
        
        # Rerun the app to update the chat display
        st.rerun()
