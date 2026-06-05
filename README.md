## Overview
**AI Code Companion** with RAG & vLLM is an enhanced AI-powered chatbot that serves as your personal coding assistant, offering expertise in various programming languages, debugging, code documentation, solution design, image analysis, document retrieval via Retrieval-Augmented Generation (RAG), and accelerated inference using vLLM. It’s built using Ollama, LangChain, and vLLM, providing a highly interactive, multimodal experience to help you code better, faster, and with access to your own knowledge base.

Key Features:
- 🖥 Programming Language Expert: Expertise in various programming languages for code suggestions, best practices, and solutions.
- 🐞 Debugging Assistant: Help identify bugs, suggest fixes, and strategically add print statements to debug your code.
- 📝 Code Documentation: Generates meaningful documentation for your code.
- 💡 Solution Design: Offers high-level guidance for solving complex problems or designing solutions.
- 👁️ Image Analysis: Analyzes uploaded images (e.g., code screenshots) and incorporates them into responses.
- 📚 RAG Document Retrieval: Retrieves and integrates relevant context from uploaded documents (PDFs or text files) to enhance response accuracy.
- ⚡ vLLM Acceleration: Uses vLLM for high-performance inference with models like gemma3:4b when handling uploads, enabling faster processing and summarization.
- 🧠 There are three distinct types to understand, each operating at a different layer of the stack: KV caching, Prefix caching, Semantic caching

---

## Prerequisites

### 1. System Requirements

- OS: Linux, macOS, or Windows (Linux/macOS preferred for vLLM).
- Hardware: GPU recommended for vLLM (CUDA 12.1+ for NVIDIA GPUs); CPU works for Ollama and FAISS.
- Python: Version 3.8–3.11 (vLLM is sensitive to Python versions).

### 2. Install **Ollama**
To use **Ollama** models (like `deepseek-r1:1.5b`), you'll need to have **Ollama** installed on your machine.

- Download and install **Ollama** from [https://ollama.ai/](https://ollama.ai/).
- Once installed, ensure that Ollama is running locally on your machine. You should be     able to access it via the default URL `http://localhost:11434`.
- Pull required models:
```sh
ollama pull deepscaler:latest
ollama pull nomic-embed-text
```
- Ensure Ollama server is running:
```sh
ollama serve
```

### 3. Install Required Python Libraries
Clone this repository and install the required dependencies using 
```sh
pip install -r requirements.txt
```
Optional: GPU Support
If you have a GPU and want to use **faiss-gpu** or optimize vLLM:
```sh
pip install faiss-gpu
```
Ensure CUDA is installed (check with _nvcc --version_).

### 4. Set Up vLLM Server
vLLM requires a Hugging Face model for gemma3:4b. The code uses google/gemma-2-2b-it as a placeholder (confirm the exact model ID, e.g., check Hugging Face or use a custom/local path). GPU with CUDA 12.1+ is recommended.
- Start vLLM Server:
```sh
python -m vllm.entrypoints.openai.api_server --model google/gemma-2-2b-it --dtype half --gpu-memory-utilization 0.95
```
 - Replace _google/gemma-2-2b-it_ with the correct model ID for _gemma3:4b._
 - This runs an OpenAI-compatible API at **http://localhost:8000/v1** (default port).
 - Use _--tensor-parallel-size N_ for multi-GPU setups (e.g., N=2 for 2 GPUs).
- Verify Server:
```sh
curl http://localhost:8000/v1/models
```

### 5. Run Streamlit
```sh 
streamlit run LocalLLM.py
```

### Key Explanations:

1. **Installation Instructions**: Clear steps on installing **Ollama** and setting up the Python environment.
2. **Purpose of the Code**: A detailed description of the features such as debugging, code documentation, and solution design.
3. **Usage Instructions**: How to run the app and interact with the chatbot.
4. **Example Chat**: Illustrative example of how the chatbot helps with coding queries.
5. **Customizing the Chatbot**: Brief explanation on customizing the chatbot's behavior.

This README will provide a user-friendly guide to installing and using the DeepSeek Code Companion.

### Troubleshooting

- Model Not Found: Ensure models are pulled via _ollama pull_ and vLLM server is running     with the correct model ID.
- vLLM Errors: Check CUDA version and reduce _gpu-memory-utilization_ if memory is low.
- Ollama Connection: Verify **http://localhost:11434** is accessible.
- Streamlit Issues: Upgrade Streamlit with **pip install --upgrade streamlit**.


