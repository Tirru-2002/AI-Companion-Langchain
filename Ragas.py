"""
Document Evaluation Report — Google Gemini + RAGAS
-------------------------------------------------
A minimal Streamlit app that scores a Gemini LLM's RAG answers against a
predefined set of question / ground-truth pairs, using RAGAS metrics.
This uses the Google Gemini API, so you need a Google AI Studio API key.

Metrics used (these are the real RAGAS metric names — there is no metric
literally called "fairness" or "accuracy"; the closest equivalents are):
    - answer_correctness   -> how correct the answer is vs. the ground truth
    - answer_relevancy      -> how relevant the answer is to the question
    - faithfulness          -> whether the answer stays grounded in the context
    - context_precision     -> how much of the retrieved context is useful
    - context_recall        -> how much of the needed information was retrieved

Install:
    pip install streamlit langchain langchain-community langchain-google-genai \
                ragas datasets faiss-cpu pypdf nest_asyncio

Before running:
    set GOOGLE_API_KEY=your_api_key_here
    # or enter your key in the Streamlit sidebar

Run:
    streamlit run Ragas.py
"""

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

import tempfile
import difflib

import streamlit as st

# Ragas sometimes conflicts with Streamlit's own event loop. This keeps it safe.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# ──────────────────────────────────────────────────────────────────────────
# EDIT THIS: your predefined questions and their correct answers.
# Every question must have a matching ground truth at the same position.
# These defaults match the demo text used in the RAGAS example PDF, so the
# app works immediately if you upload a .txt file with those five sentences.
# ──────────────────────────────────────────────────────────────────────────
questions = [
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "Where is Mount Everest located?",
    "What is Mike's favorite color?",
]

ground_truths = [
    "Paris",
    "Jane Austen",
    "the Himalayas",
    "Pink",
]

assert len(questions) == len(ground_truths), "questions and ground_truths must be the same length."


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + MATTE THEME
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document Evaluation Report", page_icon="▪", layout="wide")

st.markdown("""
<style>
:root {
  --bg: #211f1b;
  --panel: #292620;
  --border: rgba(255,255,255,0.08);
  --text: #e7e4da;
  --text-muted: #9c988c;
  --accent: #6f7d5c;
  --accent-hover: #7f8d6a;
}
.stApp { background-color: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
p, span, label, .stMarkdown { color: var(--text); }
.stButton > button {
  background-color: var(--accent);
  color: #1b1a16;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1.2rem;
  font-weight: 600;
  box-shadow: none;
}
.stButton > button:hover { background-color: var(--accent-hover); color: #1b1a16; }
input, textarea {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}
[data-testid="stMetricValue"] { color: var(--text); font-weight: 600; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }
[data-testid="stMetricDelta"] { display: none; }
hr { border-color: var(--border); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 4px; }
::selection { background-color: var(--accent); color: #1b1a16; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────
defaults = {"vectorstore": None, "docs_ready": False, "report_df": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

google_api_key = os.getenv("GOOGLE_API_KEY", "")
# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    
    llm_model = st.selectbox(
        "LLM model",
        options=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        index=0,
    )
    embed_model = st.selectbox("Embedding model", options=["gemini-embedding-2","gemini-embedding-001"], index=0)
    retrieval_k = st.slider("Chunks to retrieve", 1, 8, 3)

    st.divider()
    st.subheader("Documents")
    uploaded_docs = st.file_uploader(
        "Upload PDF / TXT", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.session_state.vectorstore is not None:
        st.success("Documents indexed")
    else:
        st.caption("No documents indexed yet.")

    if st.button("Reset documents"):
        st.session_state.vectorstore = None
        st.session_state.docs_ready = False
        st.session_state.report_df = None
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# LOCAL MODEL LOADERS
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings(api_key, model_name):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)


def get_llm(api_key, model_name):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Google API key is required.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)


# ──────────────────────────────────────────────────────────────────────────
# DOCUMENT LOADING + CHUNKING
# ──────────────────────────────────────────────────────────────────────────
def load_documents(files):
    docs = []
    for f in files:
        suffix = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            path = tmp.name
        try:
            if suffix == ".pdf":
                docs.extend(PyPDFLoader(path).load())
            else:
                docs.extend(TextLoader(path).load())
        finally:
            os.unlink(path)
    return docs


def chunk_documents(raw_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(raw_docs)


if uploaded_docs and not st.session_state.docs_ready:
    if not google_api_key:
        st.warning("Please enter your Google API key first.")
    else:
        with st.spinner("Reading and indexing documents..."):
            embeddings = get_embeddings(google_api_key, embed_model)
            raw = load_documents(uploaded_docs)
            chunks = chunk_documents(raw)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings) if chunks else None
            st.session_state.docs_ready = True
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# RETRIEVAL + GENERATION
# ──────────────────────────────────────────────────────────────────────────
def retrieve_context(query, vectorstore, k):
    if vectorstore is None:
        return ["No document uploaded — answer generated from model knowledge only."]
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs] if docs else ["No relevant context found in the uploaded documents."]

def extract_text(content):  # to handle different content types returned by LLMs
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for item in content:

            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))

            elif hasattr(item, "text"):
                texts.append(item.text)

            else:
                texts.append(str(item))

        return "\n".join(texts)

    return str(content)

def generate_answer(llm, question, contexts):
    context_text = "\n\n---\n\n".join(contexts)
    system = SystemMessage(content=(
        "Only string output!"
        "Answer the question using only the context below. "
        "If the answer is not in the context, say 'Not found in document.' "
        "Keep the answer short and clear.\n\nContext:\n" + context_text
    ))
    human = HumanMessage(content=question)
    response = llm.invoke([system, human])
    return extract_text(response.content)

# ──────────────────────────────────────────────────────────────────────────
# RAGAS EVALUATION
# ──────────────────────────────────────────────────────────────────────────
def evaluate_qa_pairs(llm, embeddings, vectorstore, qa_pairs, k):
    """qa_pairs: list of (question, ground_truth) tuples. Returns a pandas DataFrame."""
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    rows = []
    for question, truth in qa_pairs:
        contexts = retrieve_context(question, vectorstore, k)
        answer = generate_answer(llm, question, contexts)
        answer = extract_text(answer) # handle different content types returned by LLMs
        rows.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": truth,
        })

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas()


# ──────────────────────────────────────────────────────────────────────────
# FUZZY MATCH: find the closest predefined question to what the user typed
# ──────────────────────────────────────────────────────────────────────────
def match_predefined_question(user_question, threshold=0.55):
    best_q, best_ratio = None, 0.0
    for q in questions:
        ratio = difflib.SequenceMatcher(None, user_question.lower().strip(), q.lower().strip()).ratio()
        if ratio > best_ratio:
            best_q, best_ratio = q, ratio
    if best_ratio >= threshold:
        idx = questions.index(best_q)
        return best_q, ground_truths[idx], best_ratio
    return None, None, best_ratio


# ──────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────
st.title("Document Evaluation Report")
st.caption("RAG evaluation using RAGAS metrics with the Google Gemini API.")

with st.expander("Predefined questions and ground truths", expanded=False):
    st.table({"question": questions, "ground_truth": ground_truths})

st.divider()
st.subheader("Full evaluation report")

if st.button("Run evaluation report"):
    if not google_api_key:
        st.warning("Please enter your Google API key first.")
    else:
        llm = get_llm(google_api_key, llm_model)
        embeddings = get_embeddings(google_api_key, embed_model)
        with st.spinner("Generating answers and scoring against ground truths..."):
            st.session_state.report_df = evaluate_qa_pairs(
                llm, embeddings, st.session_state.vectorstore,
                list(zip(questions, ground_truths)), retrieval_k,
            )

if st.session_state.report_df is not None:
    df = st.session_state.report_df
    metric_cols = ["answer_correctness", "answer_relevancy", "faithfulness",
                    "context_precision", "context_recall"]
    present = [c for c in metric_cols if c in df.columns]
    cols = st.columns(len(present))
    for c, col in zip(present, cols):
        col.metric(c.replace("_", " ").title(), f"{df[c].mean():.2f}")
    st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("Ask a question")
st.caption("If your question matches one of the predefined questions, it is scored against its ground truth.")

user_q = st.text_input("Your question", key="ask_input")
if st.button("Check answer") and user_q.strip():
    if not google_api_key:
        st.warning("Please enter your Google API key first.")
    else:
        llm = get_llm(google_api_key, llm_model)
        embeddings = get_embeddings(google_api_key, embed_model)
        matched_q, matched_truth, ratio = match_predefined_question(user_q)

        if matched_q:
            st.caption(f"Matched predefined question ({ratio:.0%} similarity): \"{matched_q}\"")
            with st.spinner("Evaluating against ground truth..."):
                df_single = evaluate_qa_pairs(
                    llm, embeddings, st.session_state.vectorstore,
                    [(user_q, matched_truth)], retrieval_k,
                )
            st.dataframe(df_single, use_container_width=True)
        else:
            st.info("No matching ground-truth question found. Showing answer only, without a cross-check score.")
            contexts = retrieve_context(user_q, st.session_state.vectorstore, retrieval_k)
            answer = generate_answer(llm, user_q, contexts)
            st.write(answer)