import os
import io
import uuid
from typing import List, Tuple, Dict

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# -----------------------------
# Configuration and Initialization
# -----------------------------
st.set_page_config(page_title="Internal RAG Chatbot (Local)", page_icon="🤖", layout="wide")

# Initialize OpenAI client (expects env vars for local self-hosted endpoint configuration if applicable)
client = OpenAI()

# Persistent ChromaDB directory
DEFAULT_DB_DIR = "./chroma_db"


# -----------------------------
# Utility Functions
# -----------------------------
def load_text_from_pdf(file_obj: io.BytesIO) -> str:
    try:
        reader = PdfReader(file_obj)
        texts = []
        for page in reader.pages:
            content = page.extract_text() or ""
            texts.append(content)
        return "\n".join(texts)
    except Exception:
        return ""


def load_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def get_chroma_collection(db_dir: str, collection_name: str = "internal_docs"):
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
    # Ensure embedding function is set (get_collection may not preserve it)
    collection._embedding_function = embedding_fn
    return collection


def add_documents(collection, docs: List[Dict], namespace: str):
    # docs: List of {"text": str, "metadata": dict}
    if not docs:
        return
    ids = []
    metadatas = []
    documents = []
    for d in docs:
        doc_id = f"{namespace}-{uuid.uuid4()}"
        ids.append(doc_id)
        metadatas.append(d.get("metadata", {}))
        documents.append(d.get("text", ""))
    collection.add(ids=ids, metadatas=metadatas, documents=documents)


def query_collection(collection, query: str, n_results: int = 5) -> Tuple[List[str], List[Dict]]:
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0] if results else []
    metas = results.get("metadatas", [[]])[0] if results else []
    return docs, metas


def build_messages(contexts: List[str], query: str) -> List[Dict[str, str]]:
    context_block = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question accurately and concisely. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer using only the information in the context."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Internal RAG Chatbot (Local)")

with st.sidebar:
    st.header("Settings")
    db_dir = st.text_input("ChromaDB Directory", value=DEFAULT_DB_DIR)
    top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=10, value=5, step=1)
    chunk_size = st.number_input("Chunk Size (tokens approximated by words)", min_value=100, max_value=4000, value=1000, step=50)
    chunk_overlap = st.number_input("Chunk Overlap (words)", min_value=0, max_value=1000, value=200, step=10)

    col1, col2 = st.columns(2)
    with col1:
        clear_db = st.button("Clear Collection")
    with col2:
        rebuild = st.button("Rebuild Index")

# Initialize Chroma collection in session state
if "collection" not in st.session_state or rebuild:
    st.session_state.collection = get_chroma_collection(db_dir=db_dir)
    st.success("ChromaDB collection initialized.")

if clear_db:
    # Drop and recreate collection
    try:
        chroma_client = chromadb.PersistentClient(path=db_dir)
        chroma_client.delete_collection(name="internal_docs")
    except Exception:
        pass
    st.session_state.collection = get_chroma_collection(db_dir=db_dir)
    st.warning("Collection cleared.")

collection = st.session_state.collection

# File Uploader
st.subheader("Upload Internal Documents")
uploaded_files = st.file_uploader(
    "Upload one or more TXT or PDF documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# Process and index uploaded files
if uploaded_files:
    for f in uploaded_files:
        file_name = f.name
        file_type = os.path.splitext(file_name)[1].lower()

        if file_type == ".pdf":
            bytes_data = f.read()
            text = load_text_from_pdf(io.BytesIO(bytes_data))
        else:
            text = load_text_from_txt(f.read())

        chunks = chunk_text(text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))

        doc_entries = []
        for i, chunk in enumerate(chunks):
            doc_entries.append({
                "text": chunk,
                "metadata": {
                    "source": file_name,
                    "chunk_index": i
                }
            })

        if doc_entries:
            add_documents(collection, doc_entries, namespace=file_name)
            st.success(f"Indexed {len(doc_entries)} chunks from {file_name}")
        else:
            st.warning(f"No extractable text found in {file_name}")

# Query interface
st.subheader("Ask a Question")
query = st.text_input("Enter your query about the uploaded documents:")

if st.button("Get Answer") and query.strip():
    with st.spinner("Retrieving relevant context..."):
        retrieved_docs, metadatas = query_collection(collection, query, n_results=int(top_k))

    if not retrieved_docs:
        st.info("No relevant documents found. Try uploading documents or rephrasing your query.")
    else:
        with st.expander("Retrieved Context Chunks", expanded=False):
            for i, (doc, meta) in enumerate(zip(retrieved_docs, metadatas), start=1):
                src = meta.get("source", "unknown")
                idx = meta.get("chunk_index", "n/a")
                st.write(f"[{i}] {src} (chunk {idx})")
                st.write(doc)
                st.write("---")

        with st.spinner("Generating answer..."):
            messages = build_messages(retrieved_docs, query)
            try:
                answer = generate_answer(messages)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Model generation failed: {e}")
else:
    st.caption("Upload documents, enter a question, and click 'Get Answer'.")