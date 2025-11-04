import os
import re
import gc
import torch
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from rag_classes import SemanticRAG, AnswerQuestion

# --------------------------
# Page Configuration #
# --------------------------
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 RAG Chat Interface")
st.caption("Retrieve relevant context and generate answers using Mistral-7B and FAISS")

# --------------------------
# Environment Variables
# --------------------------

os.environ["FAISS_INDEX_PATH"] = "/content/drive/MyDrive/rag_app/metadata/semantic_index.faiss"
os.environ["FAISS_META_CSV"] = "/content/drive/MyDrive/rag_app/metadata/embedding_metadata.csv"
os.environ["EMBEDDING_MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["MODEL_PATH"] = "/content/drive/MyDrive/huggingface_models/mistral-7b-instruct-v0.1"
os.environ["MAX_NEW_TOKENS"] = "1024"
os.environ["LOAD_IN_8BIT"] = "false"


index_path = os.environ.get("FAISS_INDEX_PATH", "semantic_index.faiss")
metadata_csv_path = os.environ.get("FAISS_META_CSV", "embedding_metadata.csv")
embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
model_path = os.environ.get("MODEL_PATH", "")
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
load_in_8bit = os.environ.get("LOAD_IN_8BIT", "false").lower() == "false"

# --------------------------
# Utility Functions
# --------------------------
def format_paragraphs(text: str) -> str:
    return re.sub(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", "\n\n", text)

# --------------------------
# Load Models and Classes
# --------------------------
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    return SentenceTransformer(embedding_model_name)

@st.cache_resource(show_spinner=True)
def load_retriever(index_path, metadata_csv_path, embedding_model_name):
    return SemanticRAG(index_path=index_path, metadata_csv_path=metadata_csv_path, embedding_model_name=embedding_model_name)


embedding_model = load_embedding_model()
retriever = load_retriever(index_path, metadata_csv_path, embedding_model_name)

# ---------------- Load QA model ONCE ---------------- #
@st.cache_resource(show_spinner=True)
def load_qa_model():
    model_path = os.environ.get("MODEL_PATH", "")
    return AnswerQuestion(retriever)

# Only one instance exists in memory for the whole session
qa_model = load_qa_model()

# --------------------------
# Chat Interface
# --------------------------
query = st.text_area("Enter your question:", placeholder="e.g., Describe how sampleview distinguishes Water/Oil/Gas. Is the bulb UV or IR?", height=150)

temperature = st.slider(
    "Creativity (Temperature)",
    min_value=0.0,
    max_value=1.5,
    value=0.7,
    step=0.1,
    help="Lower values make answers more focused and deterministic. Higher values increase creativity and variability."
)

if st.button("Generate Answer") and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        answer = qa_model.answer(query, max_new_tokens=max_new_tokens, temperature=temperature)
        st.subheader("📝 Answer")
        st.write(format_paragraphs(answer))

        st.subheader("🔎 Retrieved Sources")
        sources_df = retriever.retrieve(query)
        for i, row in sources_df.iterrows():
            st.markdown(f"**{row['topic']}**")
            st.caption(f"Equipment: {row['equipment_name']} • Score: {row['score']:.3f}")
            st.write(row['chunk_text'])
            st.divider()