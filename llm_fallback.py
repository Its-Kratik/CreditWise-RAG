# llm_fallback.py
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ------------------------
# Configure Gemini
# ------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ------------------------
# SentenceTransformer + FAISS Setup
# ------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = None
index = None
docs = []


def build_vector_store(document_list):
    global doc_embeddings, index, docs
    docs = document_list
    doc_embeddings = embedding_model.encode(docs)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))


def retrieve_similar_docs(question, top_k=5):
    if index is None:
        raise ValueError("FAISS index not initialized. Call build_vector_store() first.")
    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]


def generate_answer_llm(context, question):
    prompt = f"""
You are a helpful assistant answering questions based on the following tabular context:

{context}

Question: {question}
Answer:
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
