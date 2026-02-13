# rag.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import gdown


# ---------------- DOWNLOAD IF NOT EXISTS ----------------
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=False)


# 🔥 PUT YOUR IDS HERE
CHUNKS_ID = "136fGKb8PXW73ock0of5jl8zONN1uQ-Kl"
EMBEDDINGS_ID = "1grISb-FwRjtZokoqDd_vkD6lQrkwImyz"

download_file(CHUNKS_ID, "chunks.pkl")
download_file(EMBEDDINGS_ID, "embeddings.npy")


# ---------------- LOAD EVERYTHING ONCE ----------------
@st.cache_resource
def load_system():

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    vectors = np.load("embeddings.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    load_dotenv()
    client = Groq(api_key=os.getenv("groq_api_key"))

    return chunks, model, index, client


chunks, model, index, client = load_system()


# ---------------- MAIN FUNCTION ----------------
def ask_rag(question):

    query = model.encode([question])

    k = 3
    distances, indices = index.search(query, k)

    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
You are a question answering system.

Use ONLY the information provided in the context.

If the answer is not present, say:
"Sorry, I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
