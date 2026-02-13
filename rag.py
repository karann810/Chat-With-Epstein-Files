# import pickle 
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from dotenv import load_dotenv
# import os




# with open("notebook/chunks.pkl" , "rb") as f:
#     chunks = pickle.load(f)
# print(len(chunks))



# vectors = np.load("notebook/embeddings.npy")
# print(vectors.shape)


# model = SentenceTransformer("all-MiniLM-L6-v2")

# dimension = vectors.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(vectors)

# print("enter the question")
# question = input()
# query = model.encode([question])

# k = 5
# distances, indices = index.search(query, k)

# for i in indices[0]:
#     print("-----")
#     print(chunks[i])

# context = "\n\n".join([chunks[i] for i in indices[0]])

# prompt = f"""
# You are a question answering system.

# Use ONLY the information provided in the context.
# Do NOT use outside knowledge.

# If the answer is not present in the context,
# reply exactly with:
# "Sorry, I don't know."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """



# load_dotenv()  # read .env file
# groq_api_key = os.getenv("groq_api_key")
# client = Groq(api_key=groq_api_key)
# response = client.chat.completions.create(
#    model = "llama-3.1-8b-instant",
#     messages=[
#         {"role": "user", "content": prompt}
#     ]
# )

# print(response.choices[0].message.content)



# rag.py
# rag.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import streamlit as st   # ⭐ needed for caching


# ---------------- LOAD ONCE ----------------
@st.cache_resource
def load_system():

    with open("notebook/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    vectors = np.load("notebook/embeddings.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    load_dotenv()
    client = Groq(api_key=os.getenv("groq_api_key"))

    return chunks, model, index, client


# ⭐ load from cache
chunks, model, index, client = load_system()


# ---------------- ASK FUNCTION ----------------
def ask_rag(question):

    query = model.encode([question])

    k = 3  # smaller = faster
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
