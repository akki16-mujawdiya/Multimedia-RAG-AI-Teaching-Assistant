import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Settings
# ==========================
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 10

# ==========================
# Load embeddings once
# ==========================
@st.cache_resource
def load_data():
    return joblib.load("embeddings.joblib")

df = load_data()

# ==========================
# Create embedding
# ==========================
def create_embedding(text):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": [text]
        }
    )

    response_json = response.json()

    if "embeddings" not in response_json:
        raise Exception(response_json)

    return np.array(response_json["embeddings"][0])

# ==========================
# LLM inference
# ==========================
def inference(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# ==========================
# UI
# ==========================
st.title("🎓 Multimedia RAG AI Teaching Assistant")

query = st.text_input("Ask a question")

if st.button("Search"):

    if query != "":

        with st.spinner("Thinking..."):

            question_embedding = create_embedding(query)

            valid_embeddings = np.vstack(df["embedding"].values)

            similarities = cosine_similarity(
                valid_embeddings,
                [question_embedding]
            ).flatten()

            top_indices = similarities.argsort()[::-1][:TOP_K]

            new_df = df.iloc[top_indices]

            context = ""

            for _, row in new_df.iterrows():

                context += f"""
Video Number : {row['number']}
Video Title : {row['title']}
Timestamp : {row['start']} - {row['end']}
Content : {row['text']}

"""

            prompt = f"""
You are an AI teaching assistant.

Use ONLY the information below.

{context}

Rules:
1. Never invent timestamps.
2. Never invent video numbers.
3. Use only provided information.
4. If answer not found say:
Not found in provided videos.

Question:
{query}
"""

            answer = inference(prompt)

            st.success("Answer Generated")

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Chunks")
            st.dataframe(
                new_df[
                    ["number", "title", "start", "end", "text"]
                ]
            )