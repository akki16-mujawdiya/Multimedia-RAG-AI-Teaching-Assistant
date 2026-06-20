import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 10


# Load embeddings
df = joblib.load("embeddings.joblib")

print("\nColumns inside embeddings.joblib:")
print(df.columns)


# Create embedding
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



# LLM inference
def inference(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response_json = response.json()

    return response_json["response"]



# User input
incoming_query = input("\nAsk a question: ").strip()

print("\nThinking...")


# Query embedding
question_embedding = create_embedding(incoming_query)

# Stored embeddings
valid_embeddings = np.vstack(df["embedding"].values)

# Similarities
similarities = cosine_similarity(
    valid_embeddings,
    [question_embedding]
).flatten()

# Top K chunks
top_indices = similarities.argsort()[::-1][:TOP_K]

new_df = df.iloc[top_indices]
video_names = {
    "14": "Introduction to CSS",
    "15": "Inline, Internal & External CSS",
    "17": "CSS Selectors MasterClass",
    "18": "CSS Box Model",
    "19": "CSS Fonts, Text & Color Properties"
}

new_df["title"] = new_df["number"].astype(str).map(
    lambda x: video_names.get(x, f"Video {x}")
)

# DEBUG
print("\n========== RETRIEVED CHUNKS ==========\n")

print(
    new_df[
        ["number", "title", "start", "end", "text"]
    ]
)

# Context
context = ""

for _, row in new_df.iterrows():

    context += f"""
Video Number : {row['number']}
Video Title  : {row['title']}
Timestamp    : {row['start']} - {row['end']}
Content      : {row['text']}

"""



# Prompt
prompt = f"""
You are an AI teaching assistant for a web development course.

Use ONLY the information below.

{context}

Rules:

1. Never make up video numbers.
2. Never invent timestamps.
3. Use only provided videos.
4. If answer is not present, reply exactly:

Not found in provided videos.

Question:

{incoming_query}

Answer format:

Video Number:
Video Title:
Timestamp:
Explanation:
"""

# Save prompt
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# Generate response
response = inference(prompt)

# Save response
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)

# Output
print("\n==========================")
print("AI RESPONSE")
print("==========================\n")

print(response)
