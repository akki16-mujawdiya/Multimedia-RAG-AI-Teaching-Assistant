import requests
import os
import json
import pandas as pd
import joblib

# -----------------------------
# Create embeddings
# -----------------------------
def create_embedding(text_list):

    cleaned_list = [
        str(text) if pd.notna(text) and text != "" else " "
        for text in text_list
    ]

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": cleaned_list
        }
    )

    response = r.json()

    if "embeddings" not in response:
        raise Exception(response)

    return response["embeddings"]


# -----------------------------
# Read all json files
# -----------------------------
json_files = os.listdir("jsons")

all_chunks = []

chunk_id = 0

for file in json_files:

    if not file.endswith(".json"):
        continue

    print("Processing :", file)

    with open(f"jsons/{file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]

    texts = [c["text"] for c in chunks]

    embeddings = create_embedding(texts)

    for i, chunk in enumerate(chunks):

        row = {
            "chunk_id": chunk_id,
            "number": chunk["number"],
            "title": chunk["title"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "embedding": embeddings[i]
        }

        all_chunks.append(row)

        chunk_id += 1


# -----------------------------
# DataFrame
# -----------------------------
df = pd.DataFrame(all_chunks)

print("\nColumns:")
print(df.columns)

# save
joblib.dump(df, "embeddings.joblib")

print("\nembeddings.joblib created successfully.")
