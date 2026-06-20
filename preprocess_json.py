# # import requests
# # import os
# # import json
# # import pandas as pd

# # def create_embedding(text_list):
# #     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
# #     r = requests.post("http://localhost:11434/api/embed", json={
# #         "model": "bge-m3",
# #         "input": text_list
# #     })

# #     embedding = r.json()["embeddings"] 
# #     return embedding


# # jsons = os.listdir("jsons")  # List all the jsons 
# # my_dicts = []
# # chunk_id = 0

# # for json_file in jsons:
# #     with open(f"jsons/{json_file}") as f:
# #         content = json.load(f)
# #     print(f"Creating Embeddings for {json_file}")
# #     embeddings = create_embedding([c['text'] for c in content['chunks']])
       
# #     for i, chunk in enumerate(content['chunks']):
# #         chunk['chunk_id'] = chunk_id
# #         chunk['embedding'] = embeddings[i]
# #         chunk_id += 1
# #         my_dicts.append(chunk) 
# # # print(my_dicts)

# # df = pd.DataFrame.from_records(my_dicts)
# # print(df)
# # # a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# # # print(a)



# import requests
# import os
# import json
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib

# def create_embedding(text_list):
#     # Filter out any NaN or empty strings from the list to prevent Ollama from crashing
#     # We replace None/NaN with a space " " so the API still gets a valid string
#     cleaned_list = [str(text) if (pd.notna(text) and text != "") else " " for text in text_list]

#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "all-minilm",
#         "input": cleaned_list
#     })

#     response_json = r.json()
    
#     # Check if 'embeddings' exists in response to avoid KeyError
#     if "embeddings" in response_json:
#         return response_json["embeddings"]
#     else:
#         print(f"Error from Ollama: {response_json.get('error', 'Unknown Error')}")
#         # Return a list of zeros as a placeholder if it fails
#         return [None] * len(text_list)


# jsons = os.listdir("jsons")
# my_dicts = []
# chunk_id = 0

# for json_file in jsons:
#     # Ensure we only read .json files
#     if not json_file.endswith(".json"):
#         continue

#     with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
#         content = json.load(f)
    
#     print(f"Creating Embeddings for {json_file}")
    
#     # Extract all text chunks
#     texts = [c.get('text', "") for c in content['chunks']]
    
#     # Get embeddings in bulk
#     embeddings = create_embedding(texts)
       
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         # Safety check: ensure we don't go out of bounds
#         if i < len(embeddings):
#             chunk['embedding'] = embeddings[i]
#         else:
#             chunk['embedding'] = None
            
#         chunk_id += 1
#         my_dicts.append(chunk) 
#         # if(i==5):
#         #     break
#     # break

# # Convert to DataFrame
# df = pd.DataFrame.from_records(my_dicts)


# # Final clean up: Remove rows where embedding failed
# df = df.dropna(subset=['embedding'])
# # save this dataframe
# joblib.dump(df, 'embeddings.joblib')

# # print(df)


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
