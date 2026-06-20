# from RRead_chunks import create_embedding
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
# from openai import OpenAI
# from config import api_key

# client = OpenAI(api_key = api_key)

def create_embedding(text_list):
    # Filter out any NaN or empty strings from the list to prevent Ollama from crashing
    # We replace None/NaN with a space " " so the API still gets a valid string
    cleaned_list = [str(text) if (pd.notna(text) and text != "") else " " for text in text_list]

    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "all-minilm",
        "input": cleaned_list
    })

    response_json = r.json()
    
    # Check if 'embeddings' exists in response to avoid KeyError
    if "embeddings" in response_json:
        return response_json["embeddings"]
    else:
        print(f"Error from Ollama: {response_json.get('error', 'Unknown Error')}")
        # Return a list of zeros as a placeholder if it fails
        return [None] * len(text_list)
    
    
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

# def inference_openai(prompt):
#     print("Thinking.....")
#     response = client.responses.create(
#         model="gpt-5",
#         input="prompt"
#     )
    
#     return response.output_text

df= joblib.load('embeddings.joblib')



incoming_query=input("Ask a question:")
question_embedding=create_embedding([incoming_query])[0]
# print(question_embedding)
# Optional: Save your work so you don't have to run it again
# df.to_pickle("embeddings_data.pkl")


#Find similarities of question_embedding with other embeddings

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])


prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
# print(response)

# response = inference_openai(prompt) 
# print(response)

with open("response.txt", "w") as f:
    f.write(response)

for index ,item in new_df.iterrows():
    print(index, item["title"], item["number"],item["text"], item["start"], item["end"])
    
    
    
    
    