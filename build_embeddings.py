import requests 
import pandas as pd 

import joblib
import numpy as np
# Load your chunks 
df = pd.read_json("output.json") 
#  Generate embeddings
response = requests.post( "http://localhost:11434/api/embed", 
                         json={
                             "model": "nomic-embed-text", 
                             "input": df["text"].astype(str).tolist() } ) 
embeddings = response.json()["embeddings"] 
df["embedding"] = list(np.array(embeddings)) 
# Save
joblib.dump(df, "embeddings.joblib")
print("embeddings.joblib created successfully!")