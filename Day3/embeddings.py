from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

text = 'Gue ganteng banget'

def get_embeddings(text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )

    embedding_data = response.data[0].embedding
    return embedding_data

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    dot_product = np.dot(vector1, vector2)
    magnitue1 = np.linalg.norm(vector1)
    magnitue2 = np.linalg.norm(vector2)
    
    return dot_product/(magnitue1 * magnitue2)

text1 = 'I love dogs'
text2 = 'I hate aliens'

emb1 = get_embeddings(text1)
emb2 = get_embeddings(text2)

similarity = cosine_similarity(emb1, emb2)

print(f'Text 1: {text1}')
print(f'Text 2: {text2}')
print(f'Cosine Similatiry: {similarity}')
