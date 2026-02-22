#!/usr/bin/env python3

from embeddings import OpenAIEmbeddings
from config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY

embedding = OpenAIEmbeddings(OPENAI_API_KEY, EMBEDDING_MODEL_NAME)

text1 = "Kelly got a kitty for Christmas"
embedding1 = embedding.get_embedding(text1)

text2 = "Jonas wanted a pet as a gift"
embedding2 = embedding.get_embedding(text2)

# Unrelated text
text3 = "I had a dream last night"
embedding3 = embedding.get_embedding(text3)

print(f"text1: {text1}")
print(f"=> embedding length: {len(embedding1)}")
print(f"text2: {text2}")
print(f"=> embedding length: {len(embedding2)}")
print(f"text3: {text3}")
print(f"=> embedding length: {len(embedding3)}")
print()

print(f"Cosine similarity (1 and 2): {embedding.cosine_similarity(embedding1, embedding2)}")
print(f"Cosine similarity (1 and 3): {embedding.cosine_similarity(embedding1, embedding3)}")
print(f"Cosine similarity (2 and 3): {embedding.cosine_similarity(embedding2, embedding3)}")