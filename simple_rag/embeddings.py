#!/usr/bin/env python3

from typing import List
import numpy as np
from openai import OpenAI

class BaseEmbeddings:
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        Get embedding for the given text.
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        cos(\theta) = (A \dot B) / (||A|| * ||B||)
        """
        dot = np.dot(vector1, vector2)
        mag = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if mag == 0:
            # avoid dividing by 0
            return 0
        return dot / mag
    
class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        # Create OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_embedding(self, text) -> List[float]:
        # From https://github.com/openai/openai-python/issues/418, seems
        # like newline used to impact performance; this is prob not an
        # issue anymore but just keeping this here.
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding