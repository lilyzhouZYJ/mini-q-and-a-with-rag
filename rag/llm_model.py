#!/usr/bin/env python3

from typing import List
from openai import OpenAI

PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions about the following context:

{context}

Question: {question}

If the given question is not related to the context, say "I don't know."
Otherwise, answer the question based on the context.

Answer:
"""

class BaseModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate_response(self, question: str, context: str, history: List[dict]) -> str:
        raise NotImplementedError

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str) -> None:
        super().__init__(model_name)
        self.api_key = api_key

    def generate_response(self, question: str, context: str, history: List[dict]) -> str:
        client = OpenAI(api_key=self.api_key)
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        history.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model_name,
            messages=history,
            max_tokens = 200,
            temperature = 0.7)
        return response.choices[0].message.content