from ollama import Client
from utils import load_ollama_config
import re
import os

class LLMReranker:
    def __init__(self, language="en"):
        self.language = language
        self.config = load_ollama_config()
        self.client = Client(host=self.config["host"])
        self.model = self.config["model"]

    def rerank(self, query, chunks, top_k=None):
        if not chunks:
            return []

        scored_chunks = []
        for chunk in chunks:
            if 'contextual_summary' in chunk['metadata']:
                text = f"{chunk['metadata']['contextual_summary']}\n{chunk['page_content']}"
            else:
                text = chunk['page_content']
            try:
                score = self._get_score(query, text)
                chunk['metadata']['rerank_score'] = score
                scored_chunks.append(chunk)
            except Exception as e:
                print(f"Error reranking chunk: {e}")
                chunk['metadata']['rerank_score'] = 0
                scored_chunks.append(chunk)

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x['metadata']['rerank_score'], reverse=True)

        if top_k:
            return scored_chunks[:top_k]
        return scored_chunks

    def _get_score(self, query, content):
        if self.language == "zh":
            prompt = f"""請評估以下文檔與問題的相關性，並給出 0 到 10 的評分。
0 代表完全不相關，10 代表非常相關。
請只輸出一行數字，不要有任何其他文字。

問題：{query}

文檔：{content}

評分："""
        else:
            prompt = f"""Rate the relevance of the following document to the question on a scale of 0 to 10.
0 means completely irrelevant, 10 means highly relevant.
Output ONLY a single number from 0 to 10. Do not explain.

Question: {query}

Document: {content}

Score:"""

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0, # Deterministic
                "num_ctx": 4096,
            }
        )
        
        output = response["response"].strip()
        # Extract number using regex in case LLM is verbose
        match = re.search(r'\d+(\.\d+)?', output)
        if match:
            return float(match.group())
        return 0.0
