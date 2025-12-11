import requests
import numpy as np
import os
from typing import List
from llama_index.core.schema import NodeWithScore

# --- 1. 助教提供的 API Wrapper (直接複製過來) ---
class RemoteFlagReranker:
    """
    Fake FlagReranker class: internally calls a remote API.
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    def compute_score(self, pairs, max_length=1024):
        payload = {"pairs": [{"text1": a, "text2": b} for a, b in pairs]}
        resp = requests.post(self.api_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"API request failed ({resp.status_code}): {resp.text}")
        scores = resp.json()["scores"]
        return np.array(scores)

# --- 2. 修改後的 Reranker Class (整合 LlamaIndex 邏輯) ---
class Reranker:
    def __init__(self, api_url="http://ollama-gateway:11434/rerank", top_n=3):
        """
        初始化 Reranker
        :param api_url: 助教提供的 API Endpoint
        :param top_n: 重排序後要保留前幾名
        """
        print(f"🚀 Loading Remote Reranker from: {api_url}...")
        self.model = RemoteFlagReranker(api_url=api_url)
        self.top_n = top_n
        self.batch_size = 32  # 助教規定的 API 限制

    def rerank(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """
        執行重排序
        :param nodes: 初步檢索到的節點列表 (LlamaIndex NodeWithScore)
        :param query: 使用者的查詢字串
        :return: 排序後的節點列表
        """
        if not nodes:
            return []

        # Prepare pairs [Query, Document] for API input
        pairs = [[query, node.node.get_content()] for node in nodes]
        
        all_scores = []
        
        # Batch processing
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            
            try:
                # Call API to get scores
                batch_scores = self.model.compute_score(batch_pairs)
                all_scores.extend(batch_scores)
            except Exception as e:
                print(f"⚠️ Rerank API error in batch {i}: {e}")
                # Fill with low score when error occurs
                all_scores.extend([-999.0] * len(batch_pairs))

        # Write scores b    ack to NodeWithScore objects
        for node, score in zip(nodes, all_scores):
            node.score = float(score)  # Ensure converted to Python float

        # Sort nodes by score (descending)
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

        # Return top N nodes
        return sorted_nodes[:self.top_n]