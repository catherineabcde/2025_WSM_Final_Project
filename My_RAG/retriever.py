from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from ollama import Client
import faiss
from utils import rrf_fusion, SimpleHit
import os

class HybridRetriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense Index
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama-gateway:11434')
        # ollama_host = 'http://localhost:11434'
        self.client = Client(host=ollama_host)

        if language == "zh":
            self.model_name = 'qwen3-embedding:0.6b'
        else:
            self.model_name = 'embeddinggemma:300m'
        
        embeddings = []
        for text in self.corpus:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])
            
        self.doc_embeddings = np.array(embeddings).astype('float32')
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)

    def retrieve(self, query, top_k=5):
        candidate_k = min(100, len(self.chunks))
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_hits = [SimpleHit(docid=idx, score=bm25_scores[idx]) for idx in sparse_top_indices]

        response = self.client.embeddings(model=self.model_name, prompt=query)
        query_vector = np.array([response['embedding']]).astype('float32')
        faiss.normalize_L2(query_vector)
        dense_scores, dense_indices = self.index.search(query_vector, candidate_k)
        
        dense_hits = []
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx != -1:
                dense_hits.append(SimpleHit(docid=idx, score=score))

        rrf_results = rrf_fusion(sparse_hits, dense_hits, k=60)
        top_chunks = []
        for docid, score in rrf_results[:top_k]:
            original_chunk = self.chunks[docid]
            emb = self.doc_embeddings[docid]
            result_chunk = {
                'page_content': original_chunk['page_content'],
                'metadata': {
                    'id': docid,
                    'score': score,
                    'type': 'hybrid_rf'
                },
                'embedding': emb # return embedding for later use
            }
            top_chunks.append(result_chunk)

        return top_chunks

def create_retriever(chunks, language):
    """Creates a hybrid retriever from document chunks."""
    return HybridRetriever(chunks, language)
