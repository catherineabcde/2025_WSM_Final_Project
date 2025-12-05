from rank_bm25 import BM25Okapi
import jieba
import numpy as np
import ollama
import faiss
from utils import rrf_fusion, SimpleHit , load_embedding_config
import os
from tqdm import tqdm

class HybridRetriever:
    def __init__(self, chunks, language="en" , use_ollama=False, ollama_config=None):
        self.chunks = chunks
        self.language = language
        self.use_ollama = use_ollama
        self.ollama_config = ollama_config
        self.corpus = [chunk['page_content'] for chunk in chunks]
        
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense Index
        if self.use_ollama and self.ollama_config:
            print(f"Using Ollama model: {ollama_config['model']} at {ollama_config['host']}")
            self.client = ollama.Client(host=ollama_config['host'])
            self.model_name = ollama_config['model']
            
            # Generate embeddings using Ollama
            # Note: Ollama python client might not support batch embedding efficiently in all versions, 
            # but let's try to iterate if needed or use batch if supported.
            # The denseRetriever.py uses self.client.embeddings(model=..., prompt=...) which is single input.
            # We need to loop for corpus.
            print("Generating embeddings with Ollama...")
            embeddings = []
            batch_size = 32
            for i in tqdm(range(0, len(self.corpus), batch_size), desc="Generating embeddings"):
                batch_docs = self.corpus[i : i + batch_size]
                try:
                    response = self.client.embed(model=self.model_name, input=batch_docs)
                    embeddings.extend(response['embeddings'])
                except Exception as e:
                    print(f"Error embedding batch {i}: {e}")
                    raise e
        else:
            raise ValueError("No Ollama config found. Please check your configuration files.")
            
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
    ollama_config = load_embedding_config(language=language)

    if not ollama_config:
        raise ValueError("Failed to load Ollama config. Please check your configuration files.")

    return HybridRetriever(chunks, language, use_ollama=True, ollama_config=ollama_config)
