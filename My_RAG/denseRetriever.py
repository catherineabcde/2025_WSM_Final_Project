import numpy as np
import faiss
from utils import rrf_fusion, SimpleHit
import ollama
from utils import load_embedding_config
from tqdm import tqdm

class DenseRetriever:
    def __init__(self, chunks, language="en", use_ollama=False, ollama_config=None):
        self.chunks = chunks
        self.language = language
        self.use_ollama = use_ollama
        self.ollama_config = ollama_config
        self.corpus = [chunk['page_content'] for chunk in chunks]
        

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
        # 2. Dense Retrieval (Embedding + FAISS)
        # Generate query embedding
        if self.use_ollama:
            response = self.client.embeddings(model=self.model_name, prompt=query)
            query_vector = np.array([response['embedding']]).astype('float32')
        else:
            raise ValueError("No Ollama config found. Please check your configuration files.")
            
        # Normalize query vector for Cosine Similarity (Inner Product on normalized vectors)
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        dense_scores, dense_indices = self.index.search(query_vector, top_k)
        
        # Create SimpleHit objects for RRF fusion
        dense_hits = []
        for score, idx in zip(dense_scores[0], dense_indices[0]):
            if idx != -1:   
                dense_hits.append(SimpleHit(docid=idx, score=score))

        
        # 4. Format Results
        top_chunks = []
        for hit in dense_hits:
            docid = hit.docid
            score = hit.score
            # Retrieve the original chunk using the integer index
            # Use .copy() to avoid modifying the original chunk in memory
            original_chunk = self.chunks[docid].copy()
            original_chunk['metadata'] = original_chunk['metadata'].copy()
            
            # Inject RRF score into metadata for downstream usage
            original_chunk['metadata']['score'] = score
            top_chunks.append(original_chunk)

        return top_chunks

def dense_retriever(chunks, language):
    """Creates a hybrid retriever from document chunks."""
    ollama_config = load_embedding_config(language=language)
    
    if not ollama_config:
        raise ValueError("Failed to load Ollama config. Please check your configuration files.")

    return DenseRetriever(chunks, language, use_ollama=True, ollama_config=ollama_config)
