"""
Hybrid Retriever using LlamaIndex with Ollama embeddings (fully offline).
Combines BM25 sparse retrieval with Vector dense retrieval (RRF fusion).
Includes SimilarityPostprocessor for contextual compression.
"""
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from pyserini_bm25 import PyseriniBM25Retriever
import jieba
import os
from utils import load_ollama_config
#from flag_reranker import Reranker 
from flag_reranker_submit import Reranker

# Disable OpenAI defaults - use Ollama only (fully offline)
Settings.llm = None
Settings.embed_model = None


def load_stopwords(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

STOPWORDS_EN = load_stopwords('en_stopword')
STOPWORDS_ZH = load_stopwords('zh_stopword')


def tokenize(text: str) -> list[str]:
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    if is_chinese:
        tokens = list(jieba.cut(text))
        return [t for t in tokens if t.strip() and t not in STOPWORDS_ZH]
    else:
        tokens = text.lower().split()
        return [t for t in tokens if t not in STOPWORDS_EN]


class Retriever:
    def __init__(self, chunks, language="en", chunksize=1024, similarity_threshold=0.5):
        self.language = language
        
        # Convert to LlamaIndex nodes
        nodes = [
            TextNode(
                text=c['page_content'],
                metadata={**c.get('metadata', {}), 'chunk_id': i}
            ) for i, c in enumerate(chunks)
        ]
        
        # Select embedding model (Ollama - offline)
        model = "embeddinggemma:300m" if language == "en" else "qwen3-embedding:0.6b"
        self.embed_model = OllamaEmbedding(
            model_name=model,
            base_url=load_ollama_config()['host'],
            embed_batch_size=64,
            options={"num_parallel": 6}
        )
        Settings.embed_model = self.embed_model
        
        self.retrieve_topk = 100
        # 1. BM25 Retriever
        
        if isinstance(chunksize, (int, float)) or str(chunksize).replace('.', '').isdigit():
            bm25_index_path = f"./bm25_index_cache/{language}_chunksize{chunksize}"
        else:
            bm25_index_path = f"./bm25_index_cache/{language}_{chunksize}"
        bm25 = PyseriniBM25Retriever.from_defaults(
            nodes=nodes,
            language=language,
            similarity_top_k=self.retrieve_topk,
            index_path=bm25_index_path,
            k1=1.2,
            b=0.75,
        )
        
        vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=True)        
        vector = vector_index.as_retriever(similarity_top_k=self.retrieve_topk)
        if language == "zh":
            bm25_weight = 1
            vector_weight = 0
        else:
            bm25_weight = 1
            vector_weight = 0
        # 2. Hybrid Fusion (RRF)
        self.retriever = QueryFusionRetriever(
            retrievers=[bm25, vector],
            retriever_weights=[bm25_weight, vector_weight],
            similarity_top_k=self.retrieve_topk,
            num_queries=1,
            mode="relative_score",
        )
        # Cross-encoder rerank
        self.reranker_module = Reranker(
            top_n=self.retrieve_topk
        )
        
    def retrieve(self, query, top_k=None):
        if top_k is None:
            if self.language == "zh":
                top_k = 3
            else:
                top_k = 5
        # Get initial results
        init_nodes = self.retriever.retrieve(query)

        nodes = init_nodes[:20]
        final_nodes = self.reranker_module.rerank(nodes, query)
        final_nodes = final_nodes[:top_k]
                
        return [
            {
                'page_content': n.node.get_content(),
                'metadata': {
                    'score': n.score, # Use the reranked score
                    'type': 'hybrid_llamaindex_compressed',
                    **n.node.metadata
                }
            } for i, n in enumerate(final_nodes)
        ]


def create_retriever(chunks, language ,chunksize, similarity_threshold=0.5):
    return Retriever(chunks, language ,chunksize, similarity_threshold)
