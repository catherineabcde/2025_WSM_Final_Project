from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
# from pyserini.search.hybrid import HybridSearcher
from pyserini.encode import TctColBertQueryEncoder, AutoQueryEncoder
import json
from utils import rrf_fusion

EN_SPARSE_INDEX = 'en_sparse_indexes/collections'
EN_DENSE_INDEX = 'en_dense_index'

ZH_SPARSE_INDEX = 'zh_sparse_indexes/collections'
ZH_DENSE_INDEX = 'zh_dense_index'

class HybridRetriever:
    def __init__(self, language="en"):
        self.language = language
        if language == "zh":
            # Sparse
            self.sparse_searcher = LuceneSearcher(ZH_SPARSE_INDEX)
            self.sparse_searcher.set_language('zh')
            # Dense
            encoder = AutoQueryEncoder('BAAI/bge-base-zh-v1.5', pooling='mean', device='cpu')
            self.dense_searcher = FaissSearcher(ZH_DENSE_INDEX, encoder)
        else:
            # Sparse
            self.sparse_searcher = LuceneSearcher.from_prebuilt_index(EN_SPARSE_INDEX)
            # Dense
            encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco', device='cpu')
            self.dense_searcher = FaissSearcher.from_prebuilt_index(EN_DENSE_INDEX, encoder)
        # Hybrid
        self.hybrid_searcher = HybridSearcher(self.dense_searcher, self.sparse_searcher)

    def retrieve(self, query_text, top_k=5):
        hits = self.hybrid_searcher.search(query_text, k=top_k)
        sparse_ratio = 0.3
        dense_ratio = 0.7
        retrieved_chunks = []
        for hit in hits:
            original_doc = self.sparse_searcher.doc(hit.docid)
            if original_doc:
                raw_json = json.loads(original_doc.raw())
                chunk = {
                    'page_content': raw_json.get('contents', ''), # 對應 chunker 的 'page_content'
                    'metadata': {
                        'id': hit.docid,
                        'score': hit.score,
                        'sparse_ratio': hit.sparse_ratio,
                        'dense_ratio': hit.dense_ratio
                    }
                }
                retrieved_chunks.append(chunk)
    
        return retrieved_chunks    
def create_retriever(chunks, language):
    return HybridRetriever(language)