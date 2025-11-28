from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader as IndexReader
from pyserini.analysis import get_lucene_analyzer, Analyzer
import json

class BM25Okapi:
    def __init__(self, index, k1=1.2, b=0.75,language='zh',k=5):
        self.searcher = LuceneSearcher(index)
        self.searcher.set_bm25(k1=k1, b=b)
        self.searcher.set_language(language)
        self.k = k
    
    def retrieve(self, query):
        hits = self.searcher.search(query, k=self.k)
        results = []

        for hit in hits:
            # 1. 從 hit 取得 docid
            doc_id = hit.docid
    
            # 2. 使用 searcher 根據 docid 抓取完整文件
            doc = self.searcher.doc(doc_id)
            
            # 3. 解析內容 (doc.raw() 回傳的是 JSON 字串)
            if doc:
                raw_json = doc.raw()
                content = json.loads(raw_json)
                
                # Extract text content
                text = content.get('contents', '')
                
                # store metadata
                metadata = content
                metadata.pop('contents', None)
                metadata.pop('id', None)
                metadata.pop('chunk_id', None)
                metadata.pop('language', None) 
                metadata.pop('doc_id', None) 

                results.append({
                    'page_content': text,
                    'metadata': metadata
                })
        return results




