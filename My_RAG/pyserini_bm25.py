"""
Pyserini BM25 Retriever wrapped as LlamaIndex BaseRetriever.
This allows Pyserini's BM25 to be used with LlamaIndex's QueryFusionRetriever.
Uses Pyserini's built-in Lucene analyzer for consistent tokenization.
"""
import os
import json
import tempfile
import shutil
from typing import List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from pyserini.search.lucene import LuceneSearcher


class PyseriniBM25Retriever(BaseRetriever):
    """
    A LlamaIndex-compatible retriever that uses Pyserini's BM25 implementation.
    Wraps Pyserini's LuceneSearcher to return NodeWithScore objects.
    
    Uses Pyserini's built-in Lucene analyzer for tokenization, ensuring
    consistency between indexing and search phases.
    """
    
    def __init__(
        self,
        nodes: List[TextNode],
        language: str = "en",
        similarity_top_k: int = 100,
        index_path: Optional[str] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize PyseriniBM25Retriever.
        
        Args:
            nodes: List of TextNode objects to index
            language: Language code ('en' or 'zh')
            similarity_top_k: Number of top results to return
            index_path: Optional path to save/load the Lucene index.
                       If None, uses a temporary directory.
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        super().__init__()
        self.nodes = nodes
        self.language = language
        self.similarity_top_k = similarity_top_k
        self.k1 = k1
        self.b = b
        
        # Create node lookup by docid
        self._node_dict = {}
        
        # Build or load index
        self._index_path = index_path
        self._temp_dir = None
        self._searcher = self._build_index(nodes)
        
        # Set BM25 parameters
        self._searcher.set_bm25(k1, b)
        
        # Set language-specific analyzer for search
        # This ensures query tokenization matches index tokenization
        if language == "zh":
            self._searcher.set_language("zh")
    
    def _build_index(self, nodes: List[TextNode]) -> LuceneSearcher:
        """
        Build Pyserini Lucene index from nodes.
        Uses Pyserini's built-in analyzer for tokenization.
        """
        # Determine index path
        if self._index_path and os.path.exists(self._index_path):
            # Load existing index
            print(f"Loading existing Pyserini index from: {self._index_path}")
            searcher = LuceneSearcher(self._index_path)
            # Rebuild node dict from stored raw data
            self._rebuild_node_dict(searcher, len(nodes))
            return searcher
        
        # Create temporary directory for index if not specified
        if self._index_path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pyserini_bm25_")
            index_dir = self._temp_dir
        else:
            index_dir = self._index_path
            os.makedirs(index_dir, exist_ok=True)
        
        print(f"Building Pyserini BM25 index at: {index_dir}")
        
        # Prepare documents for Pyserini (raw text, let Lucene handle tokenization)
        docs_dir = tempfile.mkdtemp(prefix="pyserini_docs_")
        docs_file = os.path.join(docs_dir, "docs.jsonl")
        
        try:
            with open(docs_file, 'w', encoding='utf-8') as f:
                for i, node in enumerate(nodes):
                    doc_id = f"doc_{i}"
                    # Store node reference
                    self._node_dict[doc_id] = node
                    
                    # Use original content - Lucene analyzer will handle tokenization
                    content = node.get_content()
                    
                    doc = {
                        "id": doc_id,
                        "contents": content,  # Raw text, Lucene handles tokenization
                        # Store original content and metadata in raw field for retrieval
                        "raw": json.dumps({
                            "original_content": content,
                            "metadata": node.metadata
                        }, ensure_ascii=False)
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            # Build index using Pyserini with language-specific analyzer
            import subprocess
            
            cmd = [
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", docs_dir,
                "--index", index_dir,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "4",
                "--storeRaw"
            ]
            
            # Add language-specific analyzer settings
            # For Chinese, use CJKAnalyzer; for English, use default analyzer
            if self.language == "zh":
                cmd.extend(["--language", "zh"])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Indexing stderr: {result.stderr}")
                raise RuntimeError(f"Failed to build Pyserini index: {result.stderr}")
            
            print(f"Successfully built index with {len(nodes)} documents")
            
        finally:
            # Clean up temporary docs directory
            shutil.rmtree(docs_dir, ignore_errors=True)
        
        return LuceneSearcher(index_dir)
    
    def _rebuild_node_dict(self, searcher: LuceneSearcher, expected_count: int):
        """Rebuild node dict from stored raw data when loading existing index."""
        for i in range(expected_count):
            doc_id = f"doc_{i}"
            doc = searcher.doc(doc_id)
            if doc:
                try:
                    raw_str = doc.raw()
                    if raw_str:
                        # Pyserini returns: {"id": ..., "contents": ..., "raw": ...}
                        outer_data = json.loads(raw_str)
                        
                        # Try to get from our nested "raw" field first
                        inner_raw = outer_data.get("raw")
                        if inner_raw:
                            inner_data = json.loads(inner_raw)
                            content = inner_data.get("original_content")
                            if content:
                                self._node_dict[doc_id] = TextNode(
                                    text=content,
                                    metadata=inner_data.get("metadata", {})
                                )
                                continue
                        
                        # Fallback to "contents" field from outer data
                        content = outer_data.get("contents")
                        if content:
                            self._node_dict[doc_id] = TextNode(text=content)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using Pyserini BM25 search.
        
        Args:
            query_bundle: Query bundle containing the query string
            
        Returns:
            List of NodeWithScore objects
        """
        query = query_bundle.query_str
        
        # Search using Pyserini - analyzer handles tokenization automatically
        hits = self._searcher.search(query, k=self.similarity_top_k)
        
        results = []
        for hit in hits:
            doc_id = hit.docid
            score = hit.score
            
            # Try to get node from our lookup dict first
            if doc_id in self._node_dict:
                node = self._node_dict[doc_id]
                results.append(NodeWithScore(node=node, score=score))
                continue
            
            # Fallback: reconstruct node from stored raw data
            doc = self._searcher.doc(doc_id)
            if not doc:
                continue
                
            node = None
            try:
                raw_str = doc.raw()
                if raw_str:
                    # Pyserini returns: {"id": ..., "contents": ..., "raw": ...}
                    outer_data = json.loads(raw_str)
                    
                    # Try to get content from our nested "raw" field first
                    inner_raw = outer_data.get("raw")
                    if inner_raw:
                        inner_data = json.loads(inner_raw)
                        content = inner_data.get("original_content")
                        if content:
                            node = TextNode(
                                text=content,
                                metadata=inner_data.get("metadata", {})
                            )
                    
                    # Fallback to "contents" field from outer data
                    if node is None:
                        content = outer_data.get("contents")
                        if content:
                            node = TextNode(text=content)
                            
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            
            if node is not None:
                results.append(NodeWithScore(node=node, score=score))
        
        return results
    
    def __del__(self):
        """Clean up temporary directory if created."""
        if hasattr(self, '_temp_dir') and self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    @classmethod
    def from_defaults(
        cls,
        nodes: List[TextNode],
        language: str = "en",
        similarity_top_k: int = 100,
        index_path: Optional[str] = None,
        k1: float = 1.2,
        b: float = 0.75,
        **kwargs,
    ) -> "PyseriniBM25Retriever":
        """
        Factory method to create PyseriniBM25Retriever.
        Matches the interface of LlamaIndex's BM25Retriever.from_defaults().
        
        Args:
            nodes: List of TextNode objects to index
            language: Language code ('en' or 'zh')
            similarity_top_k: Number of top results to return
            index_path: Optional path to save/load the Lucene index
            k1: BM25 k1 parameter
            b: BM25 b parameter
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            PyseriniBM25Retriever instance
        """
        return cls(
            nodes=nodes,
            language=language,
            similarity_top_k=similarity_top_k,
            index_path=index_path,
            k1=k1,
            b=b,
        )
