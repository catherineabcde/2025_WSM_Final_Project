from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import os

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", top_n=3, use_fp16=True):
        """
        初始化 Reranker 模型
        :param model_name: 使用的模型名稱 (預設 bge-reranker-v2-m3)
        :param top_n: 重排序後要保留前幾名
        :param use_fp16: 是否開啟半精度 (節省記憶體)
        """
        local_model_path = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-v2-m3")
        target_model = local_model_path if os.path.exists(local_model_path) else model_name
        self.model = FlagEmbeddingReranker(
            model=target_model,
            top_n=top_n,
            use_fp16=use_fp16
        )

    def rerank(self, nodes, query):
        """
        執行重排序
        :param nodes: 初步檢索到的節點列表 (LlamaIndex NodeWithScore)
        :param query: 使用者的查詢字串
        :return: 排序後的節點列表
        """
        # Rerank
        return self.model.postprocess_nodes(nodes, query_str=query)
