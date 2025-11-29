from langchain_experimental.text_splitter import SemanticChunker
from ollama import Client
from generator import load_ollama_config

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

class OllamaEmbeddings:
    """
    Minimal embeddings wrapper compatible with LangChain's Embeddings interface.
    It calls Ollama's /embeddings endpoint.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, host: str | None = None):
        # Reuse generator.py config to get correct host (local vs submit)
        if host is None:
            ollama_cfg = load_ollama_config()
            host = ollama_cfg["host"]
        self.client = Client(host=host)
        self.model_name = model_name

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            res = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(res["embedding"])
        return embeddings

    def embed_query(self, text):
        res = self.client.embeddings(model=self.model_name, prompt=text)
        return res["embedding"]


def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):

    chunks = []
    semantic_chunker = None

    for doc_index, doc in enumerate(docs):
        if "content" not in doc or "language" not in doc:
            continue
        text = doc["content"]
        lang = doc["language"]
        if not isinstance(text, str):
            continue

        # Only process docs that match the target language parameter
        if lang != language:
            continue

        # ---- Case 1: Chinese (zh) → semantic chunking ----
        if lang == "zh":
            # Initialize semantic chunker only once
            if semantic_chunker is None:
                ollama_cfg = load_ollama_config()
                host = ollama_cfg["host"]
                embeddings = OllamaEmbeddings(model_name=EMBEDDING_MODEL, host=host)
                semantic_chunker = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile",
                )

            # Create semantic chunks 
            semantic_docs = semantic_chunker.create_documents([text])

            merged_chunks = []
            current_chunk_text = ""
            MIN_CHUNK_SIZE = 500

            for semantic_doc in semantic_docs:
                content = semantic_doc.page_content

                if not current_chunk_text:
                    current_chunk_text = content
                else:
                    if len(current_chunk_text) < MIN_CHUNK_SIZE:
                        current_chunk_text += "\n" + content
                    else:
                        merged_chunks.append(current_chunk_text)
                        current_chunk_text = content

            if current_chunk_text:
                merged_chunks.append(current_chunk_text)

            for i, chunk_text in enumerate(merged_chunks):
                chunk_metadata = doc.copy()
                chunk_metadata.pop("content", None)
                chunk_metadata["chunk_index"] = i

                chunk = {
                    "page_content": chunk_text,
                    "metadata": chunk_metadata,
                }
                chunks.append(chunk)

        # ---- Case 2: English (en) → fixed-length chunking ----
        elif lang == "en":
            text_len = len(text)
            start_index = 0
            chunk_count = 0

            while start_index < text_len:
                end_index = min(start_index + chunk_size, text_len)

                chunk_metadata = doc.copy()
                chunk_metadata.pop("content", None)
                chunk_metadata["chunk_index"] = chunk_count

                chunk = {
                    "page_content": text[start_index:end_index],
                    "metadata": chunk_metadata,
                }
                chunks.append(chunk)

                start_index += chunk_size - chunk_overlap
                chunk_count += 1

    return chunks