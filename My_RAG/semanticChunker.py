from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from utils import load_ollama_config
from tqdm import tqdm
import nltk


def chunk_documents(docs, language):
    # Initialize embedding model for semantic chunking
    # Using the same model as the retriever for consistency
    model = "embeddinggemma:300m" if language == "en" else "qwen3-embedding:0.6b"
    
    # Use FastOllamaEmbeddings for semantic chunker
    print(f"Using Ollama model: {model}")
    embeddings = OllamaEmbeddings(
        model=model,
        base_url=load_ollama_config()['host']
    )

     # Collect texts and metadatas for batch processing
    texts = []
    metadatas = []
    
    # documents preprocessing
    for doc in docs:
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']
            if language == "en":
                text = original_text
            else:
                text = original_text.replace("\n", "")
            
            if lang == language:
                texts.append(text)
                meta = doc.copy()
                meta.pop("content", None)
                metadatas.append(meta)

    # set semantic chunker
    if language=="en":
        # Not yet tested
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80,
            buffer_size=3,
            sentence_split_regex="(?<=\n)"
        )
    elif language=="zh":
        # tested
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80,
            buffer_size=3,
            sentence_split_regex="(?<=。|！|？)"
        )
    
    # batch chunking
    chunks = []
    if texts:
        print(f"Batch chunking {len(texts)} documents...")
        batch_size = 32
        semantic_docs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch Chunking"):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            try:
                batch_result = semantic_chunker.create_documents(batch_texts, metadatas=batch_metadatas)
                semantic_docs.extend(batch_result)
            except Exception as e:
                print(f"Error chunking batch {i}: {e}")
        
        for i, semantic_doc in enumerate(semantic_docs):
            chunk = {
                "page_content": semantic_doc.page_content,
                "metadata": semantic_doc.metadata,
            }
            
            if chunk["page_content"] != "":
                chunks.append(chunk)

    return chunks