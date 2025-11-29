from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    # Initialize embedding model for semantic chunking
    # Using the same model as the retriever for consistency
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Initialize Semantic Chunker
    # breakpoint_threshold_type="percentile" is a good starting point
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            lang = doc['language']
            
            # Only apply semantic chunking to the target language (or all if desired)
            # Here we apply it to the specified language, falling back to original logic or just skipping others
            if lang == language:
                # Create semantic chunks
                semantic_docs = semantic_chunker.create_documents([text])
                
                # Post-process: Merge small chunks
                merged_chunks = []
                current_chunk_text = ""
                
                # Minimum size for a chunk to be considered "complete"
                # We use chunk_size / 2 as a heuristic, or a fixed reasonable minimum like 500
                MIN_CHUNK_SIZE = 500 
                
                for semantic_doc in semantic_docs:
                    content = semantic_doc.page_content
                    
                    if not current_chunk_text:
                        current_chunk_text = content
                    else:
                        # If current buffer is too small, merge with next
                        if len(current_chunk_text) < MIN_CHUNK_SIZE:
                            current_chunk_text += "\n" + content
                        else:
                            # Current buffer is big enough, save it and start new
                            merged_chunks.append(current_chunk_text)
                            current_chunk_text = content
                
                # Don't forget the last chunk
                if current_chunk_text:
                    merged_chunks.append(current_chunk_text)

                for i, chunk_text in enumerate(merged_chunks):
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = i
                    
                    chunk = {
                        'page_content': chunk_text,
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    
    return chunks