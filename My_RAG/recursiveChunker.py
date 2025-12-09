from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def recursive_chunk(docs, language, chunk_size):
    """Split documents into chunks using recursive character splitting."""
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_size // 5}")
    if language == "en":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            length_function=len,
            is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", "：", " ", ""])
    
    chunks = []
    
    for doc in tqdm(docs, desc="Recursive Chunking"):   
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']

            if lang == language:
                meta = doc.copy()
                meta.pop("content", None)
                
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for text_chunk in split_texts:
                        if text_chunk.strip():
                            chunks.append({
                                "page_content": text_chunk,
                                "metadata": meta,
                            })
                except Exception as e:
                    print(f"Error chunking doc: {e}")
    
    print(f"Created {len(chunks)} chunks")
    return chunks