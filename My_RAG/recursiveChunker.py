from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import save_jsonl
from tqdm import tqdm
import os


def recursive_chunk(docs, language, chunk_size):
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_size // 5}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],  # Will split by these in order
    )
    
    chunks = []
    
    for doc in tqdm(docs, desc="Recursive Chunking"):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            original_text = doc['content']
            lang = doc['language']

            if lang == language:
                # Prepare metadata (everything except content)
                meta = doc.copy()
                meta.pop("content", None)
                
                # Split text into chunks (returns list of strings)
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for text_chunk in split_texts:
                        if text_chunk.strip():  # Skip empty chunks
                            chunks.append({
                                "page_content": text_chunk,
                                "metadata": meta,
                            })
                except Exception as e:
                    print(f"Error chunking doc: {e}")

    return chunks