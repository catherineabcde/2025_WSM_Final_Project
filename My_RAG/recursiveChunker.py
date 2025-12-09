from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from utils import load_ollama_config
from ollama import Client

def _generate_doc_context(language, doc_text):
    """
    Generate contextual summary for a document using LLM.
    This summary will be shared by all chunks in the document.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    if language == "zh":
        prompt = f"""请用简体中文为以下文件写一个简短的摘要（20-50字），说明这份文件的主题、涉及的主要实体或事件。只输出摘要，不要其他内容。

文件内容：
{doc_text[:3000]}

摘要："""
    else:
        prompt = f"""Write a brief summary (20-30 words) for the following document in english, describing its main topic, entities, or events. Output ONLY the summary.

Document:
{doc_text[:3000]}

Summary:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
            }
        )
        context = response["response"].strip()
        return context
    except Exception as e:
        print(f"Error generating context: {e}")
        return ""

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
                # Generate contextual summary for the document
                doc_context = _generate_doc_context(language, original_text)
                meta = doc.copy()
                meta.pop("content", None)
                
                try:
                    split_texts = text_splitter.split_text(original_text)
                    for text_chunk in split_texts:
                        if text_chunk.strip():
                            chunks.append({
                                "page_content": text_chunk,
                                "metadata": {
                                    "contextual_summary": doc_context,
                                    **meta
                                },
                            })
                except Exception as e:
                    print(f"Error chunking doc: {e}")
    
    print(f"Created {len(chunks)} chunks")
    return chunks