from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import os
import json

def _generate_chunk_context(language, doc_text, chunk_text, metadata=None):
    """
    Generate contextual description for a specific chunk using Ollama.
    This follows Anthropic's Contextual Retrieval approach.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # Truncate doc_text to avoid token limits (keep first 6000 chars for smaller models)
    doc_text_truncated = doc_text[:6000]
    
    # Extract subject name from metadata
    subject_name = ""
    if metadata:
        if "company_name" in metadata:
            subject_name = metadata["company_name"]
        elif "hospital_patient_name" in metadata:
            subject_name = metadata["hospital_patient_name"]
        elif "court_name" in metadata:
            subject_name = metadata["court_name"]
    
    if language == "zh":
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\n重要：本文档的主体是「{subject_name}」。"
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

以下是我们想要在整个文档中定位的块：
<chunk>
{chunk_text}
</chunk>

任务：为这个块生成一个上下文描述（50-80字），说明这段内容在整个文档中的位置和作用。{subject_instruction}

要求：
- 必须用简体中文回答
- 必须说明这段内容位于文档的哪个部分（如：文档开头、第二部分、结尾部分、财务指标部分等）
- 必须包含主体名称（公司名称/法院名称/医院名称）
- 如果是法律文档，必须包含法院名称和被告人/当事人姓名
- 如果是病历文档，必须包含医院名称和患者姓名
- 禁止直接复制原文内容
- 禁止使用代词如"该公司"、"该患者"、"本文档"等

格式示例：「这段内容位于[文档位置]，描述了[主体名称]的[主要内容]。」

请直接输出上下文描述："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\nIMPORTANT: The subject of this document is \"{subject_name}\"."
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Task: Generate a short context (30-50 words) describing WHERE this chunk is located in the document and its role.{subject_instruction}

Requirements:
- You MUST respond in English only
- MUST specify the location in the document (e.g., "at the beginning", "in the financial section", "at the end")
- MUST include the explicit subject name (company name/court name/hospital name)
- For legal documents, include both court name and defendant/party names
- For medical records, include both hospital name and patient name
- DO NOT copy the original text directly
- DO NOT use pronouns like "the company", "this document", "it", etc.

Format example: "This section, located in [document position], describes [subject name]'s [main content]."

Output ONLY the context description:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],  # Uses your configured model
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        )
        context = response["response"].strip()
        return context
    except Exception as e:
        print(f"Error generating chunk context: {e}")
        return ""

def recursive_chunk(docs, language, chunk_size):
    """Split documents into chunks using recursive character splitting."""
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_size // 5}")

    # Build cache path
    cache_path = f"./chunk_cache/{language}_contextual_chunksize{chunk_size}"
    
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunk cache hit: {cache_path}")
        return chunks

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