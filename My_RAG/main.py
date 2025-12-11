from tqdm import tqdm
from pathlib import Path
from utils import load_jsonl, save_jsonl
from retriever import create_retriever
from recursiveChunker import recursive_chunk
# from generator import generate_answer, judge_relevance
from generator import generate_answer
import argparse
from llama_query_rewriter import rewrite_query
# from reranker import LLMReranker
from merge_model import merge_files
import os

def main(query_path, docs_path, language, output_path):
    
    # For reranker model merging
    # model_dir = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-v2-m3")
    # If model.safetensors does not exist, merge the model
    # model_path = os.path.join(model_dir, "model.safetensors")
    
    # if not Path(model_path).exists():
    #     merge_files(model_dir, "model.safetensors", "model.safetensors.part_")
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    if language=="zh":
        chunks = recursive_chunk(docs_for_chunking, language, chunk_size=128)
    else:
        chunks = recursive_chunk(docs_for_chunking, language, chunk_size=512)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")

        # Define rewrite mode based on language strategies
    if language == 'zh':
        # Chinese strategy: Multi (High accuracy and robustness)
        rewrite_mode = 'none'
        #print("Using strategy: Multi-Query Rewrite")
    else:
        # English strategy: Routing (Best balance of retrieval and generation)
        rewrite_mode = 'none'
        #print("Using strategy: Semantic Routing")

    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query["query"]["content"]
        if language == "zh":
            FINAL_TOP_K = 5
        elif language == "en":
            FINAL_TOP_K = 3

        
        # choose mode: "none" / "multi" / "hyde" / "decompose" / "stepback"
        rewritten_queries = rewrite_query(
            query_text,
            language=language,
            mode=rewrite_mode,      # or "multi", "hyde", "decompose", "stepback" ,"none"
            num_queries=3      # optional, for "multi"
        )
       
        CANDIDATE_FACTOR = 3

        all_chunks = []
        for q in rewritten_queries:
            # for hyde mode
            #retrieved = retriever.retrieve(q, top_k=FINAL_TOP_K)
            if hasattr(q, 'query_text'): # 處理不同 rewrite 回傳格式的防呆
                q_text = q.query_text
            else:
                q_text = str(q)
                
            retrieved = retriever.retrieve(q_text, top_k=FINAL_TOP_K)
            # for multi mode
            # retrieved = retriever.retrieve(q.query_text, top_k=FINAL_TOP_K*CANDIDATE_FACTOR)
            all_chunks.extend(retrieved)

        # Deduplicate by retriever_id
        seen, unique = set(), []
        for c in all_chunks:
            key = c.get("metadata", {}).get("id") or id(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        # retrieved_chunks = unique
        final_chunks = unique[:FINAL_TOP_K]
        
        
        # 5. Generate Answer
        print("Generating answer...")
        answer = generate_answer(query_text, final_chunks, language)
        print(f"Generated Answer: {answer}") #for test prompt

        query["prediction"]["content"] = answer
        if final_chunks:
            query["prediction"]["references"] = [c["metadata"].get("original_content", c["page_content"]) for c in final_chunks]   
        else:
            query["prediction"]["references"] = []  

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
