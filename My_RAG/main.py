from tqdm import tqdm
from pathlib import Path
from utils import load_jsonl, save_jsonl
#from chunker import chunk_documents
from retriever import create_retriever
from denseRetriever import dense_retriever 
from dynamicChunker import chunk_documents
from recursiveChunker import recursive_chunk
from generator import generate_answer
import argparse
from query_rewriter import rewrite_query

def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    if language=="zh":
        chunks = chunk_documents(docs_for_chunking, language)
    else:
        chunks = recursive_chunk(docs_for_chunking, language, chunk_size=1024)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    if language=="zh":
        retriever = create_retriever(chunks, language)
    else:
        retriever = dense_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query["query"]["content"]
        if language == "zh":
            FINAL_TOP_K = 5
        elif language == "en":
            FINAL_TOP_K = 5
        '''
        # choose mode: "none" / "multi" / "hyde" / "decompose" / "stepback"
        rewritten_queries = rewrite_query(
            query_text,
            language=language,
            mode="hyde",      # or "multi", "hyde", "decompose", "stepback" ,"none"
            num_queries=3      # optional, for "multi"
        )
       
        # CANDIDATE_FACTOR = 4

        all_chunks = []
        for q in rewritten_queries:
            # for hyde mode
            retrieved = retriever.retrieve(q, top_k=FINAL_TOP_K)
            # for multi mode
            # retrieved = retriever.retrieve(q, top_k=FINAL_TOP_K*CANDIDATE_FACTOR)
            all_chunks.extend(retrieved)

        # Deduplicate by retriever_id
        unique = {}
        for c in all_chunks:
            meta = c.get("metadata", {})
            key = meta.get("id")
            if key is None:
                key = id(c)
            if key not in unique:
                unique[key] = c

        final_chunks = list(unique.values())[:FINAL_TOP_K]
        '''
        final_chunks = retriever.retrieve(query_text, top_k=FINAL_TOP_K)
        # 5. Generate Answer
        print("Generating answer...")
        answer = generate_answer(query_text, final_chunks, language)
        print(f"Generated Answer: {answer}") #for test prompt

        query["prediction"]["content"] = answer
        if final_chunks:
            query["prediction"]["references"] = [c["page_content"] for c in final_chunks]   
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
