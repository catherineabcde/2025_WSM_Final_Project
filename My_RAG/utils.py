import jsonlines

def load_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def save_jsonl(file_path, data):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)

# For Hybrid Retriever
def rrf_fusion(sparse_hits, dense_hits, k=60):
    scores = {}
    for rank, hit in enumerate(sparse_hits):
        docid = hit.docid
        if docid not in scores:
            scores[docid] = 0
        scores[docid] += 1 / (k + rank + 1)
    for rank, hit in enumerate(dense_hits):
        docid = hit.docid
        if docid not in scores:
            scores[docid] = 0
        scores[docid] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)