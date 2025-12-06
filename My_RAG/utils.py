import jsonlines
from pathlib import Path
import yaml

class SimpleHit:
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score

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
def rrf_fusion(sparse_hits, dense_hits, sparse_ratio, dense_ratio,k=60):
    scores = {}
    for rank, hit in enumerate(sparse_hits):
        docid = hit.docid
        if docid not in scores:
            scores[docid] = 0
        scores[docid] += sparse_ratio * 1 / (k + rank + 1)
    for rank, hit in enumerate(dense_hits):
        docid = hit.docid
        if docid not in scores:
            scores[docid] = 0
        scores[docid] += dense_ratio * 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]

def load_embedding_config(language) -> dict:
    # Try to find config in parent directories
    current_path = Path(__file__).resolve()
    
    # We'll look up to 3 levels up for a 'configs' directory
    configs_folder = None
    for i in range(4):
        candidate = current_path.parents[i] / "configs"
        if candidate.exists():
            configs_folder = candidate
            break
            
    if not configs_folder:
         # Fallback to hardcoded path if relative search fails
         configs_folder = Path(__file__).parent.parent / "configs"

    # Priority: config_local.yaml > config_submit.yaml
    config_files = ["config_local.yaml", "config_submit.yaml"]
    
    config_data = None
    for fname in config_files:
        fpath = configs_folder / fname
        if fpath.exists():
            try:
                with open(fpath, "r") as file:
                    config_data = yaml.safe_load(file)
                print(f"Loaded config from {fpath}")
                break
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
    
    if not config_data:
        return None

    if language == "en":
        return config_data.get("EN")
    elif language == "zh":
        return config_data.get("ZH")
    
    return None