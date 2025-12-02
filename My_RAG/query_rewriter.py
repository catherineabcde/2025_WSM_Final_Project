import json
import re
from typing import List
from ollama import Client
from utils import load_ollama_config


_OLLAMA_CLIENT: Client | None = None
_OLLAMA_MODEL: str | None = None


def _get_ollama_client() -> tuple[Client, str]:

    global _OLLAMA_CLIENT, _OLLAMA_MODEL

    if _OLLAMA_CLIENT is not None and _OLLAMA_MODEL is not None:
        return _OLLAMA_CLIENT, _OLLAMA_MODEL

    cfg = load_ollama_config()
    if not cfg or "host" not in cfg or "model" not in cfg:
        raise RuntimeError("Invalid Ollama configuration. Please check configs/config_*.yaml.")

    _OLLAMA_CLIENT = Client(host=cfg["host"])
    _OLLAMA_MODEL = cfg["model"]
    return _OLLAMA_CLIENT, _OLLAMA_MODEL


def _strip_code_fence(text: str) -> str:
    """
    Remove common Markdown code fence wrappers around JSON. 
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_parse_queries(raw: str, fallback_n: int | None = None) -> List[str]:
    
    raw = _strip_code_fence(raw)
    queries: List[str] = []

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            for q in data["queries"]:
                if isinstance(q, str):
                    q = q.strip()
                    if q:
                        queries.append(q)
    except Exception:
        pass

    if not queries:
        lines = [ln.strip() for ln in raw.splitlines()]
        for ln in lines:
            if ln:
                queries.append(ln)
        if fallback_n is not None:
            queries = queries[:fallback_n]

    seen = set()
    dedup: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            dedup.append(q)

    return dedup


def rewrite_query_multi(query: str, language: str = "zh", num_queries: int = 3) -> List[str]:

    client, model = _get_ollama_client()

    if language == "zh":
        prompt = f"""你是一個查詢重寫助手。根據使用者的原始問題，產生 {num_queries} 個意思相同、
但用不同說法的查詢句子，適合用來檢索長文件。

請嚴格只輸出下列 JSON 格式：
{{"queries": ["句子1", "句子2", "句子3"]}}

不要加入任何額外文字或註解。

原始問題：
{query}
"""
    else:
        prompt = f"""You are a query rewriting assistant. Based on the user's question,
generate {num_queries} alternative queries that keep the same meaning but use different wording.
The rewritten queries should be suitable for document retrieval.

You MUST output only valid JSON in the form:
{{"queries": ["sentence 1", "sentence 2", "sentence 3"]}}

Do not add explanations or extra text.

Original question:
{query}
"""

    resp = client.generate(model=model, prompt=prompt)
    raw = resp.get("response", "")
    queries = _safe_parse_queries(raw, fallback_n=num_queries)

    if query not in queries:
        queries.insert(0, query)

    return queries


def rewrite_query_hyde(query: str, language: str = "zh") -> List[str]:
    print("Rewriting query with hyde...")
    client, model = _get_ollama_client()

    if language == "zh":
        prompt = f"""你是一个用来协助文件检索的回答生成器。
根据下面的问题，写出一段合理、正式且精简的回答，就像文件中的一段说明文字。
请在内容中尽量包含关键实体名称与重要细节，以利检索相关文件。

问题：
{query}

请只输出回答内容，不要加任何额外说明：
"""
    else:
        prompt = f"""You are an assistant that generates a hypothetical answer to help document retrieval.
Given the following question, write a concise and plausible answer paragraph as if it were
a passage from a document. Try to include important entities and key details to improve retrieval.

Question:
{query}

Please output only the answer paragraph, with no explanations:
"""

    resp = client.generate(model=model, prompt=prompt)
    answer = resp.get("response", "").strip()
    if not answer:
        return [query]

    combined = f"{query}\n{answer}"
    return [combined]


def rewrite_query_decompose(query: str, language: str = "zh", max_subqueries: int = 3) -> List[str]:
    
    client, model = _get_ollama_client()

    if language == "zh":
        prompt = f"""你是一個問題分解助手。請將下面的複雜問題拆成最多 {max_subqueries} 個
可以獨立回答的子問題，每個子問題都應該明確且具體，適合用來做文件檢索。

請嚴格只輸出下列 JSON 格式：
{{"queries": ["子問題1", "子問題2", "子問題3"]}}

不要加入任何額外文字或註解。

原始問題：
{query}
"""
    else:
        prompt = f"""You are a question decomposition assistant.
Please decompose the following complex question into at most {max_subqueries} simpler
sub-questions that can be answered independently and are suitable for document retrieval.

You MUST output only valid JSON in the form:
{{"queries": ["sub-question 1", "sub-question 2", "sub-question 3"]}}

Do not add explanations or any extra text.

Original question:
{query}
"""

    resp = client.generate(model=model, prompt=prompt)
    raw = resp.get("response", "")
    subqs = _safe_parse_queries(raw, fallback_n=max_subqueries)

    result: List[str] = []
    if query not in subqs:
        result.append(query)
    result.extend(subqs)
    return result


def rewrite_query_stepback(query: str, language: str = "zh", num_stepbacks: int = 2) -> List[str]:
    
    client, model = _get_ollama_client()

    if language == "zh":
        prompt = f"""你是一個資料檢索助手。請根據下面的具體問題，
提出 {num_stepbacks} 個更一般化、更高層次的「step-back 問題」，
這些問題應該描述同一主題的背景或核心概念，有助於檢索到更完整的相關文件。

請嚴格只輸出下列 JSON 格式：
{{"queries": ["step-back 問題1", "step-back 問題2"]}}

不要加入任何額外文字或註解。

原始問題：
{query}
"""
    else:
        prompt = f"""You are an information retrieval assistant. Based on the specific question below,
propose {num_stepbacks} more general, higher-level "step-back" questions that describe the
underlying topic or background. These questions should help retrieve broader and more
comprehensive documents.

You MUST output only valid JSON in the form:
{{"queries": ["step-back question 1", "step-back question 2"]}}

Do not add explanations or extra text.

Original question:
{query}
"""

    resp = client.generate(model=model, prompt=prompt)
    raw = resp.get("response", "")
    stepbacks = _safe_parse_queries(raw, fallback_n=num_stepbacks)

    result: List[str] = []
    result.append(query)
    for q in stepbacks:
        if q not in result:
            result.append(q)
    return result


def rewrite_query(
    query: str,
    language: str = "zh",
    mode: str = "none",
    **kwargs,
) -> List[str]:

    mode = (mode or "none").lower()

    if mode == "multi":
        num_queries = int(kwargs.get("num_queries", 3))
        return rewrite_query_multi(query, language=language, num_queries=num_queries)
    if mode == "hyde":
        return rewrite_query_hyde(query, language=language)
    if mode in ("decompose", "decomposition"):
        max_subqueries = int(kwargs.get("max_subqueries", 3))
        return rewrite_query_decompose(query, language=language, max_subqueries=max_subqueries)
    if mode in ("stepback", "step_back"):
        num_stepbacks = int(kwargs.get("num_stepbacks", 2))
        return rewrite_query_stepback(query, language=language, num_stepbacks=num_stepbacks)

    return [query]
