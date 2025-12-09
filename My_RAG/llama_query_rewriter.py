from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from utils import load_ollama_config

# LlamaIndex imports
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import ToolMetadata as LlamaToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class ToolMetadata:
    name: str
    description: str

@dataclass
class TransformedQuery:
    query_text: str
    tool_name: str = "default_retriever"
    metadata: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# LlamaIndex LLM Singleton
# -----------------------------------------------------------------------------

_LLM_INSTANCE: Optional[Ollama] = None

def _get_llm() -> Ollama:
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    cfg = load_ollama_config()
    host = cfg.get("host", "http://localhost:11434") if cfg else "http://localhost:11434"
    model_name = cfg.get("model", "llama3") if cfg else "llama3"

    # Initialize Ollama LLM
    # Explicitly limit context window to prevent OOM (defaults might be too high)
    _LLM_INSTANCE = Ollama(
        model=model_name, 
        base_url=host, 
        request_timeout=120.0, 
        context_window=4096,
        additional_kwargs={"num_ctx": 4096}
    )
    
    # Optional: Set as global default, though we often use it explicitly
    Settings.llm = _LLM_INSTANCE
    return _LLM_INSTANCE

# -----------------------------------------------------------------------------
# (a) Routing-based Transform (using LLMSingleSelector)
# -----------------------------------------------------------------------------

def rewrite_with_routing(
    query: str, 
    tools: List[ToolMetadata], 
    language: str = "zh"
) -> List[TransformedQuery]:
    """
    Decide which tool(s) should be used using LlamaIndex LLMSingleSelector.
    """
    llm = _get_llm()
    selector = LLMSingleSelector.from_defaults(llm=llm)
    
    # Convert internal ToolMetadata to LlamaIndex ToolMetadata
    llama_tools = [
        LlamaToolMetadata(name=t.name, description=t.description) 
        for t in tools
    ]
    
    # Select
    result = selector.select(choices=llama_tools, query=query)
    
    transformed = []
    if result.selections:
        for sel in result.selections:
            # internal tools list index
            idx = sel.index
            selected_tool = tools[idx]
            transformed.append(TransformedQuery(
                query_text=query,
                tool_name=selected_tool.name,
                metadata={"reason": sel.reason, "ind": sel.index}
            ))
    
    if not transformed:
        transformed.append(TransformedQuery(query_text=query, tool_name="default_retriever"))
        
    return transformed

# -----------------------------------------------------------------------------
# (b) Multi-query Rewrite (Custom Prompt with LLM)
# -----------------------------------------------------------------------------

def rewrite_multi_query(
    query: str, 
    language: str = "zh", 
    num_queries: int = 3
) -> List[TransformedQuery]:
    """
    Generate diverse queries using LlamaIndex LLM and PromptTemplate.
    """
    llm = _get_llm()
    
    if language == "zh":
        prompt_str = (
            "你是一個查詢重寫助手。根據使用者的原始問題，產生 {num_queries} 個意思相同、"
            "但用不同說法或角度的查詢句子，適合用來檢索長文件。\n"
            "請每一行只列出一個查詢句子，不要加編號或額外符號。\n\n"
            "原始問題：{query}\n\n"
            "查詢句子："
        )
    else:
        prompt_str = (
            "You are a query rewriting assistant. Generate {num_queries} alternative search queries "
            "related to the following input query.\n"
            "Please listing one query per line, without numbers or bullets.\n\n"
            "Query: {query}\n\n"
            "Queries:"
        )

    prompt = PromptTemplate(prompt_str)
    response = llm.predict(prompt, num_queries=num_queries, query=query)
    
    # Parse lines
    queries = [line.strip() for line in response.split('\n') if line.strip()]
    
    if query not in queries:
        queries.insert(0, query)
        
    return [TransformedQuery(query_text=q, tool_name="default_retriever") for q in queries]

# -----------------------------------------------------------------------------
# (c) HyDE-style Rewrite (using HyDEQueryTransform)
# -----------------------------------------------------------------------------

def rewrite_hyde(
    query: str, 
    language: str = "zh"
) -> List[TransformedQuery]:
    """
    Generate hypothetical answer using LlamaIndex HyDEQueryTransform.
    """
    llm = _get_llm()
    
    # HyDEQueryTransform in LlamaIndex uses a default prompt (usually English).
    # We can override the prompt_template if needed for Chinese.
    hyde = HyDEQueryTransform(include_original=True, llm=llm)
    
    if language == "zh":
        # Custom prompt for Chinese HyDE
        hyde_prompt = (
            "你是一個用來協助文件檢索的回答生成器。\n"
            "根據下面的問題，寫出一段合理、正式且精簡的回答，就像文件中的一段說明文字。\n"
            "請在內容中盡量包含關鍵實體名稱與重要細節。\n\n"
            "問題：{query_str}\n\n"
            "回答："
        )
        hyde = HyDEQueryTransform(
            include_original=True, 
            llm=llm,
            hyde_prompt=PromptTemplate(hyde_prompt)
        )

    # run() returns a QueryBundle
    bundle = hyde.run(query)
    
    # The 'custom_embedding_strs' usually contains the hypothetic answer(s).
    # If include_original=True, the query_str is the original.
    # To use HyDE effectively for retrieval, we often concatenate or use the embedding string.
    
    # In LlamaIndex, `custom_embedding_strs` is a list of strings used for embedding.
    # We will combine them or return the main one.
    
    combined_text = bundle.query_str
    if bundle.custom_embedding_strs:
        # Usually contains the generated answer
        combined_text += "\n" + "\n".join(bundle.custom_embedding_strs)
        
    return [TransformedQuery(
        query_text=combined_text,
        tool_name="default_retriever",
        metadata={"type": "hyde", "embedding_strs": bundle.custom_embedding_strs}
    )]

# -----------------------------------------------------------------------------
# (d) Sub-question Decomposition (using LLMQuestionGenerator)
# -----------------------------------------------------------------------------

def rewrite_subquestions(
    query: str, 
    tools: List[ToolMetadata], 
    language: str = "zh",
    max_subqueries: int = 3
) -> List[TransformedQuery]:
    """
    Decompose question using LlamaIndex LLMQuestionGenerator.
    """
    llm = _get_llm()
    
    # Convert tools
    llama_tools = [
        LlamaToolMetadata(name=t.name, description=t.description) 
        for t in tools
    ]
    
    # Initialize Generator
    # Note: LLMQuestionGenerator from_defaults usually sets up a specific prompt.
    generator = LLMQuestionGenerator.from_defaults(llm=llm)
    
    # We might need to override prompts for Chinese, but LLMQuestionGenerator logic is complex to customize fully 
    # without subclassing or deep prompt injection. We'll try default first or inject prompt if API allows.
    # The `generate` method takes tools and query.
    
    from llama_index.core import QueryBundle
    try:
        sub_questions = generator.generate(tools=llama_tools, query=QueryBundle(query))
    except Exception as e:
        print(f"Warning: SubQuestion generation failed: {e}")
        return [TransformedQuery(query_text=query, tool_name="default_retriever")]
    
    results = []
    if sub_questions:
        for sq in sub_questions:
            results.append(TransformedQuery(
                query_text=sq.sub_question,
                tool_name=sq.tool_name,
                metadata={"type": "sub_question", "parent": query}
            ))
            
    if not results:
         results.append(TransformedQuery(query_text=query, tool_name="default_retriever"))
         
    return results

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def rewrite_query(
    query: str,
    language: str = "zh",
    mode: str = "none",
    **kwargs
) -> List[TransformedQuery]:
    
    mode = (mode or "none").lower()
    
    # Default tools
    tools = kwargs.get("tools")
    if not tools:
        # Mock tools for demonstration if none provided
        tools = [
            ToolMetadata(name="default_retriever", description="Useful for retrieving information about the user's document.")
        ]

    try:
        if mode == "routing":
            return rewrite_with_routing(query, tools, language=language)
        
        elif mode == "multi":
            num = int(kwargs.get("num_queries", 3))
            return rewrite_multi_query(query, language=language, num_queries=num)
        
        elif mode == "hyde":
            return rewrite_hyde(query, language=language)
        
        elif mode in ("subquestion", "decompose", "decomposition"):
            max_sub = int(kwargs.get("max_subqueries", 3))
            return rewrite_subquestions(query, tools, language=language, max_subqueries=max_sub)
            
    except Exception as e:
        print(f"Error in query rewrite mode '{mode}': {e}")
        # Fallback
        return [TransformedQuery(query_text=query, tool_name="default_retriever")]
    
    # Default/None
    return [TransformedQuery(query_text=query, tool_name="default_retriever")]
