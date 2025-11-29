from ollama import Client
from pathlib import Path
import yaml


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


def generate_answer(query, context_chunks, language):
    formatted_context = []
    for i, chunk in enumerate(context_chunks):
        # 根據 metadata 加入公司名稱或時間
        meta_info = ""
        if 'metadata' in chunk:
            company = chunk['metadata'].get('company_name', '') 
            meta_info = f"[Company: {company}]" if company else ""
        formatted_context.append(f"--- Document Fragment {i+1} {meta_info} ---\n{chunk['page_content']}")

    context = "\n\n".join(formatted_context)
   
    if language=="en":
        prompt = f"""You are an intelligent assistant with strong logical reasoning capabilities. Your task is to answer the user's question based **ONLY** on the provided context.\

### Context Data:\n{context}\n
### User Question:\n{query}\n

### Reasoning Steps (Instructions):\n
Before generating the final answer, please follow these steps for logical reasoning:\n
1. **Timeline Extraction**: \n
   - Carefully search for **specific timestamps** related to the question.\n
   - **WARNING**: Do not rely solely on the broad date at the beginning of a paragraph! A paragraph may contain sub-events that occurred in different months (e.g., "Significant changes occurred in March 2021, where the CEO was appointed in January 2021" -> The CEO appointment time must be January). Ensure you lock onto the precise time immediately adjacent to the specific event.\
2. **Entity Check**: \n
   - Verify which **Company** or **Person** the action or event belongs to. Avoid attributing actions of Company A to Company B.\
3. **Information Synthesis**: \n
   - If the question is a summary (e.g., "Summarize all changes"), scan all document fragments to ensure no events from the beginning or end of the timeline are omitted.\
4. **Generation**: \n
   - Generate the final answer based on the analysis above.\n

### Formatting Constraints:\n
- **Language**: Answer strictly in **English**.\n
- **Refusal**: If the answer is not present in the context, reply exactly: "Unable to answer." Do NOT make up an answer.\n
- **Conciseness**: Provide the conclusion directly; do not output your internal thought process.\n

### Answer:
""" 
    elif language=="zh":
        prompt = f"""你是一个拥有强大逻辑推理能力的智能助手。你的任务是**仅**基于提供的上下文回答用户的问题。\n

### 上下文数据 (Context)：\n
{context}\n

### 用户问题 (Question)：\n
{query}\n

### 思考步骤 (Instructions)：\n
在生成最终答案之前，请遵循以下步骤进行逻辑推理：\n
1. **时间线提取 (Timeline Extraction)**：\n
   - 仔细寻找与问题相关的**具体时间戳**。\n
   - **警示**：不要只看段落开头的大日期！段落中可能包含发生在不同月份的子事件（例如：“2021年3月发生了变动，其中任命CEO是在2021年1月” -> CEO任命时间应为1月）。务必锁定事件旁边最精确的时间。\n
2. **实体核对 (Entity Check)**：\n
   - 确认该动作或事件是属于哪个**公司**或**人物**的，避免将A公司的事件安在B公司头上。\n
3. **信息整合 (Synthesis)**：\n
   - 如果问题是摘要性质（如“总结所有变动”），请扫描所有文档片段，不要遗漏开头或结尾的月份。\n
4. **生成回答 (Generation)**：\n
   - 基于上述分析生成最终答案。\n

### 格式约束：\n
- **语言**：请使用**简体中文**回答。\n
- **拒答**：如果上下文中确实没有答案，请直接回复：“无法回答。” 不要编造。\n
- **简洁**：直接给出结论，不需要输出你的思考过程。\n

### 回答：\n
"""

    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(model=ollama_config["model"], prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"Error using Ollama Python client: {e}"


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)