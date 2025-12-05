from ollama import Client
from pathlib import Path
import yaml
from utils import load_ollama_config


def generate_answer(query, context_chunks, language):
    formatted_context = []
    for i, chunk in enumerate(context_chunks):
        # 根據 metadata 加入公司名稱或時間
        meta_info = ""
        if 'metadata' in chunk:
         if'company_name' in chunk['metadata']:
            company = chunk['metadata'].get('company_name', '') 
            meta_info = f"[Company: {company}]" if company else ""
         elif 'court_name' in chunk['metadata']:
            court = chunk['metadata'].get('court_name', '') 
            meta_info = f"[Court: {court}]" if court else ""
         elif 'hospital_patient_name' in chunk['metadata']:
            patient = chunk['metadata'].get('hospital_patient_name', '') 
            meta_info = f"[Patient: {patient}]" if patient else ""
        formatted_context.append(f"--- Document Fragment {i+1} {meta_info} ---\n{chunk['page_content']}")

    context = "\n\n".join(formatted_context)
   
    if language=="en":
        prompt = f"""You are an intelligent assistant with strong logical reasoning capabilities. Your task is to answer the user's question based **ONLY** on the provided context.

### Context Data:
{context}

### User Question:
{query}

### Reasoning Steps (Instructions):
Before generating the final answer, please follow these steps for logical reasoning:
1. **Timeline Extraction**: 
   - Carefully search for **specific timestamps** related to the question.
   - **WARNING**: Do not rely solely on the broad date at the beginning of a paragraph! A paragraph may contain sub-events that occurred in different months (e.g., "Significant changes occurred in March 2021, where the CEO was appointed in January 2021" -> The CEO appointment time must be January). Ensure you lock onto the precise time immediately adjacent to the specific event.
2. **Entity Check**: 
   - Verify which **Company** or **Person** the action or event belongs to. Avoid attributing actions of Company A to Company B.
3. **Information Synthesis**: 
   - If the question is a summary (e.g., "Summarize all changes"), scan all document fragments to ensure no events from the beginning or end of the timeline are omitted.
4. **Numerical Comparison**:
   - When comparing numerical values, first convert all numbers to the same unit before comparison.
   - Example: 120 million = 12000 (in units of 10,000), 50 million = 5000 (in units of 10,000).
5. **Generation**: 
   - Generate the final answer based on the analysis above.

### Formatting Constraints:
- **Language**: Answer strictly in **English**.
- **Partial Answers**: If only **partial information** is available, provide what you can based on available data. 
- **Refusal**: Reply "Unable to answer." **ONLY** if there is **absolutely no relevant information** in the context. If you find ANY related information, you MUST attempt to answer.
- **Conciseness**: Provide the conclusion directly; do not output your internal thought process.

### Example:
Question: Company A's acquisition amount is 150 million, Company B's acquisition amount is 80 million. Which is larger?
Answer: 150 million > 80 million, so Company A's acquisition amount is larger.

### Answer:
""" 
    elif language=="zh":
        prompt = f"""你是一个拥有强大逻辑推理能力的智能助手。你的任务是**仅**基于提供的上下文回答用户的问题。

### 上下文数据：
{context}

### 用户问题：
{query}

### 思考步骤：
在生成最终答案之前，请遵循以下步骤进行逻辑推理：
1. **时间线提取**：
   - 仔细寻找与问题相关的**具体时间戳**。
   - **警示**：不要只看段落开头的大日期！段落中可能包含发生在不同月份的子事件（例如："2021年3月发生了变动，其中任命CEO是在2021年1月" -> CEO任命时间应为1月）。务必锁定事件旁边最精确的时间。
2. **实体核对**：
   - 确认该动作或事件是属于哪个**公司**或**人物**的，避免将A公司的事件安在B公司头上。
3. **信息整合**：
   - 如果问题是摘要性质（如"总结所有变动"），请扫描所有文档片段，不要遗漏开头或结尾的月份。
4. **数值比较**：
   - 进行数值比较时，请先将所有数字统一转换为相同单位（例如：万元）再进行比较。
   - 例如：1.2亿元 = 12000万元，5000万元 = 5000万元。
5. **生成回答**：
   - 基于上述分析生成最终答案。

### 格式约束：
- **语言**：请使用**简体中文**回答。
- **部分回答**：如果只有**部分信息**可用，请基于现有数据尽量回答。
- **拒答**：**只有**当上下文中**完全没有任何相关信息**时，才回复："无法回答。" 如果找到任何相关信息，你必须尝试回答。
- **简洁**：直接给出结论，不需要输出你的思考过程。

### 示例：
问题：A公司收购金额1.5亿元，B公司收购金额8000万元，哪个更大？
回答：1.5亿元 = 15000万元 > 8000万元，所以A公司收购金额更大。

### 回答：
"""
    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(
            model=ollama_config["model"], 
            prompt=prompt,
            stream=False,
            options={
               "temperature": 0.2,
               "num_ctx":131072
            }
         )
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