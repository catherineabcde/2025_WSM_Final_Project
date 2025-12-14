from ollama import Client
from pathlib import Path
import yaml
from utils import load_ollama_config

def judge_relevance(query, chunk_text, language):
    """
    Use LLM to judge if a chunk is relevant to the query.
    Returns True if relevant, False otherwise.
    """
    if language == "zh":
        prompt = f"""判斷以下內容是否與問題相關。只回答「是」或「否」。

問題：{query}

內容：{chunk_text}

回答（是/否）："""
    else:
        prompt = f"""Determine if the following content is relevant to the question. Answer only "Yes" or "No".

Question: {query}

Content: {chunk_text}

Answer (Yes/No):"""

    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.1,
                "num_ctx": 2048,
            }
        )
        answer = response["response"].strip().lower()
        # Check for positive answers
        return any(word in answer for word in ["是", "yes", "相關", "relevant"])
    except Exception as e:
        # If error, assume relevant to avoid losing chunks
        return True

def _domain_router_en(user_query, context_snippet_preview):
    """
    Classify the domain of the query based on query and context (English version).
    Returns one of: 'FINANCE', 'MEDICAL', 'LAW', 'GENERAL'
    """
    prompt = f'''You are a classification expert.
Analyze the following user query and retrieved context snippet.
Determine the domain of the inquiry.

Output ONLY one of the following tags (just the word, no brackets):
- FINANCE (For questions about companies, revenue, stocks, acquisitions, business timeline, financial reports, ESG reports)
- MEDICAL (For questions about patients, symptoms, hospitals, diagnosis, medical history, treatments)
- LAW (For questions about court cases, judgments, lawsuits, contracts, legal disputes, defendants, plaintiffs, verdicts)
- GENERAL (For anything else)

User Query: {user_query}
Context Snippet: {context_snippet_preview[:500]}

Domain:'''
    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(
            model=ollama_config["model"], 
            prompt=prompt,
            stream=False, 
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
            })
        result = response["response"].strip().upper()
        # Extract domain from response
        if "FINANCE" in result:
            return "FINANCE"
        elif "MEDICAL" in result:
            return "MEDICAL"
        elif "LAW" in result:
            return "LAW"
        else:
            return "GENERAL"
    except Exception as e:
        print(f"Domain router error: {e}")
        return "GENERAL"


def _domain_router_zh(user_query, context_snippet_preview):
    """
    Classify the domain of the query based on query and context (Chinese version).
    Returns one of: 'FINANCE', 'MEDICAL', 'LAW', 'GENERAL'
    """
    prompt = f'''你是一位分类专家。
分析以下用户查询和检索到的上下文片段。
判断这个问题属于哪个领域。

只输出以下其中一个标签（只需输出单词，不要括号）：
- FINANCE（关于公司、营收、股票、并购、企业时间线、财务报告、ESG报告的问题）
- MEDICAL（关于病患、症状、医院、诊断、病史、治疗的问题）
- LAW（关于法院案件、判决、诉讼、合约、法律纠纷、被告、原告、裁决的问题）
- GENERAL（其他任何问题）

用户查询：{user_query}
上下文片段：{context_snippet_preview[:500]}

领域：'''
    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(
            model=ollama_config["model"], 
            prompt=prompt,
            stream=False, 
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
            })
        result = response["response"].strip().upper()
        # Extract domain from response
        if "FINANCE" in result:
            return "FINANCE"
        elif "MEDICAL" in result:
            return "MEDICAL"
        elif "LAW" in result:
            return "LAW"
        else:
            return "GENERAL"
    except Exception as e:
        print(f"Domain router error: {e}")
        return "GENERAL"


def domain_router(user_query, context_snippet_preview, language="en"):
    """
    Classify the domain of the query based on query and context.
    Selects the appropriate language version based on the language parameter.
    
    Args:
        user_query: The user's query string
        context_snippet_preview: Preview of the retrieved context
        language: Language code ('en' for English, 'zh' for Chinese)
    
    Returns one of: 'FINANCE', 'MEDICAL', 'LAW', 'GENERAL'
    """
    if language == "zh":
        return _domain_router_zh(user_query, context_snippet_preview)
    else:
        return _domain_router_en(user_query, context_snippet_preview)

def _get_domain_prompt_en(query, context, domain):
    """Get English prompt based on domain."""
    
    if domain == "FINANCE":
        return f"""You are a financial analysis expert. Your task is to answer financial questions based **ONLY** on the provided context.

### Context Data:
{context}

### User Question:
{query}

### Domain-Specific Guidelines for Finance:
1. **Timeline Extraction**: 
   - Carefully search for **specific timestamps** related to the question (year, month, date).
   - **WARNING**: Do not rely solely on the broad date at the beginning of a paragraph! A paragraph may contain sub-events that occurred in different months. Ensure you lock onto the precise time immediately adjacent to the specific event.
2. **Entity Check**: 
   - Verify which **Company** or **Person** the action or event belongs to. Avoid attributing actions of Company A to Company B.
3. **Numerical Analysis**:
   - When comparing monetary values, first convert all numbers to the same unit before comparison.
   - Pay attention to currency units (million, billion, thousand).
4. **Financial Indicator Focus**:
   - For financial metrics (revenue, profit, assets, liabilities), ensure you identify the correct reporting period.
   - Distinguish between different types of events: asset acquisition, equity acquisition, debt restructuring, dividend distribution, etc.

### Few-Shot Examples:

**Example 1 - Timeline extraction:**
Context: "In 2021, significant changes were made. The new CEO was appointed in January 2021. The company's merger was completed in March 2021."
Question: When was the new CEO appointed?
Answer: January 2021.

**Example 2 - Numerical comparison:**
Context: "Company A's acquisition was valued at 120 million yuan. Company B's acquisition was valued at 80 million yuan."
Question: Which company had the larger acquisition?
Answer: Company A had the larger acquisition at 120 million yuan, compared to Company B's 80 million yuan.

**Example 3 - Information not found:**
Context: "The company reported revenue of $50 million in 2022 and expanded its operations in Asia."
Question: What was the company's net profit in 2022?
Answer: Unable to answer.

### Formatting Constraints:
- **Language**: Answer strictly in **English**.
- **Partial Answers**: If only **partial information** is available, provide what you can based on available data.
- **Refusal**: Reply "Unable to answer." **ONLY** if there is **absolutely no relevant information** in the context.
- **Conciseness**: Provide the conclusion directly; do not output your internal thought process.

### Answer:
"""
    
    elif domain == "MEDICAL":
        return f"""You are a medical information specialist. Your task is to answer medical questions based **ONLY** on the provided context.

### Context Data:
{context}

### User Question:
{query}

### Domain-Specific Guidelines for Medical:
1. **Patient Identification**: 
   - Carefully identify which **patient** the information refers to.
   - Do not mix up medical records of different patients.
2. **Timeline of Medical Events**:
   - Pay close attention to **admission dates, diagnosis dates, treatment dates**.
   - Medical events often occur in sequence; ensure chronological accuracy.
3. **Medical Terminology**:
   - Look for specific diagnoses, symptoms, treatments, medications, and procedures.
   - If the question asks about a condition, look for related symptoms and diagnostic findings.
4. **Hospital/Department Information**:
   - Note which hospital, department, or attending physician is mentioned.
   - Different departments may have different records for the same patient.

### Few-Shot Examples:

**Example 1 - Patient identification:**
Context: "Patient John Smith was admitted on March 5, 2023. He presented with persistent cough and fever. Diagnosis: pneumonia."
Question: What was John Smith's diagnosis?
Answer: Pneumonia.

**Example 2 - Treatment information:**
Context: "The patient underwent appendectomy on April 10. Post-operative recovery was uneventful. Discharged on April 15."
Question: When was the patient discharged?
Answer: April 15.

**Example 3 - Information not found:**
Context: "Patient presented with headache and dizziness. CT scan was ordered."
Question: What medication was prescribed to the patient?
Answer: Unable to answer.

### Formatting Constraints:
- **Language**: Answer strictly in **English**.
- **Partial Answers**: If only **partial information** is available, provide what you can based on available data.
- **Refusal**: Reply "Unable to answer." **ONLY** if there is **absolutely no relevant information** in the context.
- **Conciseness**: Provide the conclusion directly; do not include reasoning steps.

### Answer:
"""
    
    elif domain == "LAW":
        return f"""You are a legal information specialist. Your task is to answer legal questions based **ONLY** on the provided context.

### Context Data:
{context}

### User Question:
{query}

### Domain-Specific Guidelines for Law:
1. **Party Identification**: 
   - Carefully identify the **plaintiff**, **defendant**, and other parties involved.
   - Do not confuse different parties or cases.
2. **Case Timeline**:
   - Pay close attention to **filing dates, hearing dates, judgment dates**.
   - Legal proceedings follow a sequence; ensure chronological accuracy.
3. **Legal Terminology**:
   - Look for specific charges, claims, verdicts, sentences, and legal provisions.
   - Identify the court name, case number, and presiding judge if mentioned.
4. **Judgment/Verdict Focus**:
   - Distinguish between different outcomes: guilty/not guilty, liable/not liable, damages awarded, etc.
   - Note any appeals or subsequent rulings.

### Few-Shot Examples:

**Example 1 - Party identification:**
Context: "In the case of Smith v. Johnson Corp, the plaintiff alleged breach of contract. The court ruled in favor of the plaintiff."
Question: Who won the case?
Answer: The plaintiff, Smith, won the case.

**Example 2 - Judgment details:**
Context: "The defendant was found guilty of fraud. The court sentenced the defendant to 3 years imprisonment and ordered restitution of $50,000."
Question: What was the sentence?
Answer: 3 years imprisonment and restitution of $50,000.

**Example 3 - Information not found:**
Context: "The case was filed on January 15, 2022. The defendant denied all allegations."
Question: What was the final verdict?
Answer: Unable to answer.

### Formatting Constraints:
- **Language**: Answer strictly in **English**.
- **Partial Answers**: If only **partial information** is available, provide what you can based on available data.
- **Refusal**: Reply "Unable to answer." **ONLY** if there is **absolutely no relevant information** in the context.
- **Conciseness**: Provide the conclusion directly; do not include reasoning steps.

### Answer:
"""
    
    else:  # GENERAL
        return f"""You are an intelligent assistant with strong logical reasoning capabilities. Your task is to answer the user's question based **ONLY** on the provided context.

### Context Data:
{context}

### User Question:
{query}

### Reasoning Steps (Instructions):
1. **Information Extraction**: Carefully read and extract relevant information from the context.
2. **Entity Check**: Verify the entities (people, organizations, locations) mentioned in the question.
3. **Information Synthesis**: If multiple fragments contain relevant information, synthesize them coherently.

### Few-Shot Examples:

**Example 1 - Direct extraction:**
Context: "The Eiffel Tower is located in Paris, France. It was completed in 1889."
Question: When was the Eiffel Tower completed?
Answer: 1889.

**Example 2 - Information synthesis:**
Context: "The project started in January. By March, the first phase was completed. The entire project was finished in June."
Question: How long did the project take from start to finish?
Answer: The project took 6 months, from January to June.

**Example 3 - Information not found:**
Context: "The company was founded in 2010 and is headquartered in New York."
Question: Who is the CEO of the company?
Answer: Unable to answer.

### Formatting Constraints:
- **Language**: Answer strictly in **English**.
- **Partial Answers**: If only **partial information** is available, provide what you can based on available data.
- **Refusal**: Reply "Unable to answer." **ONLY** if there is **absolutely no relevant information** in the context.
- **Conciseness**: Provide the conclusion directly; do not output your internal thought process.

### Answer:
"""


def _get_domain_prompt_zh(query, context, domain):
    """Get Chinese prompt based on domain."""
    
    if domain == "FINANCE":
        return f"""你是一位金融分析专家。你的任务是**仅**基于提供的上下文回答金融相关问题。

### 上下文数据：
{context}

### 用户问题：
{query}

### 金融领域特定指南：
1. **时间线提取**：
   - 仔细寻找与问题相关的**具体时间戳**（年份、月份、日期）。
   - **警示**：不要只看段落开头的大日期！段落中可能包含发生在不同月份的子事件。务必锁定事件旁边最精确的时间。
2. **实体核对**：
   - 确认该动作或事件是属于哪个**公司**或**人物**的，避免将A公司的事件安在B公司头上。
3. **数值分析**：
   - 进行数值比较时，请先将所有数字统一转换为相同单位再进行比较。
   - 注意货币单位（万元、亿元）。
4. **财务指标关注**：
   - 对于财务指标（营收、利润、资产、负债），确保正确识别报告期间。
   - 区分不同类型的事件：资产收购、股权收购、债务重组、股利分配等。

### 示例：

**示例1 - 时间线提取：**
上下文："2021年发生了重大变化。新任CEO于2021年1月上任。公司合并于2021年3月完成。"
问题：新任CEO何时上任？
回答：2021年1月。

**示例2 - 数值比较：**
上下文："A公司收购金额1.2亿元，B公司收购金额8000万元。"
问题：哪家公司的收购金额更大？
回答：A公司收购金额更大，为1.2亿元（即12000万元），大于B公司的8000万元。

**示例3 - 信息不足：**
上下文："公司2022年营业收入为5000万元，并在亚洲扩展了业务。"
问题：公司2022年的净利润是多少？
回答：无法回答。

### 格式约束：
- **语言**：请使用**简体中文**回答。
- **部分回答**：如果只有**部分信息**可用，请基于现有数据尽量回答。
- **拒答**：**只有**当上下文中**完全没有任何相关信息**时，才回复："无法回答。"
- **简洁**：直接给出结论，不需要输出思考过程。

### 回答：
"""
    
    elif domain == "MEDICAL":
        return f"""你是一位医疗信息专家。你的任务是**仅**基于提供的上下文回答医疗相关问题。

### 上下文数据：
{context}

### 用户问题：
{query}

### 医疗领域特定指南：
1. **患者识别**：
   - 仔细识别信息是关于哪位**患者**的。
   - 不要混淆不同患者的医疗记录。
2. **医疗事件时间线**：
   - 特别注意**入院日期、诊断日期、治疗日期**。
   - 医疗事件通常按顺序发生，确保时间顺序的准确性。
3. **医学术语**：
   - 寻找具体的诊断、症状、治疗方案、药物和手术。
   - 如果问题涉及某种疾病，请查找相关症状和诊断发现。
4. **医院/科室信息**：
   - 注意提到的是哪家医院、哪个科室或主治医生。
   - 不同科室可能对同一患者有不同的记录。

### 示例：

**示例1 - 患者识别：**
上下文："患者张三于2023年3月5日入院。主诉持续咳嗽和发热。诊断：肺炎。"
问题：张三的诊断是什么？
回答：肺炎。

**示例2 - 治疗信息：**
上下文："患者于4月10日接受阑尾切除术。术后恢复良好。4月15日出院。"
问题：患者何时出院？
回答：4月15日。

**示例3 - 信息不足：**
上下文："患者主诉头痛和头晕。医生开具了CT检查。"
问题：患者开了什么药？
回答：无法回答。

### 格式约束：
- **语言**：请使用**简体中文**回答。
- **部分回答**：如果只有**部分信息**可用，请基于现有数据尽量回答。
- **拒答**：**只有**当上下文中**完全没有任何相关信息**时，才回复："无法回答。"
- **简洁**：直接给出结论，不需要输出推理步骤。

### 回答：
"""
    
    elif domain == "LAW":
        return f"""你是一位法律信息专家。你的任务是**仅**基于提供的上下文回答法律相关问题。

### 上下文数据：
{context}

### 用户问题：
{query}

### 法律领域特定指南：
1. **当事人识别**：
   - 仔细识别**原告**、**被告**及其他相关当事人。
   - 不要混淆不同的当事人或案件。
2. **案件时间线**：
   - 特别注意**立案日期、开庭日期、判决日期**。
   - 法律程序按时序进行，确保时间顺序的准确性。
3. **法律术语**：
   - 寻找具体的指控、诉讼请求、判决结果、刑罚和法律条款。
   - 识别法院名称、案件编号和审判法官（如有提及）。
4. **判决/裁决关注**：
   - 区分不同的判决结果：有罪/无罪、承担责任/不承担责任、判赔金额等。
   - 注意任何上诉或后续裁定。

### 示例：

**示例1 - 当事人识别：**
上下文："在张三诉李四公司案中，原告指控被告违约。法院判决原告胜诉。"
问题：谁赢得了诉讼？
回答：原告张三胜诉。

**示例2 - 判决详情：**
上下文："被告被判欺诈罪成立。法院判处被告有期徒刑3年，并赔偿5万元。"
问题：判决结果是什么？
回答：有期徒刑3年，赔偿5万元。

**示例3 - 信息不足：**
上下文："案件于2022年1月15日立案。被告否认所有指控。"
问题：最终判决结果是什么？
回答：无法回答。

### 格式约束：
- **语言**：请使用**简体中文**回答。
- **部分回答**：如果只有**部分信息**可用，请基于现有数据尽量回答。
- **拒答**：**只有**当上下文中**完全没有任何相关信息**时，才回复："无法回答。"
- **简洁**：直接给出结论，不需要输出推理步骤。

### 回答：
"""
    
    else:  # GENERAL
        return f"""你是一个拥有强大逻辑推理能力的智能助手。你的任务是**仅**基于提供的上下文回答用户的问题。

### 上下文数据：
{context}

### 用户问题：
{query}

### 思考步骤：
1. **信息提取**：仔细阅读并从上下文中提取相关信息。
2. **实体核对**：确认问题中提到的实体（人物、组织、地点）。
3. **信息整合**：如果多个文档片段包含相关信息，请将它们连贯地整合起来。

### 示例：

**示例1 - 直接提取：**
上下文："埃菲尔铁塔位于法国巴黎。它于1889年建成。"
问题：埃菲尔铁塔何时建成？
回答：1889年。

**示例2 - 信息整合：**
上下文："项目于1月启动。3月完成第一阶段。6月整个项目完工。"
问题：项目从开始到结束用了多长时间？
回答：项目从1月到6月，历时6个月。

**示例3 - 信息不足：**
上下文："该公司成立于2010年，总部位于纽约。"
问题：公司的CEO是谁？
回答：无法回答。

### 格式约束：
- **语言**：请使用**简体中文**回答。
- **部分回答**：如果只有**部分信息**可用，请基于现有数据尽量回答。
- **拒答**：**只有**当上下文中**完全没有任何相关信息**时，才回复："无法回答。"
- **简洁**：直接给出结论，不需要输出思考过程。

### 回答：
"""

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
    context_snippet_preview = context_chunks[0]['page_content'][:500] if context_chunks else ""
    
    # Get domain classification
    domain = domain_router(query, context_snippet_preview, language)
    print(f"[Domain Router] Detected domain: {domain}")
    
    # Get domain-specific prompt based on detected domain
    if language == "en":
        prompt = _get_domain_prompt_en(query, context, domain)
    elif language == "zh":
        prompt = _get_domain_prompt_zh(query, context, domain)
    else:
        # Fallback to English GENERAL prompt
        prompt = _get_domain_prompt_en(query, context, domain)
    
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