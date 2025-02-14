from sentence_transformers import SentenceTransformer
import fitz
import os
from openai import OpenAI
import json
import re


# 初始化嵌入模型
embed_model = SentenceTransformer("BAAI/bge-m3")
api_key = "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"
client = OpenAI(api_key=api_key)  
documents = []
def read_pdf(pdf_path):
    """讀取 PDF 檔案並返回其文本內容"""
    doc = fitz.open(pdf_path)
    text_content = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(text_content)


def chunk_text(text, chunk_size):
    """將文本拆分為指定大小的 chunks，確保 token 長度符合要求"""
    tokens = embed_model.tokenizer.tokenize(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = embed_model.tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def process_pdf_folder(folder_path="./pdf"):
    """處理 PDF，並將非重複 chunk 存入 ChromaDB"""
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        content = read_pdf(full_path)

        chunk_size = 512
        chunks = chunk_text(content, chunk_size)
        for idx, chunk in enumerate(chunks):

            print(f"Processing unique chunk: {pdf_file} (size: {chunk_size}, index: {idx})")
            
            prompt = f"""
            請為以下文本生成 JSON 格式的元數據：
            1. **最多 6 個代表性關鍵字**（返回陣列，若無適當關鍵字則回傳 `[]`）。
            2. **摘要段落**（使用**客觀描述**，不得包含「本文」「這篇文章」「文章探討」「文章內」等主觀詞彙，若無合適摘要則回傳 `""`）。
            3. **前綴（Naive Representation）**，即該新聞的前兩句話（若無適當內容則回傳 `""`）。

            **文本**：
            "{chunk}"

            **摘要要求**：
            - 直接概括文本的 **核心資訊**，不可包含「本文說明」「文章探討」「文本描述」「這篇文章描述」等語句。
            - 內容應當以 **客觀事實** 開頭。
            - 若文本無法生成適當摘要，請返回 `""`。
            **輸出格式：** 你的回應**必須**是 JSON 格式，**不允許包含任何額外的文字**，且內容必須符合以下格式：
            請以 JSON 格式返回結果：
            {{
                "keywords": [...],  // 代表性關鍵字
                "summary": "...",  // 該文章的摘要
                "naive_representation": "..."  // 文章的前兩完整句子
            }}
            
            """
            completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一個可以為文章生成分類、摘要與關鍵字的助手，並提供合理的分類依據與上下文。"},
                {"role": "user", "content": prompt}
            ])

            data = completion.choices[0].message.content.strip()

            # Use regex to extract valid JSON content
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, data)
            if json_match:
                data = json_match.group()
            
            temp = json.loads(data)
            temp['file_name'] = pdf_file
            temp['content'] = chunk
            documents.append(temp)




process_pdf_folder("./pdf")

# Save documents to JSON file
with open('pdf_512_chunk.json', 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)
