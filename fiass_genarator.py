from sentence_transformers import SentenceTransformer, util
import fitz
import os
import faiss  
import numpy as np
import json




# 初始化嵌入模型
embed_model = SentenceTransformer("BAAI/bge-m3")






local_chunk_set = []


def read_pdf(pdf_path):
    """讀取 PDF 檔案並依據文本頁數返回其內容"""
    page_content= []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_content.append({'page':page_num+1,'content':text})
    doc.close()
    return page_content


def chunk_text(pages, chunk_size):

    """將文本拆分為指定大小的 chunks，允許 chunk 跨越頁數，並標記 chunk 涉及的頁數範圍"""
    chunks = []
    current_chunk = []
    current_pages = set()
    current_length = 0
    max_token_length = chunk_size  

    for page in pages:
        text = page['content']
        text_tokens = embed_model.tokenizer.tokenize(text)

        while text_tokens:
            space_left = max_token_length - current_length

            # 如果當前 chunk 還有空間
            if space_left > 0:
                tokens_to_add = text_tokens[:space_left]
                text_tokens = text_tokens[space_left:]

                current_chunk.extend(tokens_to_add)
                current_pages.add(page['page'])
                current_length += len(tokens_to_add)

            # 當 chunk 滿了，就存入 chunks，並重置變數
            if current_length >= max_token_length or (current_length > 0 and len(text_tokens) > 0):
                chunks.append({
                    'pages': sorted(current_pages),
                    'content': embed_model.tokenizer.convert_tokens_to_string(current_chunk)
                })
                current_chunk = []
                current_pages = set()
                current_length = 0

    # 處理最後一個未滿的 chunk
    if current_length > 0:
        chunks.append({
            'pages': sorted(current_pages),
            'content': embed_model.tokenizer.convert_tokens_to_string(current_chunk)
        })

    return chunks
data = []
folder_path = 'pdf'
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

result = []
faiss_metadata = {}
chunk_sizes = [128]  # 測試不同 chunk sizes
for size in chunk_sizes:
    EMBEDDING_DIM = 1024  
    index = faiss.IndexFlatL2(EMBEDDING_DIM)  # 建立 L2 距離索引
    files_to_delete = ['faiss_metadata.json', 'faiss_index.idx']
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        content = read_pdf(full_path)
        chunks = chunk_text(content, size)
        for idx, chunk in enumerate(chunks):
            text = chunk["content"]
            # 計算 embedding 並存入 cache
            embedding = embed_model.encode(text).astype(np.float32)
            # 儲存 metadata
            data.append({
                "File_Name": pdf_file,
                "content": text,
                "Page_Num": ','.join(str(p) for p in chunk["pages"])
            })
            index.add(np.array([embedding]))  # 加入 FAISS
            faiss_metadata[len(faiss_metadata)] = data[-1]  # FAISS ID 對應 metadata

with open('law_data.json', 'r') as f:
    law_data = json.load(f)

for each in law_data:
    text = each["law_text"]
    # 計算 embedding 並存入 cache
    embedding = embed_model.encode(text).astype(np.float32)
    # 儲存 metadata
    data.append({
        "File_Name": each["law_name"],
        "content": text
    })
    index.add(np.array([embedding]))  # 加入 FAISS
    faiss_metadata[len(faiss_metadata)] = data[-1]  # FAISS ID 對應 metadata


faiss.write_index(index, "faiss_index.idx")
# 儲存 metadata
with open('faiss_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(faiss_metadata, f, ensure_ascii=False)
