from sentence_transformers import SentenceTransformer
import fitz
import os
from chromadb import PersistentClient

# 初始化嵌入模型
embed_model = SentenceTransformer("BAAI/bge-m3")

# 初始化 ChromaDB 持久化客戶端
client = PersistentClient()

# 創建或獲取 collection
collection = client.get_or_create_collection("cert_pdf")
collection.delete(where={"id": {"$ne": ""}})
print("Old data deleted successfully.")
# 使用 set() 存儲已處理的 chunk，以便本地查重
local_chunk_set = set()


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

        chunk_sizes = [512]

        for chunk_size in chunk_sizes:
            chunks = chunk_text(content, chunk_size)

            for idx, chunk in enumerate(chunks):
                # 本地查詢是否已經出現過這個 chunk
                if chunk in local_chunk_set:
                    print(f"Skipping duplicate chunk: {pdf_file} (size: {chunk_size}, index: {idx})")
                    continue  # 跳過重複的 chunk

                local_chunk_set.add(chunk)  # 加入本地集合
                print(f"Processing unique chunk: {pdf_file} (size: {chunk_size}, index: {idx})")

                # 計算 chunk 的 embedding
                embedding = embed_model.encode(chunk).tolist()

                # 儲存到 ChromaDB
                collection.add(
                    documents=[chunk],  # chunk 內容
                    embeddings=[embedding],  # 存入 embedding
                    metadatas=[{"article": pdf_file, "content": chunk}],  # metadata
                    ids=[f"{pdf_file}_{chunk_size}_{idx}"]
                )


# 執行 PDF 處理並存入 ChromaDB
process_pdf_folder("./pdf")
