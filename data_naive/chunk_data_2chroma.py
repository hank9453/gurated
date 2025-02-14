from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import json


# 初始化嵌入模型
embed_model = SentenceTransformer("BAAI/bge-m3")

# 連接 ChromaDB
client = PersistentClient()
collection = client.get_or_create_collection("pdf_512_chunk")
# 讀取 JSON 檔案
with open('pdf_512_chunk.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍歷每個文件
for item in data:
    # 提取文本數據和元數據
    summary = item['summary']
    naive_representation = item['naive_representation']
    keywords = item['keywords']
    file_name = item['file_name']
    content = item['content']

    # 創建要嵌入的文本：可以將 summary 和 naive_representation 結合起來
    text_to_embed = f"Summary: {summary}\nNaive Representation: {naive_representation}"

    # 計算嵌入向量
    embedding = embed_model.encode(text_to_embed).tolist()

    # 添加到 ChromaDB 集合
    collection.add(
        embeddings=[embedding],  # 嵌入向量
        metadatas={
            'keywords': ",".join(keywords),  # 關鍵字列表
            'file_name': file_name,  # 文件名
            'content': content, #文件內容
            'summary': summary,
            'naive_representation': naive_representation
        },
        ids=[file_name + "_" + str(hash(content)) ]  # 文件名 + 內容hash 作為ID
    )

print("Data added to ChromaDB.")