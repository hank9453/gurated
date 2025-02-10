from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import json

# 初始化 BGE-M3 模型
embed_model = SentenceTransformer("BAAI/bge-m3")

# 指定存儲位置
persist_directory = "/Users/hank/Desktop/project/chroma"
client = PersistentClient(path=persist_directory)

# 創建或獲取 collection
collection = client.get_or_create_collection("law_articles")

# **步驟 1：清空現有數據**
collection.delete(where={"id": {"$ne": ""}})
print("Old data deleted successfully.")

# **步驟 2：讀取新的 data.json**
json_file_path = "/Users/hank/Desktop/project/data.json"
with open(json_file_path, "r", encoding="utf-8") as file:
    data_list = json.load(file)

# **步驟 3：批量插入新數據**
ids = []
embeddings = []
metadatas = []

for data in data_list:
    if "id" in data and "content" in data:
        ids.append(data["id"])
        embeddings.append(embed_model.encode(data["content"]).tolist())  # 產生向量
        metadatas.append({
            "chapter": data.get("chapter", ""),
            "article": data.get("article", ""),
            "content": data["content"],
            "tags": ", ".join(data.get("tags", [])) 
        })

# **步驟 4：插入新的資料**
if ids:
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"Successfully inserted {len(ids)} articles into ChromaDB!")
else:
    print("No valid data found in data.json.")
