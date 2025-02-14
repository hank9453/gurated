import numpy as np
import jieba.analyse
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("BAAI/bge-m3")
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def search_documents(query_text):
    client2 = PersistentClient()
    pdf_data = client2.get_collection("pdf_512_chunk")
    # ---------------------------
    # 2. 進行語意查詢
    # 將查詢文字直接轉換為語意嵌入向量
    semantic_query_embedding = embed_model.encode(query_text).tolist()

    # 在 collection 中以語意向量進行查詢，取得相似度最高的文件（此處 n_results=1）
    semantic_results = pdf_data.query(
        query_embeddings=[semantic_query_embedding],
        n_results=1
    )

    # 假設回傳結果中包含 "ids"，取得語意查詢最佳文件 ID
    se_ids = semantic_results['ids'][0][0]
    se_data = semantic_results["metadatas"][0][0]['content']

    

    # ---------------------------
    # 3. 進行關鍵字查詢
    # (a) 先從查詢文字中抽取關鍵字（使用 jieba）
    extracted_keywords = jieba.analyse.extract_tags(query_text, topK=3)

    # (b) 計算每個抽取到關鍵字的嵌入向量
    query_keyword_embeddings = [embed_model.encode(kw).tolist() for kw in extracted_keywords]

    # (c) 取得所有文件以便比對關鍵字（注意：若文件數量很多，請用批次或其他索引方式）
    all_docs = pdf_data.get()  # 假設回傳 dict 包含 "ids" 與 "metadatas"
    all_ids = all_docs["ids"]
    all_metadatas = all_docs["metadatas"]
    query_keyword_embeddings = np.array(query_keyword_embeddings)  # shape: (K, d)
    query_norms = np.linalg.norm(query_keyword_embeddings, axis=1, keepdims=True)
    query_normalized = query_keyword_embeddings / (query_norms + 1e-8)

    # 變數用來紀錄關鍵字查詢中最高的 cosine similarity 與對應文件 ID
    best_keyword_score = -1
    best_keyword_id = None
    best_keyword_doc = None
    for doc_id, metadata in zip(all_ids, all_metadatas):
        doc_keywords = metadata.get("keywords", [])
        if not doc_keywords:
            continue

        # 批次計算該文件中所有關鍵字的嵌入向量
        doc_keyword_embeddings = embed_model.encode(doc_keywords)
        doc_keyword_embeddings = np.array(doc_keyword_embeddings)

        # 若只有一筆關鍵字，則轉換成 2 維陣列
        if doc_keyword_embeddings.ndim == 1:
            doc_keyword_embeddings = doc_keyword_embeddings[np.newaxis, :]
        # 正規化文件關鍵字向量
        doc_norms = np.linalg.norm(doc_keyword_embeddings, axis=1, keepdims=True)
        doc_normalized = doc_keyword_embeddings / (doc_norms + 1e-8)
        # 計算該文件中所有關鍵字與查詢關鍵字之間的 cosine similarity
        # dot 會計算內積，結果矩陣 shape 為 (N, K)
        sims = np.dot(doc_normalized, query_normalized.T)
        # 取得這個文件中最大相似度
        max_sim_for_doc = np.max(sims)

        # 若此文件的最高關鍵字相似度超過目前記錄，更新最佳文件
        if max_sim_for_doc > best_keyword_score:
            best_keyword_score = max_sim_for_doc
            best_keyword_doc = metadata['content']
            print(best_keyword_doc)
            best_keyword_id = doc_id

    # ---------------------------
    # 4. 合併結果
    # 如果語意查詢與關鍵字查詢得到的文件不同，則合併兩筆結果

    final_results = set()
    final_results.add(se_data)
    if(best_keyword_id != se_ids) :
        final_results.add(best_keyword_doc)

    return final_results