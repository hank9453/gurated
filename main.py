from flask import Flask, request, Response, render_template ,send_file, abort
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain_core.runnables import RunnableSequence
from flask import send_file
import os

app = Flask(__name__)

embed_model = SentenceTransformer("BAAI/bge-m3")
my_faiss = faiss.read_index("faiss_index.idx")

with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    faiss_metadata = json.load(f)

llm = OllamaLLM(
    model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
    temperature=0.3,top_p=0.6
)

prompt_template = PromptTemplate(
    input_variables=["context", "input_text"],
    template = (
    "## 角色設定\n"
    "你是 **TACERT 客服代表**，負責解答使用者的技術支援與帳號管理問題，確保回應準確且清楚。\n\n"

    "## 回應規則\n"
    "1. **直接回答問題，不要重複使用者的問題。**\n"
    "2. **不得提及「參考資料」，僅從中擷取必要資訊進行回答。**\n"
    "3. **比對使用者問題與內部資料，提供準確且簡潔的回應，避免冗長或無關資訊。**\n"
    "4. **不得直接複製內部資料**，請使用自己的話表達，使內容清楚易懂。\n"
    "5. **若無法提供完整解答，請引導使用者聯繫 TACERT 或相關單位獲得進一步支援。**\n"
    "6. **若資訊不足，請請求使用者提供更多細節，以便獲得更準確的解決方案。**\n"
    "7. **適用範圍**：不限於資安事件，包括但不限於帳號管理、系統操作、故障排除等技術支援問題。\n\n"

    "## 使用者問題\n"
    "```plaintext\n{input_text}\n```\n\n"

    "## 內部資料（僅作為回答依據，請勿提及或直接輸出）\n"
    "```plaintext\n{context}\n```\n\n"

    "## 回應格式\n"
    "**請直接回答使用者問題，確保回應清晰簡潔，並使用 Markdown 格式。不得重複使用者問題或提及「參考資料」。**\n"
)

)





chain = prompt_template | llm


def search_faiss(query_text, top_k=3):
    """搜尋最相近的前 top_k 筆資料"""
    query_embedding = embed_model.encode(query_text).astype(np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS 需要 2D 陣列
    
    distance, indices = my_faiss.search(query_embedding, top_k)
    # 計算前 k 個匹配結果的平均距離
    avg_dist = np.mean(distance[0])

    # 設定閾值（如果平均距離較低，則降低閾值）
    threshold = max(0.9, avg_dist )  # 例如，動態調整閾值
    results = []
    for i, (dist, idx) in enumerate(zip(distance[0], indices[0])):
        if dist <threshold:
            results.append({
                'content': faiss_metadata[str(idx)]['content'],
                'file_name': faiss_metadata[str(idx)]['File_Name']
            })
    return results


def generate_response(input_text):
    """使用 chain.stream() 逐步傳輸字元"""
    retrieved_context = search_faiss(input_text,top_k=10)  # 先檢索 FAISS 取得 context
    if len(retrieved_context) == 0:
        yield "很抱歉，找不到相關資訊，請直接聯絡 TACERT。"
        return

    combined_context = "\n".join([item['content'] for item in retrieved_context])  # 只合併 content 欄位
    input_data = {"context": combined_context, "input_text": input_text}

    # 逐步傳輸回應
    for chunk in chain.stream(input_data):  
        yield chunk  

    response = [{"file_name": item['file_name'], "preview": embed_model.tokenizer.decode(embed_model.tokenizer.encode(item['content'])[:40])} for item in retrieved_context]
    json_response = json.dumps(response, ensure_ascii=False)  # 確保是有效 JSON
    yield f'\n[END] {json_response}'  # 確保前端可以解析



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('input_text', '')

    return Response(generate_response(input_text), content_type="text/plain")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/preview_pdf', methods=['POST'])
def preview_pdf():
    # 從 URL 的 query string 中獲取檔案名稱
    data = request.json
    pdf_name = data.get('file_name')
    # 組合檔案路徑
    pdf_path = os.path.join(os.path.dirname(__file__), 'pdf', pdf_name)
    if not os.path.exists(pdf_path):
        return '檔案不存在', 404

    # 使用 send_file 並指定 mimetype 為 application/pdf，並以 inline 模式呈現（非下載）
    return send_file(pdf_path, mimetype='application/pdf', as_attachment=False)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5278)
