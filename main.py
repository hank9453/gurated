from flask import Flask, request, Response, render_template
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from langchain_core.runnables import RunnableSequence

app = Flask(__name__)

# 1. 初始化嵌入模型與 FAISS
embed_model = SentenceTransformer("BAAI/bge-m3")
my_faiss = faiss.read_index("faiss_index.idx")

with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    faiss_metadata = json.load(f)

# 2. 設定 LLM
llm = OllamaLLM(
    model="SimonPu/llama-3-taiwan-8b-instruct-dpo",
    temperature=0.4
)

prompt_template = PromptTemplate(
    input_variables=["context", "input_text"],
    template=(
        "## 系統指示\n"
        "你是 **TACERT 客服代表**，負責解答使用者的技術支援與帳號管理問題。\n"
        "**請根據使用者的問題與提供的參考資料進行比對**，確保回應準確、清楚，且**不直接複製參考資料內容**。\n\n"

        "## 回應準則\n"
        "1. **比對使用者的問題與參考資料**，只提供與問題相關的回應，避免多餘資訊。\n"
        "2. **若使用者的問題與參考資料有明確對應**，請提供簡潔的解決方案，避免冗長回答。\n"
        "3. **若參考資料無法解答問題**，請主動告知使用者，並建議聯繫 TACERT 或相關單位獲得進一步支援。\n"
        "4. **若資訊不足**，請告知使用者可提供更多細節，以獲得更準確的解決方案。\n\n"

        "## 當資訊不足時\n"
        "> **由於您的問題需要更多細節才能正確處理，請直接聯繫相關單位獲得協助：**\n"
        "> - **TACERT 客服中心**：\n"
        ">   -  Email：service@cert.tanet.edu.tw\n"
        ">   -  Tel：+886-7-5250211\n\n"

        "##  使用者問題\n"
        "```plaintext\n{input_text}\n```\n"

        "##  參考資料（僅作為比對依據，請勿直接輸出）\n"
        "```plaintext\n{context}\n```\n"
        "**請確保回應使用 Markdown 格式 且滿足使用者問題：**\n"
    )
)




# 4. 建立 LangChain 運行流程
chain = prompt_template | llm

# 5. FAISS 檢索函數
def search_faiss(query_text, top_k=3):
    """搜尋最相近的前 top_k 筆資料"""
    query_embedding = embed_model.encode(query_text).astype(np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS 需要 2D 陣列
    
    distance, indices = my_faiss.search(query_embedding, top_k)
    results = []
    for i, (dist, idx) in enumerate(zip(distance[0], indices[0])):
        results.append(faiss_metadata[str(idx)]['content'])

    return results

# 6. 逐步產生回應
def generate_response(input_text):
    """使用 chain.stream() 逐步傳輸字元"""
    retrieved_context = search_faiss(input_text)  # 先檢索 FAISS 取得 context
    if len(retrieved_context) == 0:
        yield "很抱歉，找不到相關資訊，請直接聯絡 TACERT。"
        return
    combined_context = "\n".join(retrieved_context)  # 合併多個 context

    input_data = {"context": combined_context, "input_text": input_text}  # 確保變數對應
    for chunk in chain.stream(input_data):  
        yield chunk  # 逐步傳送字串

# 7. 定義 API 端點
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('input_text', '')

    return Response(generate_response(input_text), content_type="text/plain")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5278)
