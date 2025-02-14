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
    model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
    temperature=0.3
)

# 3. 設定 Prompt 模板
prompt_template = PromptTemplate(
    input_variables=["context", "input_text"],
    template=(
    " **角色設定**\n"
    "你是一個智慧型客服助理，專門為台灣學術網路危機處理中心 (TACERT) 提供服務。\n"
    "你的核心目標是準確、快速、直接回答使用者的問題，避免不必要的背景解釋或冗長回應。\n\n"

    " **回答原則**\n"
    "1. 直接回答：專注於解決問題，不提供多餘資訊。\n"
    "2. 簡明扼要：只用必要的字數表達完整內容。\n"
    "3. 條理清楚：使用條列式或分步驟說明，確保易讀。\n"
    "4. 避免反問：除非資訊不足，否則不主動反問。\n\n"

    " **回應格式**\n"
    "參考資料：\n{context}\n\n"
    "使用者問題：{input_text}\n"
    "AI 回應："
    )
)




# 4. 建立 LangChain 運行流程
chain = prompt_template | llm

# 5. FAISS 檢索函數
def search_faiss(query_text, top_k=3):
    """搜尋最相近的前 top_k 筆資料"""
    query_embedding = embed_model.encode(query_text).astype(np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # FAISS 需要 2D 陣列

    _, indices = my_faiss.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if str(idx) in faiss_metadata:
            results.append(faiss_metadata[str(idx)]['content'])

    return results

# 6. 逐步產生回應
def generate_response(input_text):
    """使用 chain.stream() 逐步傳輸字元"""
    retrieved_context = search_faiss(input_text)  # 先檢索 FAISS 取得 context
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
    app.run(host='0.0.0.0', port=3036)
