from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, render_template
from langchain.llms import Ollama

app = Flask(__name__)

# Define the LLM
llm = Ollama(
    model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
    max_length=128,
    temperature=0.7
)

# Define the prompt template
template = PromptTemplate(
    input_variables=["input_text"],
    template="You are a helpful assistant. Answer the following question: {input_text}"
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=template)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('input_text', '')
    response = chain.run(input_text)
    return jsonify({"response": response})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3036)
