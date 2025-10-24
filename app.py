from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "mental_health_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = Flask(__name__)
CORS(app)

# Add this root route
@app.route('/')
def home():
    return jsonify({'message': 'Mental Health Chatbot API is running', 'status': 'ok'})

def get_bot_response(message):
    result = chatbot(message, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '')
    reply = get_bot_response(user_message)
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
