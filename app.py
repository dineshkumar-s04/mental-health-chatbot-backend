from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small"
HF_TOKEN = "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"  # Your Hugging Face API token

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface(payload):
    """Query Hugging Face Inference API"""
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mental Health Chatbot API is running!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Query Hugging Face API
        response = query_huggingface({
            "inputs": user_message,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        })
        
        # Extract response
        if isinstance(response, list) and len(response) > 0:
            bot_response = response[0].get('generated_text', 'I\'m here to listen.')
        else:
            bot_response = "I'm here to support you. Tell me more."
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
