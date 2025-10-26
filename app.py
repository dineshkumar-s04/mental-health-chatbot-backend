from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

# Groq API configuration
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mental Health Chatbot API is running with Groq!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Call Groq API with Llama model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a compassionate mental health support assistant. Provide empathetic, supportive responses to help people feel heard and understood. Keep responses brief and caring."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=150
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
