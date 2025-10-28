from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Groq API configuration - use environment variable (Updated)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("✓ Groq client initialized successfully")
except Exception as e:
    print(f"✗ Error initializing Groq client: {str(e)}")
    traceback.print_exc()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mental Health Chatbot API is running with Groq!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        print(f"Received message: {user_message}")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Call Groq API with Llama model
        print("Calling Groq API...")
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
            model="llama3-70b-8192,
            temperature=0.7,
            max_tokens=150
        )
        
        bot_response = chat_completion.choices[0].message.content
        print(f"Bot response: {bot_response}")
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in /chat endpoint: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong: {error_msg}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
