from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer with proper configuration
print("Loading model...")
model_path = "./mental_health_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True  # Optimize memory usage
)
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({"status": "Mental Health Chatbot API is running!", "version": "1.0"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Use get_json(silent=True) to avoid type warnings when body is empty/invalid
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Tokenize with proper settings
        inputs = tokenizer.encode(
            user_message + tokenizer.eos_token, 
            return_tensors='pt',
            truncation=True,  # FIX: Explicitly enable truncation
            max_length=100    # FIX: Limit input length
        )
        
        # Generate response with optimized settings
        with torch.no_grad():  # Disable gradient calculation
            outputs = model.generate(
                inputs,
                max_new_tokens=50,      # FIX: Reduced from 256
                min_length=10,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the user message from response
        if response.startswith(user_message):
            response = response[len(user_message):].strip()
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
