from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from RAG import answer_question

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        
        if not user_question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get answer from RAG system
        answer = answer_question(user_question)
        
        return jsonify({
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
