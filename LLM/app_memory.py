from flask import Flask, render_template, request, jsonify
import json
import os
import base64
from memory_llm import Qwen3VLAgent
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'memory-agent-secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

agent = None
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_agent():
    global agent
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return False
    try:
        agent = Qwen3VLAgent(api_key=api_key)
        return True
    except Exception as e:
        print(f"Agent başlatma hatası: {e}")
        return False


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/status')
def status():
    if agent is None:
        return jsonify({"status": "offline", "message": "Agent başlatılmamış"}), 503
    
    try:
        return jsonify({
            "status": "online",
            "model": agent.model_id,
            "memory": {
                "stm": agent.stm.get_summary(),
                "ltm": agent.ltm.get_summary(),
                "episodic_count": agent.episodic.count()
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    data = request.json
    user_message = data.get('message', '').strip()
    image_base64 = data.get('image', None)
    
    if not user_message:
        return jsonify({"error": "Mesaj boş"}), 400
    
    try:
        result = agent.chat(user_message, image_base64=image_base64)
        
        chat_entry = {
            "user": user_message,
            "assistant": result['response'],
            "logs": result['logs'],
            "memory_state": result['memory_state'],
            "metadata": result['metadata']
        }
        chat_history.append(chat_entry)
        
        return jsonify({
            "success": True,
            "response": result['response'],
            "logs": result['logs'],
            "memory_state": result['memory_state'],
            "prompt_used": result['prompt_used'],
            "metadata": result['metadata'],
            "history_length": len(chat_history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/image/encode', methods=['POST'])
def encode_image():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yok"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Geçersiz dosya"}), 400
    
    try:
        file_data = file.read()
        base64_data = base64.b64encode(file_data).decode()
        return jsonify({
            "success": True,
            "base64": base64_data,
            "filename": secure_filename(file.filename)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/view')
def view_memory():
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        stm_data = agent.stm.get_summary()
        ltm_count = agent.ltm.count()
        visuals = agent.ltm.get_all_visuals()
        episodic_recent = agent.episodic.recent(10)
        
        return jsonify({
            "short_term": stm_data,
            "long_term": {
                "document_count": ltm_count,
                "visual_count": agent.ltm.get_visual_count(),
                "embedder": "all-MiniLM-L6-v2"
            },
            "visuals": visuals,
            "episodic": episodic_recent
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        stm_count = len(agent.stm.messages)
        ltm_count = agent.ltm.count()
        
        agent.stm.messages = []
        global chat_history
        chat_history = []
        
        return jsonify({
            "success": True,
            "cleared": {
                "stm_messages": stm_count,
                "ltm_documents": ltm_count,
                "history_entries": len(chat_history)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history')
def history():
    """Sohbet geçmişi"""
    return jsonify({
        "total": len(chat_history),
        "history": chat_history[-50:]  # Son 50
    })


@app.route('/api/logs/latest')
def latest_logs():
    """En son loglar"""
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        if len(chat_history) == 0:
            return jsonify({"logs": []})
        
        latest_entry = chat_history[-1]
        return jsonify({"logs": latest_entry.get('logs', [])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics')
def analytics():
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        total_chats = len(chat_history)
        
        stats = {
            "total_messages": total_chats,
            "avg_response_length": 0,
            "error_count": 0,
            "success_count": 0,
            "messages_with_images": 0
        }
        
        if total_chats > 0:
            response_lengths = [len(entry['assistant']) for entry in chat_history]
            stats['avg_response_length'] = sum(response_lengths) // len(response_lengths)
            stats['success_count'] = sum(1 for entry in chat_history if not entry['metadata'].get('error'))
            stats['error_count'] = total_chats - stats['success_count']
            stats['messages_with_images'] = sum(1 for entry in chat_history if entry['metadata'].get('has_image'))
        
        episodic_stats = agent.episodic.get_stats()
        
        return jsonify({
            "chat_stats": stats,
            "episodic_stats": episodic_stats,
            "memory": {
                "stm": agent.stm.get_summary(),
                "ltm": agent.ltm.get_summary(),
                "episodic": agent.episodic.count()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Flask Server başlatılıyor...")
    
    if init_agent():
        print("✅ Agent başarıyla yüklendi")
        print("🌐 Server: http://localhost:5000")
        app.run(debug=True, port=5000)
    else:
        print("❌ Agent başlatılamadı. OLLAMA_API_KEY kontrol edin.")
        app.run(debug=True, port=5000)
