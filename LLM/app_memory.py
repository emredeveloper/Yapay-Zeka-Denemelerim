from flask import Flask, render_template, request, jsonify
import json
import os
from memory_llm import Qwen3VLAgent
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'memory-agent-secret'

# Global agent instance
agent = None
chat_history = []

def init_agent():
    """Agent'ı başlat"""
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
    """Ana sayfa"""
    return render_template('dashboard.html')


@app.route('/api/status')
def status():
    """Sistem durumu"""
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
    """Chat endpoint"""
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "Mesaj boş"}), 400
    
    try:
        # Agent'tan yanıt al
        result = agent.chat(user_message)
        
        # Geçmişe ekle
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
            "metadata": result['metadata'],
            "history_length": len(chat_history)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/view')
def view_memory():
    """Bellek durumunu görüntüle"""
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        # STM (Kısa Vadeli Bellek)
        stm_data = agent.stm.get_summary()
        
        # LTM (Uzun Vadeli Bellek) - örnek veriler
        ltm_count = agent.ltm.count()
        
        # Episodik log
        episodic_recent = agent.episodic.recent(10)
        
        return jsonify({
            "short_term": stm_data,
            "long_term": {
                "document_count": ltm_count,
                "embedder": "all-MiniLM-L6-v2"
            },
            "episodic": episodic_recent
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    """Belleği temizle"""
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        # Önce bellek özetini al
        stm_count = len(agent.stm.messages)
        ltm_count = agent.ltm.count()
        
        # STM'i temizle
        agent.stm.messages = []
        
        # Global geçmişi temizle
        global chat_history
        chat_history = []
        
        return jsonify({
            "success": True,
            "cleared": {
                "stm_messages": stm_count,
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
    """Analitikler"""
    if agent is None:
        return jsonify({"error": "Agent offline"}), 503
    
    try:
        total_chats = len(chat_history)
        
        # İstatistikler
        stats = {
            "total_messages": total_chats,
            "avg_response_length": 0,
            "error_count": 0,
            "success_count": 0
        }
        
        if total_chats > 0:
            response_lengths = [len(entry['assistant']) for entry in chat_history]
            stats['avg_response_length'] = sum(response_lengths) // len(response_lengths)
            stats['success_count'] = sum(1 for entry in chat_history if not entry['metadata'].get('error'))
            stats['error_count'] = total_chats - stats['success_count']
        
        return jsonify({
            "stats": stats,
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
    
    # Agent'ı başlat
    if init_agent():
        print("✅ Agent başarıyla yüklendi")
        print("🌐 Server: http://localhost:5000")
        app.run(debug=True, port=5000)
    else:
        print("❌ Agent başlatılamadı. OLLAMA_API_KEY kontrol edin.")
        app.run(debug=True, port=5000)
