"""
Gelişmiş Multi-Agent Quiz Sistemi - Flask Uygulaması
"""
from flask import Flask, render_template, request, jsonify, session
from quiz_agents import (
    QuizGeneratorAgent, LearningAnalystAgent, AdaptiveAgent, 
    TutorAgent, GamificationAgent
)
from turkce_quiz_uretici import makale_cek
import importlib.util
import sys
from pathlib import Path
from datetime import datetime
import json

# Agent'ları başlat
quiz_generator = QuizGeneratorAgent()
learning_analyst = LearningAnalystAgent()
adaptive_agent = AdaptiveAgent()
tutor_agent = TutorAgent()
gamification = GamificationAgent()

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Production'da değiştirin

# Basit session storage (production'da database kullanılmalı)
user_sessions = {}

@app.route('/')
def index():
    """Ana sayfa"""
    if 'user_id' not in session:
        session['user_id'] = f"user_{datetime.now().timestamp()}"
        user_sessions[session['user_id']] = {
            'total_quizzes': 0,
            'total_score': 0,
            'perfect_scores': 0,
            'hard_quizzes': 0,
            'streak': 0,
            'last_quiz_date': None,
            'achievements': [],
            'learning_history': []
        }
    return render_template('quiz_advanced.html')

@app.route('/api/generate', methods=['POST'])
def generate_quiz():
    """Quiz üretme API - Agent 1 kullanır"""
    try:
        data = request.get_json()
        konu = data.get('konu', '').strip()
        url = data.get('url', '').strip()
        soru_sayisi = int(data.get('soru_sayisi', 5))
        zorluk = data.get('zorluk', 'orta')
        
        user_id = session.get('user_id')
        user_stats = user_sessions.get(user_id, {})
        
        # Eksik konuları al (varsa)
        eksik_konular = data.get('eksik_konular', [])
        if user_stats.get('learning_history'):
            last_analysis = user_stats['learning_history'][-1]
            eksik_konular = last_analysis.get('eksik_konular', [])
        
        if not konu and not url:
            return jsonify({
                'success': False,
                'error': 'Lütfen bir konu veya URL girin.'
            }), 400
        
        # URL'den içerik çek
        icerik_metni = None
        if url:
            icerik_metni = makale_cek(url)
            if not icerik_metni:
                return jsonify({
                    'success': False,
                    'error': 'Makale içeriği çekilemedi.'
                }), 400
            if not konu:
                konu = "Makale İçeriği"
        
        # Agent 1: Quiz üret
        quiz_data = quiz_generator.generate_quiz(
            konu, soru_sayisi, zorluk, icerik_metni, eksik_konular
        )
        
        # Quiz ID oluştur
        quiz_id = f"quiz_{datetime.now().timestamp()}"
        session['current_quiz_id'] = quiz_id
        session['current_quiz_data'] = quiz_data
        
        return jsonify({
            'success': True,
            'quiz_id': quiz_id,
            **quiz_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Hata: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_performance():
    """Performans analizi - Agent 2 kullanır"""
    try:
        data = request.get_json()
        user_answers = data.get('answers', {})
        
        quiz_data = session.get('current_quiz_data')
        if not quiz_data:
            return jsonify({
                'success': False,
                'error': 'Quiz bulunamadı.'
            }), 400
        
        # Agent 2: Performans analizi
        analysis = learning_analyst.analyze_performance(quiz_data, user_answers)
        
        # Agent 3: Adaptif öneriler
        user_id = session.get('user_id')
        current_difficulty = data.get('difficulty', 'orta')
        new_difficulty = adaptive_agent.adapt_difficulty(
            user_id, analysis, current_difficulty
        )
        
        next_quiz = adaptive_agent.suggest_next_quiz(
            analysis.get('eksik_konular', [])
        )
        
        # Agent 5: Gamification
        user_stats = user_sessions.get(user_id, {})
        user_stats['total_quizzes'] = user_stats.get('total_quizzes', 0) + 1
        user_stats['total_score'] = user_stats.get('total_score', 0) + analysis.get('genel_puan', 0)
        
        if analysis.get('yuzde', 0) == 100:
            user_stats['perfect_scores'] = user_stats.get('perfect_scores', 0) + 1
        
        # Streak kontrolü
        today = datetime.now().date()
        last_date = user_stats.get('last_quiz_date')
        if last_date == today:
            pass  # Aynı gün
        elif last_date and (today - last_date).days == 1:
            user_stats['streak'] = user_stats.get('streak', 0) + 1
        else:
            user_stats['streak'] = 1
        user_stats['last_quiz_date'] = today
        
        # Achievement kontrolü
        achievements = gamification.check_achievements(user_stats)
        new_achievements = [a for a in achievements if a not in user_stats.get('achievements', [])]
        user_stats['achievements'].extend(new_achievements)
        
        # Level hesapla
        level_info = gamification.calculate_level(
            user_stats['total_quizzes'],
            user_stats['total_score']
        )
        
        # Learning history'ye ekle
        user_stats['learning_history'].append(analysis)
        user_sessions[user_id] = user_stats
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'adaptive': {
                'new_difficulty': new_difficulty,
                'next_quiz': next_quiz
            },
            'gamification': {
                'level': level_info,
                'achievements': achievements,
                'new_achievements': new_achievements,
                'stats': {
                    'total_quizzes': user_stats['total_quizzes'],
                    'total_score': user_stats['total_score'],
                    'streak': user_stats['streak']
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Hata: {str(e)}'
        }), 500

@app.route('/api/tutor/hint', methods=['POST'])
def get_hint():
    """Tutor agent'tan ipucu al"""
    try:
        data = request.get_json()
        question_index = data.get('question_index')
        attempt = data.get('attempt', 1)
        
        quiz_data = session.get('current_quiz_data')
        if not quiz_data:
            return jsonify({
                'success': False,
                'error': 'Quiz bulunamadı.'
            }), 400
        
        soru = quiz_data['sorular'][question_index]
        
        # Agent 4: Tutor ipucu
        hint = tutor_agent.get_hint(soru, attempt)
        
        return jsonify({
            'success': True,
            'hint': hint
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Hata: {str(e)}'
        }), 500

@app.route('/api/tutor/explain', methods=['POST'])
def explain_concept():
    """Tutor agent'tan konu açıklaması al"""
    try:
        data = request.get_json()
        konu = data.get('konu', '')
        
        if not konu:
            return jsonify({
                'success': False,
                'error': 'Konu belirtilmedi.'
            }), 400
        
        # Agent 4: Konu açıklaması
        explanation = tutor_agent.explain_concept(konu)
        
        return jsonify({
            'success': True,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Hata: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Kullanıcı istatistiklerini getir"""
    user_id = session.get('user_id')
    user_stats = user_sessions.get(user_id, {})
    
    level_info = gamification.calculate_level(
        user_stats.get('total_quizzes', 0),
        user_stats.get('total_score', 0)
    )
    
    return jsonify({
        'success': True,
        'stats': {
            **user_stats,
            'level': level_info
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)






