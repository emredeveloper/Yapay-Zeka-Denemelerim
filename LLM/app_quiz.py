"""
Flask Web Uygulaması - Türkçe Quiz Üretici
"""
from flask import Flask, render_template, request, jsonify
import importlib.util
import sys
from pathlib import Path

# Dosya adında tire olduğu için özel import
spec = importlib.util.spec_from_file_location(
    "turkce_quiz_uretici", 
    Path(__file__).parent / "turkce-quiz-uretici.py"
)
turkce_quiz = importlib.util.module_from_spec(spec)
sys.modules["turkce_quiz_uretici"] = turkce_quiz
spec.loader.exec_module(turkce_quiz)

makale_cek = turkce_quiz.makale_cek
quiz_uret = turkce_quiz.quiz_uret

app = Flask(__name__)

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('quiz_index.html')

@app.route('/api/generate', methods=['POST'])
def generate_quiz():
    """Quiz üretme API endpoint"""
    try:
        data = request.get_json()
        konu = data.get('konu', '').strip()
        url = data.get('url', '').strip()
        soru_sayisi = int(data.get('soru_sayisi', 5))
        zorluk = data.get('zorluk', 'orta')
        
        # Validasyon
        if not konu and not url:
            return jsonify({
                'success': False,
                'error': 'Lütfen bir konu veya URL girin.'
            }), 400
        
        if soru_sayisi < 5 or soru_sayisi > 10:
            return jsonify({
                'success': False,
                'error': 'Soru sayısı 5-10 arası olmalıdır.'
            }), 400
        
        # URL'den içerik çek
        icerik_metni = None
        if url:
            icerik_metni = makale_cek(url)
            if not icerik_metni:
                return jsonify({
                    'success': False,
                    'error': 'Makale içeriği çekilemedi. Lütfen geçerli bir URL girin.'
                }), 400
            if not konu:
                konu = "Makale İçeriği"
        
        # Quiz üret
        quiz = quiz_uret(konu, soru_sayisi=soru_sayisi, zorluk=zorluk, icerik_metni=icerik_metni)
        
        # Sonuçları dict'e dönüştür
        result = {
            'success': True,
            'konu': quiz.konu,
            'baslik': quiz.baslik,
            'aciklama': quiz.aciklama,
            'toplam_puan': quiz.toplam_puan,
            'sorular': [
                {
                    'soru': s.soru,
                    'zorluk': s.zorluk,
                    'tip': s.tip,
                    'secenekler': [
                        {
                            'metin': sec.metin,
                            'dogru': sec.dogru
                        }
                        for sec in s.secenekler
                    ],
                    'dogru_cevap': s.dogru_cevap,
                    'aciklama': s.aciklama
                }
                for s in quiz.sorular
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Hata oluştu: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

