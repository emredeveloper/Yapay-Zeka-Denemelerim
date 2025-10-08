"""
Flask Web Arayüzü - YouTube Video Analiz Uygulaması
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, session, Response
from werkzeug.utils import secure_filename
import os
import json
import threading
from pathlib import Path
from datetime import datetime
import uuid

# YouTube analyzer'ı import et
from youtube_app import YouTubeVideoAnalyzer
# Ollama client'ı import et
from ollama_client import OllamaClient

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global analiz durumu takibi
analysis_status = {}
analysis_lock = threading.Lock()

# Ollama client
ollama = OllamaClient()


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Video analiz başlat (async)"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL boş olamaz!'}), 400
        
        # Ayarlar
        extract_interval = data.get('extract_interval', True)
        interval_seconds = int(data.get('interval_seconds', 30))
        
        # Unique analiz ID oluştur
        analysis_id = str(uuid.uuid4())
        
        # Durum başlat
        with analysis_lock:
            analysis_status[analysis_id] = {
                'status': 'starting',
                'progress': 0,
                'message': 'Analiz başlatılıyor...',
                'result': None,
                'error': None
            }
        
        # Analizi ayrı thread'de başlat
        thread = threading.Thread(
            target=run_analysis,
            args=(analysis_id, url, extract_interval, interval_seconds)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'message': 'Analiz başlatıldı'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_analysis(analysis_id, url, extract_interval, interval_seconds):
    """Analizi çalıştır (background thread)"""
    try:
        # Durum güncelle: Başlıyor
        update_status(analysis_id, 'running', 10, 'Video bilgileri alınıyor...')
        
        # Analyzer oluştur
        analyzer = YouTubeVideoAnalyzer(url, output_base_dir="static/results")
        
        # Video bilgilerini al
        update_status(analysis_id, 'running', 20, 'Video bilgileri alındı')
        video_info = analyzer.get_video_info()
        
        # Video indir
        update_status(analysis_id, 'running', 30, 'Video indiriliyor...')
        video_downloaded = analyzer.download_video()
        
        # Transkript al
        update_status(analysis_id, 'running', 50, 'Transkript alınıyor...')
        full_text, transcript_with_time, sentences = analyzer.get_transcript()
        
        if not full_text and not video_downloaded:
            update_status(analysis_id, 'error', 0, 'Ne transkript ne de video alınamadı!', error='Video erişilemez')
            return
        
        # Frame'leri çıkar
        sentence_frames = []
        interval_frames = []
        
        if analyzer.video_path and os.path.exists(analyzer.video_path):
            if full_text:
                update_status(analysis_id, 'running', 60, 'Cümle frame\'leri çıkarılıyor...')
                sentence_frames = analyzer.extract_sentence_frames(transcript_with_time, sentences)
            
            if extract_interval:
                update_status(analysis_id, 'running', 75, 'Düzenli frame\'ler çıkarılıyor...')
                interval_frames = analyzer.extract_frames(interval_seconds)
        
        # Metin verilerini kaydet
        update_status(analysis_id, 'running', 85, 'Veriler kaydediliyor...')
        if full_text:
            analyzer.save_text_data(video_info, full_text, transcript_with_time, sentences, sentence_frames)
        
        # Rapor oluştur
        update_status(analysis_id, 'running', 95, 'Rapor oluşturuluyor...')
        analyzer.generate_summary_report(video_info, full_text if full_text else "", sentence_frames, interval_frames)
        
        # Sonuç hazırla
        result = {
            'video_id': analyzer.video_id,
            'project_dir': str(analyzer.project_dir.relative_to('static')),
            'images_dir': str(analyzer.images_dir.relative_to('static')),
            'text_dir': str(analyzer.text_dir.relative_to('static')),
            'video_info': video_info,
            'has_transcript': full_text is not None,
            'has_video': video_downloaded,
            'sentence_frames_count': len(sentence_frames),
            'interval_frames_count': len(interval_frames),
            'transcript_length': len(full_text) if full_text else 0,
            'sentence_count': len(sentences) if sentences else 0,
            'sentence_frames': sentence_frames[:50],  # İlk 50 frame
            'interval_frames': interval_frames[:50],
            'transcript_data': transcript_with_time if transcript_with_time else [],
            'full_text': full_text if full_text else ''  # Tam transkript metni
        }
        
        # Tamamlandı
        update_status(analysis_id, 'completed', 100, 'Analiz tamamlandı!', result=result)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        update_status(analysis_id, 'error', 0, f'Hata: {str(e)}', error=error_msg)


def update_status(analysis_id, status, progress, message, result=None, error=None):
    """Analiz durumunu güncelle"""
    with analysis_lock:
        if analysis_id in analysis_status:
            analysis_status[analysis_id].update({
                'status': status,
                'progress': progress,
                'message': message,
                'result': result,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })


@app.route('/status/<analysis_id>')
def get_status(analysis_id):
    """Analiz durumunu sorgula"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        return jsonify(analysis_status[analysis_id])


@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    """Analiz sonuçlarını göster"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analiz bulunamadı", 404
        
        status_data = analysis_status[analysis_id]
        
        if status_data['status'] != 'completed':
            return "Analiz henüz tamamlanmadı", 400
        
        result = status_data['result']
        
    return render_template('results.html', 
                         analysis_id=analysis_id,
                         result=result)


@app.route('/gallery/<analysis_id>')
def gallery(analysis_id):
    """Frame galerisi"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analiz bulunamadı", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analiz henüz tamamlanmadı", 400
        
        result = status_data['result']
    
    # Tüm frame'leri al
    images_dir = Path('static') / result['images_dir']
    
    all_frames = []
    if images_dir.exists():
        for img_file in sorted(images_dir.glob('*.jpg')):
            frame_info = {
                'filename': img_file.name,
                'path': str(Path(result['images_dir']) / img_file.name),
                'type': 'sentence' if 'sentence' in img_file.name else 'interval'
            }
            
            # Zaman bilgisini çıkar
            import re
            time_match = re.search(r'time_(\d+)m(\d+)s', img_file.name)
            if time_match:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))
                frame_info['time'] = f"{minutes:02d}:{seconds:02d}"
                frame_info['timestamp'] = minutes * 60 + seconds
            
            all_frames.append(frame_info)
    
    return render_template('gallery.html',
                         analysis_id=analysis_id,
                         frames=all_frames,
                         result=result)


@app.route('/transcript/<analysis_id>')
def transcript(analysis_id):
    """Transkript görüntüle ve ara"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analiz bulunamadı", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analiz henüz tamamlanmadı", 400
        
        result = status_data['result']
    
    # Arama sorgusu
    query = request.args.get('q', '').strip()
    
    transcript_data = result.get('transcript_data', [])
    
    # Arama yap
    if query:
        filtered_transcript = []
        for entry in transcript_data:
            if query.lower() in entry['text'].lower():
                filtered_transcript.append(entry)
    else:
        filtered_transcript = transcript_data
    
    return render_template('transcript.html',
                         analysis_id=analysis_id,
                         transcript=filtered_transcript,
                         query=query,
                         total_count=len(transcript_data),
                         filtered_count=len(filtered_transcript),
                         result=result)


@app.route('/api/search/<analysis_id>')
def api_search(analysis_id):
    """Transkript arama API"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analiz henüz tamamlanmadı'}), 400
        
        result = status_data['result']
    
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Arama sorgusu boş'}), 400
    
    transcript_data = result.get('transcript_data', [])
    
    matches = []
    for i, entry in enumerate(transcript_data):
        if query.lower() in entry['text'].lower():
            matches.append({
                'index': i,
                'text': entry['text'],
                'start': entry['start'],
                'duration': entry['duration'],
                'time_formatted': f"{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}"
            })
    
    return jsonify({
        'query': query,
        'total': len(transcript_data),
        'matches': len(matches),
        'results': matches
    })


@app.route('/download/<path:filepath>')
def download_file(filepath):
    """Dosya indir"""
    try:
        # filepath zaten results/xxxx/file.txt formatında geliyor
        full_path = os.path.join('static', filepath)
        
        if not os.path.exists(full_path):
            return f"Dosya bulunamadı: {filepath}", 404
        
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return f"İndirme hatası: {str(e)}", 500


# ============================================================
# OLLAMA & AI ÖZELLİKLERİ
# ============================================================

@app.route('/qa/<analysis_id>')
def qa_page(analysis_id):
    """Video Q&A sayfası"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analiz bulunamadı", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analiz henüz tamamlanmadı", 400
        
        result = status_data['result']
    
    # Ollama durumunu kontrol et
    ollama_available = ollama.check_connection()
    
    return render_template('qa.html',
                         analysis_id=analysis_id,
                         result=result,
                         ollama_available=ollama_available)


@app.route('/api/ask/<analysis_id>', methods=['POST'])
def ask_question(analysis_id):
    """Video hakkında soru sor"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analiz henüz tamamlanmadı'}), 400
        
        result = status_data['result']
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Soru boş olamaz'}), 400
    
    # Ollama kontrolü
    if not ollama.check_connection():
        return jsonify({'error': 'Ollama bağlantısı yok. Ollama çalışıyor mu?'}), 503
    
    try:
        # Transcript'i hazırla
        transcript_text = result.get('full_text', '')
        
        # Video bilgilerini hazırla
        video_info_data = result.get('video_info', {})
        video_info = {
            'title': video_info_data.get('title', 'N/A'),
            'channel': video_info_data.get('channel', 'N/A'),
            'duration': video_info_data.get('duration_formatted', 'N/A'),
            'views': video_info_data.get('views', 'N/A')
        }
        
        # İlgili frame'leri bul (opsiyonel - performans için devre dışı)
        relevant_frames = []
        use_frames = data.get('use_frames', False)  # Frontend'den gelen parametre
        
        if use_frames and 'sentence_frames' in result:
            # Sadece ilk 2 frame'i al (hız için)
            for frame_info in result['sentence_frames'][:2]:
                # frame_info bir dict, filename key'i var
                if isinstance(frame_info, dict):
                    frame_filename = frame_info.get('filename', '')
                else:
                    # Eğer string ise direk kullan
                    frame_filename = frame_info
                
                if frame_filename:
                    frame_path = os.path.join('static', result['images_dir'], frame_filename)
                    if os.path.exists(frame_path):
                        relevant_frames.append(frame_path)
        
        # Soruyu yanıtla
        answer = ollama.answer_question_with_context(
            question=question,
            transcript=transcript_text,
            video_info=video_info,
            relevant_frames=relevant_frames if use_frames else None
        )
        
        return jsonify({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'}), 500


@app.route('/api/analyze-frame/<analysis_id>/<frame_name>', methods=['POST'])
def analyze_frame(analysis_id, frame_name):
    """Bir frame'i görsel olarak analiz et"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analiz henüz tamamlanmadı'}), 400
        
        result = status_data['result']
    
    # Frame dosyasını bul
    frame_path = os.path.join('static', result['images_dir'], frame_name)
    
    if not os.path.exists(frame_path):
        return jsonify({'error': 'Frame bulunamadı'}), 404
    
    # Ollama kontrolü
    if not ollama.check_connection():
        return jsonify({'error': 'Ollama bağlantısı yok'}), 503
    
    try:
        # İsteğe göre özel soru
        data = request.get_json() or {}
        custom_question = data.get('question', '')
        
        if custom_question:
            question = custom_question
        else:
            question = "Bu görselde ne görüyorsun? Detaylı bir şekilde açıkla."
        
        # Görseli analiz et
        analysis = ollama.analyze_image(frame_path, question)
        
        return jsonify({
            'frame': frame_name,
            'question': question,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'}), 500


@app.route('/api/smart-search/<analysis_id>', methods=['POST'])
def smart_search(analysis_id):
    """Gelişmiş arama - hem metin hem görsel"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analiz bulunamadı'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analiz henüz tamamlanmadı'}), 400
        
        result = status_data['result']
    
    data = request.get_json()
    query = data.get('query', '').strip()
    search_type = data.get('type', 'both')  # 'text', 'visual', 'both'
    
    if not query:
        return jsonify({'error': 'Arama sorgusu boş'}), 400
    
    results = {
        'query': query,
        'text_results': [],
        'visual_results': []
    }
    
    try:
        # Metin araması
        if search_type in ['text', 'both']:
            transcript_data = result.get('transcript_data', [])
            
            # Basit text matching
            for i, entry in enumerate(transcript_data):
                if query.lower() in entry['text'].lower():
                    results['text_results'].append({
                        'index': i,
                        'text': entry['text'],
                        'start': entry['start'],
                        'time_formatted': f"{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}"
                    })
                    
                    if len(results['text_results']) >= 10:
                        break
        
        # Görsel arama (AKILLI YAKLAŞIM)
        if search_type in ['visual', 'both'] and ollama.check_connection():
            frames_dir = Path('static') / result['images_dir']
            
            if frames_dir.exists():
                # YÖNTEM 1: Eğer metin araması varsa, sadece o zamanların frame'lerini analiz et
                if search_type == 'both' and len(results['text_results']) > 0:
                    # Metin bulunduğu zamanların frame'lerini al
                    for text_result in results['text_results'][:5]:  # İlk 5 sonuç
                        timestamp = text_result['start']
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        
                        # Bu zamana yakın frame'i bul
                        frame_pattern = f"*_time_{minutes:02d}m{seconds:02d}s.jpg"
                        matching_frames = list(frames_dir.glob(frame_pattern))
                        
                        if not matching_frames:
                            # ±5 saniye aralığında ara
                            for offset in range(-5, 6):
                                adj_time = timestamp + offset
                                adj_min = int(adj_time // 60)
                                adj_sec = int(adj_time % 60)
                                pattern = f"*_time_{adj_min:02d}m{adj_sec:02d}s.jpg"
                                matching_frames = list(frames_dir.glob(pattern))
                                if matching_frames:
                                    break
                        
                        if matching_frames:
                            frame_file = matching_frames[0]
                            try:
                                search_question = f"Bu görselde '{query}' ile alakalı bir şey var mı? Evet/Hayır + kısa açıklama."
                                analysis = ollama.analyze_image(str(frame_file), search_question)
                                
                                results['visual_results'].append({
                                    'frame': frame_file.name,
                                    'path': str(Path(result['images_dir']) / frame_file.name),
                                    'analysis': analysis,
                                    'related_text': text_result['text']
                                })
                            except:
                                continue
                
                else:
                    # YÖNTEM 2: Sadece görsel arama - rastgele 10 frame seç (20 yerine)
                    all_frames = list(frames_dir.glob('*.jpg'))
                    
                    # Rastgele değil, düzenli aralıklarla seç (daha iyi coverage)
                    if len(all_frames) > 10:
                        step = len(all_frames) // 10
                        selected_frames = all_frames[::step][:10]
                    else:
                        selected_frames = all_frames[:10]
                    
                    search_question = f"Bu görselde '{query}' var mı? Sadece Evet/Hayır + çok kısa açıklama (max 10 kelime)."
                    
                    for frame_file in selected_frames:
                        try:
                            analysis = ollama.analyze_image(str(frame_file), search_question)
                            
                            # Eğer "evet" içeriyorsa alakalı
                            if 'evet' in analysis.lower() or 'yes' in analysis.lower():
                                results['visual_results'].append({
                                    'frame': frame_file.name,
                                    'path': str(Path(result['images_dir']) / frame_file.name),
                                    'analysis': analysis
                                })
                                
                                if len(results['visual_results']) >= 5:
                                    break
                                    
                        except:
                            continue
        
        results['total_text'] = len(results['text_results'])
        results['total_visual'] = len(results['visual_results'])
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Hata: {str(e)}'}), 500


# Hata handler'ları
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Static/results klasörünü oluştur
    os.makedirs('static/results', exist_ok=True)
    
    # Ollama kontrolü
    print("\n🔍 Ollama Kontrolü...")
    if ollama.check_connection():
        print("✅ Ollama bağlantısı başarılı!")
        models = ollama.list_models()
        print(f"📦 Yüklü modeller: {', '.join(models[:3])}...")
    else:
        print("⚠️  Ollama bağlantısı yok! AI özellikleri çalışmayacak.")
        print("   Ollama'yı başlatın: ollama serve")
    
    print("\n" + "="*70)
    print("🎬 YOUTUBE VİDEO ANALİZ WEB UYGULAMASI")
    print("="*70)
    print("\n📱 Uygulama başlatılıyor...")
    print("🌐 Tarayıcınızda açın: http://localhost:5000")
    print("\n💡 Çıkmak için: CTRL+C\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
