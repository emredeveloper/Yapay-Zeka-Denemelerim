"""
Flask Web Arayüzü - YouTube Video Analiz Uygulaması
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import json
import threading
from pathlib import Path
from datetime import datetime
import uuid

# YouTube analyzer'ı import et
from youtube_app import YouTubeVideoAnalyzer

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global analiz durumu takibi
analysis_status = {}
analysis_lock = threading.Lock()


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
            'transcript_data': transcript_with_time if transcript_with_time else []
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
    
    print("\n" + "="*70)
    print("🎬 YOUTUBE VİDEO ANALİZ WEB UYGULAMASI")
    print("="*70)
    print("\n📱 Uygulama başlatılıyor...")
    print("🌐 Tarayıcınızda açın: http://localhost:5000")
    print("\n💡 Çıkmak için: CTRL+C\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
