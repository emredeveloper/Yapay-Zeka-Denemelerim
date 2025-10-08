"""
Flask Web Interface - YouTube Video Analysis Application
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, session, Response
from werkzeug.utils import secure_filename
import os
import json
import threading
from pathlib import Path
from datetime import datetime
import uuid

# Import YouTube analyzer
from youtube_app import YouTubeVideoAnalyzer
# Import Ollama client
from ollama_client import OllamaClient

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global analysis status tracking
analysis_status = {}
analysis_lock = threading.Lock()

# Ollama client
ollama = OllamaClient()


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis (async)"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL cannot be empty!'}), 400
        
        # Settings
        extract_interval = data.get('extract_interval', True)
        interval_seconds = int(data.get('interval_seconds', 30))
        
        # Create unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize status
        with analysis_lock:
            analysis_status[analysis_id] = {
                'status': 'starting',
                'progress': 0,
                'message': 'Analysis starting...',
                'result': None,
                'error': None
            }
        
        # Start analysis in separate thread
        thread = threading.Thread(
            target=run_analysis,
            args=(analysis_id, url, extract_interval, interval_seconds)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'message': 'Analysis started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_analysis(analysis_id, url, extract_interval, interval_seconds):
    """Run analysis (background thread)"""
    try:
        # Update status: Starting
        update_status(analysis_id, 'running', 10, 'Getting video information...')
        
        # Create analyzer
        analyzer = YouTubeVideoAnalyzer(url, output_base_dir="static/results")
        
        # Get video information
        update_status(analysis_id, 'running', 20, 'Video information obtained')
        video_info = analyzer.get_video_info()
        
        # Download video
        update_status(analysis_id, 'running', 30, 'Downloading video...')
        video_downloaded = analyzer.download_video()
        
        # Get transcript
        update_status(analysis_id, 'running', 50, 'Getting transcript...')
        full_text, transcript_with_time, sentences = analyzer.get_transcript()
        
        if not full_text and not video_downloaded:
            update_status(analysis_id, 'error', 0, 'Neither transcript nor video could be obtained!', error='Video inaccessible')
            return
        
        # Extract frames
        sentence_frames = []
        interval_frames = []
        
        if analyzer.video_path and os.path.exists(analyzer.video_path):
            if full_text:
                update_status(analysis_id, 'running', 60, 'Extracting sentence frames...')
                sentence_frames = analyzer.extract_sentence_frames(transcript_with_time, sentences)
            
            if extract_interval:
                update_status(analysis_id, 'running', 75, 'Extracting regular frames...')
                interval_frames = analyzer.extract_frames(interval_seconds)
        
        # Save text data
        update_status(analysis_id, 'running', 85, 'Saving data...')
        if full_text:
            analyzer.save_text_data(video_info, full_text, transcript_with_time, sentences, sentence_frames)
        
        # Create report
        update_status(analysis_id, 'running', 95, 'Creating report...')
        analyzer.generate_summary_report(video_info, full_text if full_text else "", sentence_frames, interval_frames)
        
        # Prepare result
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
            'sentence_frames': sentence_frames[:50],  # First 50 frames
            'interval_frames': interval_frames[:50],
            'transcript_data': transcript_with_time if transcript_with_time else [],
            'full_text': full_text if full_text else ''  # Full transcript text
        }
        
        # Completed
        update_status(analysis_id, 'completed', 100, 'Analysis completed!', result=result)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        update_status(analysis_id, 'error', 0, f'Error: {str(e)}', error=error_msg)


def update_status(analysis_id, status, progress, message, result=None, error=None):
    """Update analysis status"""
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
    """Query analysis status"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify(analysis_status[analysis_id])


@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    """Show analysis results"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analysis not found", 404
        
        status_data = analysis_status[analysis_id]
        
        if status_data['status'] != 'completed':
            return "Analysis not yet completed", 400
        
        result = status_data['result']
        
    return render_template('results.html', 
                         analysis_id=analysis_id,
                         result=result)


@app.route('/gallery/<analysis_id>')
def gallery(analysis_id):
    """Frame gallery"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analysis not found", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analysis not yet completed", 400
        
        result = status_data['result']
    
    # Get all frames
    images_dir = Path('static') / result['images_dir']
    
    all_frames = []
    if images_dir.exists():
        for img_file in sorted(images_dir.glob('*.jpg')):
            frame_info = {
                'filename': img_file.name,
                'path': str(Path(result['images_dir']) / img_file.name),
                'type': 'sentence' if 'sentence' in img_file.name else 'interval'
            }
            
            # Extract time information
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
    """View and search transcript"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analysis not found", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analysis not yet completed", 400
        
        result = status_data['result']
    
    # Search sorgusu
    query = request.args.get('q', '').strip()
    
    transcript_data = result.get('transcript_data', [])
    
    # Search yap
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
    """Transcript search API"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analysis not found'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analysis not yet completed'}), 400
        
        result = status_data['result']
    
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Search query empty'}), 400
    
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
    """Download file"""
    try:
        # filepath already comes in results/xxxx/file.txt format
        full_path = os.path.join('static', filepath)
        
        if not os.path.exists(full_path):
            return f"File not found: {filepath}", 404
        
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return f"Download error: {str(e)}", 500


# ============================================================
# OLLAMA & AI FEATURES
# ============================================================

@app.route('/qa/<analysis_id>')
def qa_page(analysis_id):
    """Video Q&A page"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return "Analysis not found", 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return "Analysis not yet completed", 400
        
        result = status_data['result']
    
    # Check Ollama status
    ollama_available = ollama.check_connection()
    
    return render_template('qa.html',
                         analysis_id=analysis_id,
                         result=result,
                         ollama_available=ollama_available)


@app.route('/api/ask/<analysis_id>', methods=['POST'])
def ask_question(analysis_id):
    """Ask question about video"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analysis not found'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analysis not yet completed'}), 400
        
        result = status_data['result']
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    # Check Ollama
    if not ollama.check_connection():
        return jsonify({'error': 'No Ollama connection. Is Ollama running?'}), 503
    
    try:
        # Prepare transcript
        transcript_text = result.get('full_text', '')
        
        # Prepare video information
        video_info_data = result.get('video_info', {})
        video_info = {
            'title': video_info_data.get('title', 'N/A'),
            'channel': video_info_data.get('channel', 'N/A'),
            'duration': video_info_data.get('duration_formatted', 'N/A'),
            'views': video_info_data.get('views', 'N/A')
        }
        
        # Find relevant frames (optional - disabled for performance)
        relevant_frames = []
        use_frames = data.get('use_frames', False)  # Parameter from frontend
        
        if use_frames and 'sentence_frames' in result:
            # Take only first 2 frames (for speed)
            for frame_info in result['sentence_frames'][:2]:
                # frame_info is a dict, has filename key
                if isinstance(frame_info, dict):
                    frame_filename = frame_info.get('filename', '')
                else:
                    # If string, use directly
                    frame_filename = frame_info
                
                if frame_filename:
                    frame_path = os.path.join('static', result['images_dir'], frame_filename)
                    if os.path.exists(frame_path):
                        relevant_frames.append(frame_path)
        
        # Answer the question
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
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/analyze-frame/<analysis_id>/<frame_name>', methods=['POST'])
def analyze_frame(analysis_id, frame_name):
    """Analyze a frame visually"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analysis not found'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analysis not yet completed'}), 400
        
        result = status_data['result']
    
    # Find frame file
    frame_path = os.path.join('static', result['images_dir'], frame_name)
    
    if not os.path.exists(frame_path):
        return jsonify({'error': 'Frame not found'}), 404
    
    # Check Ollama
    if not ollama.check_connection():
        return jsonify({'error': 'No Ollama connection'}), 503
    
    try:
        # Custom question as requested
        data = request.get_json() or {}
        custom_question = data.get('question', '')
        
        if custom_question:
            question = custom_question
        else:
            question = "What do you see in this image? Explain in detail."
        
        # Analyze the image
        analysis = ollama.analyze_image(frame_path, question)
        
        return jsonify({
            'frame': frame_name,
            'question': question,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/smart-search/<analysis_id>', methods=['POST'])
def smart_search(analysis_id):
    """Advanced search - both text and visual"""
    with analysis_lock:
        if analysis_id not in analysis_status:
            return jsonify({'error': 'Analysis not found'}), 404
        
        status_data = analysis_status[analysis_id]
        if status_data['status'] != 'completed':
            return jsonify({'error': 'Analysis not yet completed'}), 400
        
        result = status_data['result']
    
    data = request.get_json()
    query = data.get('query', '').strip()
    search_type = data.get('type', 'both')  # 'text', 'visual', 'both'
    
    if not query:
        return jsonify({'error': 'Search query empty'}), 400
    
    results = {
        'query': query,
        'text_results': [],
        'visual_results': []
    }
    
    try:
        # Text search
        if search_type in ['text', 'both']:
            transcript_data = result.get('transcript_data', [])
            
            # Simple text matching
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
        
        # Visual search (SMART APPROACH)
        if search_type in ['visual', 'both'] and ollama.check_connection():
            frames_dir = Path('static') / result['images_dir']
            
            if frames_dir.exists():
                # METHOD 1: If text search exists, analyze only frames from those times
                if search_type == 'both' and len(results['text_results']) > 0:
                    # Get frames from times where text was found
                    for text_result in results['text_results'][:5]:  # First 5 results
                        timestamp = text_result['start']
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        
                        # Find frame close to this time
                        frame_pattern = f"*_time_{minutes:02d}m{seconds:02d}s.jpg"
                        matching_frames = list(frames_dir.glob(frame_pattern))
                        
                        if not matching_frames:
                            # Search within ±5 seconds
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
                                search_question = f"Is there anything related to '{query}' in this image? Yes/No + short explanation."
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
                    # METHOD 2: Visual search only - select 10 random frames (instead of 20)
                    all_frames = list(frames_dir.glob('*.jpg'))
                    
                    # Not random, select at regular intervals (better coverage)
                    if len(all_frames) > 10:
                        step = len(all_frames) // 10
                        selected_frames = all_frames[::step][:10]
                    else:
                        selected_frames = all_frames[:10]
                    
                    search_question = f"Is '{query}' in this image? Yes/No + very short explanation (max 10 words)."
                    
                    for frame_file in selected_frames:
                        try:
                            analysis = ollama.analyze_image(str(frame_file), search_question)
                            
                            # If contains "yes", relevant
                            if 'yes' in analysis.lower() or 'evet' in analysis.lower():
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
        return jsonify({'error': f'Error: {str(e)}'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create static/results directory
    os.makedirs('static/results', exist_ok=True)
    
    # Ollama check
    print("\n🔍 Ollama Check...")
    if ollama.check_connection():
        print("✅ Ollama connection successful!")
        models = ollama.list_models()
        print(f"📦 Installed models: {', '.join(models[:3])}...")
    else:
        print("⚠️  No Ollama connection! AI features will not work.")
        print("   Start Ollama: ollama serve")
    
    print("\n" + "="*70)
    print("🎬 YOUTUBE VIDEO ANALYSIS WEB APPLICATION")
    print("="*70)
    print("\n📱 Application starting...")
    print("🌐 Open in browser: http://localhost:5000")
    print("\n💡 To exit: CTRL+C\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
