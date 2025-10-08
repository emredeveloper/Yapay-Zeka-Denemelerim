import os
import re
from pathlib import Path
from datetime import datetime
import cv2
from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube
import nltk
from nltk.tokenize import sent_tokenize
import json

# Download necessary data for NLTK (on first use)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class YouTubeVideoAnalyzer:
    def __init__(self, url, output_base_dir="youtube_analysis"):
        """
        YouTube video analysis class
        
        Args:
            url: YouTube video URL
            output_base_dir: Main folder where outputs will be saved
        """
        self.url = url
        self.video_id = self._extract_video_id(url)
        self.output_base_dir = output_base_dir
        self.yt = None
        self.video_path = None
        
        # Create output directories
        self.setup_directories()
        
    def _extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError("Not a valid YouTube URL!")
    
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_dir = Path(self.output_base_dir) / f"{self.video_id}_{timestamp}"
        self.images_dir = self.project_dir / "images"
        self.text_dir = self.project_dir / "text"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directories created: {self.project_dir}")
    
    def get_video_info(self):
        """Get video information"""
        try:
            self.yt = YouTube(
                self.url,
                use_oauth=False,
                allow_oauth_cache=False
            )
            
            info = {
                "Title": self.yt.title,
                "Channel": self.yt.author,
                "Views": self.yt.views,
                "Duration (seconds)": self.yt.length,
                "Duration (mm:ss)": f"{self.yt.length // 60}:{self.yt.length % 60:02d}",
                "Publish Date": str(self.yt.publish_date),
                "Description": self.yt.description[:200] + "..." if len(self.yt.description) > 200 else self.yt.description,
                "Rating": self.yt.rating if hasattr(self.yt, 'rating') else "N/A",
                "Video ID": self.video_id,
                "Thumbnail URL": self.yt.thumbnail_url
            }
            
            print("\n" + "="*60)
            print("VIDEO INFORMATION")
            print("="*60)
            for key, value in info.items():
                print(f"{key}: {value}")
            print("="*60 + "\n")
            
            return info
            
        except Exception as e:
            print(f"⚠️  Error getting video information: {e}")
            print("⚠️  Continuing with basic information...")
            # Return basic information
            return {
                "Video ID": self.video_id,
                "URL": self.url,
                "Title": "Information not available",
                "Channel": "Information not available",
                "Views": "N/A",
                "Duration (seconds)": 0,
                "Duration (mm:ss)": "N/A",
                "Publish Date": "N/A",
                "Description": "N/A",
                "Rating": "N/A",
                "Thumbnail URL": f"https://img.youtube.com/vi/{self.video_id}/maxresdefault.jpg"
            }
    
    def download_video(self):
        """Download the video"""
        try:
            print("📥 Downloading video...")
            
            # If YouTube object doesn't exist, create it
            if not self.yt:
                self.yt = YouTube(
                    self.url,
                    use_oauth=False,
                    allow_oauth_cache=False
                )
            
            stream = self.yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if stream:
                self.video_path = stream.download(output_path=str(self.project_dir), filename="video.mp4")
                print(f"✓ Video downloaded: {self.video_path}")
                return True
            else:
                print("⚠️  No suitable video stream found!")
                return False
                
        except Exception as e:
            print(f"⚠️  Error while downloading video: {e}")
            print("⚠️  Skipping video download, transcript analysis will be performed...")
            return False
    
    def get_transcript(self):
        """Get video transcript"""
        try:
            print("📝 Getting transcript...")
            
            # Create API instance
            api = YouTubeTranscriptApi()
            
            # First list existing transcripts
            try:
                transcript_list = api.list(self.video_id)
                print(f"✓ Existing transcripts detected")
                
                available_transcripts = []
                for transcript in transcript_list:
                    trans_type = "automatic" if transcript.is_generated else "manual"
                    available_transcripts.append(f"{transcript.language} ({trans_type})")
                
                if available_transcripts:
                    print(f"  Available languages: {', '.join(available_transcripts[:5])}")
                
            except Exception as e:
                print(f"⚠️  Could not get transcript list: {e}")
                transcript_list = None
            
            transcript_data = None
            
            # 1. Try direct fetch (default language)
            try:
                transcript_data = api.fetch(self.video_id)
                print(f"✓ Default transcript found")
            except Exception as e:
                print(f"  Default transcript error: {str(e)[:100]}")
            
            # 2. Try manual Turkish/English transcript
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    if transcript.language_code in ['tr', 'en'] and not transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            print(f"✓ {transcript.language} manual transcript found")
                            break
                        except:
                            continue
            
            # 3. Try automatic English transcript
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    if transcript.language_code == 'en' and transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            print(f"✓ {transcript.language} automatic transcript found")
                            break
                        except:
                            continue
            
            # 4. Try any transcript
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    try:
                        transcript_data = transcript.fetch()
                        print(f"✓ {transcript.language} transcript found")
                        break
                    except:
                        continue
            
            if not transcript_data:
                print("❌ No transcript could be obtained!")
                print("   Transcript may be disabled for this video.")
                return None, [], []
            
            # Create full text
            full_text = " ".join([entry.text for entry in transcript_data])
            
            # Split into sentences
            try:
                sentences = sent_tokenize(full_text, language='english')
            except:
                try:
                    sentences = sent_tokenize(full_text, language='turkish')
                except:
                    # Simple dot splitting
                    sentences = [s.strip() for s in full_text.split('.') if s.strip()]
            
            # Save with timestamps
            transcript_with_time = []
            for entry in transcript_data:
                transcript_with_time.append({
                    'start': entry.start,
                    'duration': entry.duration,
                    'text': entry.text
                })
            
            print(f"✓ Total {len(transcript_data)} transcript segments obtained")
            print(f"✓ Total {len(sentences)} sentences detected")
            
            return full_text, transcript_with_time, sentences
            
        except Exception as e:
            print(f"❌ Critical error while getting transcript: {e}")
            import traceback
            print(traceback.format_exc())
            return None, [], []
    
    def extract_frames(self, interval_seconds=30):
        """
        Extract frames from video at specific intervals
        
        Args:
            interval_seconds: Frame extraction interval (seconds)
        """
        if not self.video_path or not os.path.exists(self.video_path):
            print("❌ Video file not found!")
            return []
        
        try:
            print(f"🖼️  Extracting frames (every {interval_seconds} seconds)...")
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            frame_interval = int(fps * interval_seconds)
            extracted_frames = []
            
            frame_num = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_num % frame_interval == 0:
                    timestamp = frame_num / fps
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    
                    frame_filename = f"frame_{saved_count:04d}_time_{minutes:02d}m{seconds:02d}s.jpg"
                    frame_path = self.images_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append({
                        'filename': frame_filename,
                        'timestamp': timestamp,
                        'time_formatted': f"{minutes:02d}:{seconds:02d}"
                    })
                    saved_count += 1
                
                frame_num += 1
            
            cap.release()
            print(f"✓ {len(extracted_frames)} frames extracted and saved")
            
            return extracted_frames
            
        except Exception as e:
            print(f"❌ Error while extracting frames: {e}")
            return []
    
    def extract_sentence_frames(self, transcript_with_time, sentences):
        """
        Extract frames at the end of each sentence or question
        """
        if not self.video_path or not os.path.exists(self.video_path):
            print("❌ Video file not found!")
            return []
        
        try:
            print("🖼️  Extracting frames at sentence endings...")
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            sentence_frames = []
            
            # Detect sentence endings (ending with . ? !)
            sentence_end_times = []
            current_text = ""
            
            for entry in transcript_with_time:
                current_text += " " + entry['text']
                text = current_text.strip()
                
                # Check for sentence or question ending
                if text.endswith('.') or text.endswith('?') or text.endswith('!'):
                    sentence_end_times.append({
                        'time': entry['start'] + entry['duration'],
                        'text': text
                    })
                    current_text = ""
            
            # Extract frame at each sentence ending
            for idx, sentence_info in enumerate(sentence_end_times):
                timestamp = sentence_info['time']
                frame_position = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                
                if ret:
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    
                    frame_filename = f"sentence_{idx:04d}_time_{minutes:02d}m{seconds:02d}s.jpg"
                    frame_path = self.images_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    sentence_frames.append({
                        'filename': frame_filename,
                        'timestamp': timestamp,
                        'time_formatted': f"{minutes:02d}:{seconds:02d}",
                        'sentence': sentence_info['text'][:100] + "..." if len(sentence_info['text']) > 100 else sentence_info['text']
                    })
            
            cap.release()
            print(f"✓ {len(sentence_frames)} sentence ending frames extracted")
            
            return sentence_frames
            
        except Exception as e:
            print(f"❌ Error while extracting sentence frames: {e}")
            return []
    
    def save_text_data(self, video_info, full_text, transcript_with_time, sentences, sentence_frames):
        """Save text data"""
        try:
            print("💾 Saving text data...")
            
            # 1. Save video information
            info_path = self.text_dir / "video_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("VIDEO INFORMATION\n")
                f.write("="*60 + "\n\n")
                for key, value in video_info.items():
                    f.write(f"{key}: {value}\n")
            
            # 2. Full transcript text
            text_path = self.text_dir / "full_transcript.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # 3. Timed transcript
            timed_path = self.text_dir / "timed_transcript.txt"
            with open(timed_path, 'w', encoding='utf-8') as f:
                for entry in transcript_with_time:
                    minutes = int(entry['start'] // 60)
                    seconds = int(entry['start'] % 60)
                    f.write(f"[{minutes:02d}:{seconds:02d}] {entry['text']}\n")
            
            # 4. Sentences
            sentences_path = self.text_dir / "sentences.txt"
            with open(sentences_path, 'w', encoding='utf-8') as f:
                for idx, sentence in enumerate(sentences, 1):
                    f.write(f"{idx}. {sentence}\n\n")
            
            # 5. Sentence frame information
            if sentence_frames:
                frames_info_path = self.text_dir / "sentence_frames_info.txt"
                with open(frames_info_path, 'w', encoding='utf-8') as f:
                    f.write("SENTENCE ENDING FRAMES\n")
                    f.write("="*60 + "\n\n")
                    for frame_info in sentence_frames:
                        f.write(f"File: {frame_info['filename']}\n")
                        f.write(f"Time: {frame_info['time_formatted']}\n")
                        f.write(f"Sentence: {frame_info['sentence']}\n")
                        f.write("-"*60 + "\n\n")
            
            # 6. All data in JSON format
            json_path = self.text_dir / "analysis_data.json"
            analysis_data = {
                'video_info': video_info,
                'transcript': transcript_with_time,
                'sentences': sentences,
                'sentence_frames': sentence_frames
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ All text data saved: {self.text_dir}")
            
        except Exception as e:
            print(f"❌ Error while saving text data: {e}")
    
    def generate_summary_report(self, video_info, full_text, sentence_frames, interval_frames):
        """Generate summary report"""
        try:
            report_path = self.project_dir / "SUMMARY_REPORT.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("YOUTUBE VIDEO ANALYSIS REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write("📊 VIDEO STATISTICS\n")
                f.write("-"*70 + "\n")
                f.write(f"Video ID: {self.video_id}\n")
                f.write(f"Title: {video_info.get('Title', 'N/A')}\n")
                f.write(f"Channel: {video_info.get('Channel', 'N/A')}\n")
                f.write(f"Views: {video_info.get('Views', 'N/A'):,}\n")
                f.write(f"Duration: {video_info.get('Duration (mm:ss)', 'N/A')}\n")
                f.write(f"Publish Date: {video_info.get('Publish Date', 'N/A')}\n\n")
                
                f.write("📝 TEXT ANALYSIS\n")
                f.write("-"*70 + "\n")
                if full_text:
                    words = full_text.split()
                    f.write(f"Total Word Count: {len(words):,}\n")
                    f.write(f"Total Character Count: {len(full_text):,}\n")
                    f.write(f"Average Word Length: {sum(len(word) for word in words) / len(words):.2f}\n\n")
                
                f.write("🖼️  VISUAL ANALYSIS\n")
                f.write("-"*70 + "\n")
                f.write(f"Sentence Ending Frame Count: {len(sentence_frames)}\n")
                f.write(f"Regular Interval Frame Count: {len(interval_frames)}\n")
                f.write(f"Total Frame Count: {len(sentence_frames) + len(interval_frames)}\n\n")
                
                f.write("📁 OUTPUT FILES\n")
                f.write("-"*70 + "\n")
                f.write(f"Project Folder: {self.project_dir}\n")
                f.write(f"Images Folder: {self.images_dir}\n")
                f.write(f"Texts Folder: {self.text_dir}\n")
                f.write(f"Video File: {self.video_path}\n\n")
                
                f.write("="*70 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n")
            
            print(f"\n✓ Summary report created: {report_path}")
            
        except Exception as e:
            print(f"❌ Error while creating report: {e}")
    
    def analyze(self, extract_interval_frames=True, interval_seconds=30):
        """
        Perform full analysis
        
        Args:
            extract_interval_frames: Extract frames at regular intervals (True/False)
            interval_seconds: Frame extraction interval (seconds)
        """
        print("\n" + "="*70)
        print("🎬 YOUTUBE VIDEO ANALYSIS STARTING")
        print("="*70 + "\n")
        
        # 1. Get video information
        video_info = self.get_video_info()
        
        # 2. Videoyu indir
        if not self.download_video():
            print("⚠️  Video could not be downloaded, only transcript analysis will be performed")
        
        # 3. Get transcript
        full_text, transcript_with_time, sentences = self.get_transcript()
        
        if not full_text:
            print("⚠️  Transcript could not be obtained!")
            
            # If video exists, perform only frame analysis
            if self.video_path and os.path.exists(self.video_path):
                print("📊 Video exists, only frame analysis will be performed...")
                interval_frames = []
                if extract_interval_frames:
                    interval_frames = self.extract_frames(interval_seconds)
                
                # Create minimal report
                self.generate_summary_report(video_info, "", [], interval_frames)
                print("\n" + "="*70)
                print("✅ ANALYSIS COMPLETED (Only Video Frames)!")
                print("="*70)
                print(f"\n📁 All outputs here: {self.project_dir}")
                print(f"🖼️  Images: {self.images_dir}")
                print("\n")
                return
            else:
                print("❌ Neither transcript nor video available, analysis terminated!")
                return
        
        # 4. Extract frames
        sentence_frames = []
        interval_frames = []
        
        if self.video_path and os.path.exists(self.video_path):
            # Frames at sentence endings
            sentence_frames = self.extract_sentence_frames(transcript_with_time, sentences)
            
            # Frames at regular intervals (optional)
            if extract_interval_frames:
                interval_frames = self.extract_frames(interval_seconds)
        else:
            print("⚠️  Video file not available, only transcript analysis performed")
        
        # 5. Save text data
        self.save_text_data(video_info, full_text, transcript_with_time, sentences, sentence_frames)
        
        # 6. Generate summary report
        self.generate_summary_report(video_info, full_text, sentence_frames, interval_frames)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED!")
        print("="*70)
        print(f"\n📁 All outputs here: {self.project_dir}")
        print(f"🖼️  Images: {self.images_dir}")
        print(f"📝 Texts: {self.text_dir}")
        print("\n")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("🎥 YOUTUBE VIDEO ANALYSIS TOOL")
    print("="*70 + "\n")
    
    # Get URL from user
    url = input("Enter YouTube video URL: ").strip()
    
    if not url:
        print("❌ URL cannot be empty!")
        return
    
    # Settings
    print("\n⚙️  Settings:")
    extract_interval = input("Extract frames at regular intervals? (Y/N, default: Y): ").strip().upper()
    extract_interval_frames = extract_interval != 'N'
    
    interval_seconds = 30
    if extract_interval_frames:
        interval_input = input("Frame extraction interval (seconds, default: 30): ").strip()
        if interval_input.isdigit():
            interval_seconds = int(interval_input)
    
    # Start analysis
    try:
        analyzer = YouTubeVideoAnalyzer(url)
        analyzer.analyze(
            extract_interval_frames=extract_interval_frames,
            interval_seconds=interval_seconds
        )
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
