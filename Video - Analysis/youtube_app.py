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

# NLTK için gerekli veri indirme (ilk kullanımda)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class YouTubeVideoAnalyzer:
    def __init__(self, url, output_base_dir="youtube_analysis"):
        """
        YouTube video analiz sınıfı
        
        Args:
            url: YouTube video URL'si
            output_base_dir: Çıktıların kaydedileceği ana klasör
        """
        self.url = url
        self.video_id = self._extract_video_id(url)
        self.output_base_dir = output_base_dir
        self.yt = None
        self.video_path = None
        
        # Çıktı klasörlerini oluştur
        self.setup_directories()
        
    def _extract_video_id(self, url):
        """YouTube URL'sinden video ID'sini çıkar"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
            r'youtube\.com\/embed\/([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError("Geçerli bir YouTube URL'si değil!")
    
    def setup_directories(self):
        """Çıktı klasörlerini oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_dir = Path(self.output_base_dir) / f"{self.video_id}_{timestamp}"
        self.images_dir = self.project_dir / "images"
        self.text_dir = self.project_dir / "text"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Çıktı klasörleri oluşturuldu: {self.project_dir}")
    
    def get_video_info(self):
        """Video hakkında bilgi al"""
        try:
            self.yt = YouTube(
                self.url,
                use_oauth=False,
                allow_oauth_cache=False
            )
            
            info = {
                "Başlık": self.yt.title,
                "Kanal": self.yt.author,
                "Görüntülenme": self.yt.views,
                "Süre (saniye)": self.yt.length,
                "Süre (dk:sn)": f"{self.yt.length // 60}:{self.yt.length % 60:02d}",
                "Yayın Tarihi": str(self.yt.publish_date),
                "Açıklama": self.yt.description[:200] + "..." if len(self.yt.description) > 200 else self.yt.description,
                "Derecelendirme": self.yt.rating if hasattr(self.yt, 'rating') else "N/A",
                "Video ID": self.video_id,
                "Thumbnail URL": self.yt.thumbnail_url
            }
            
            print("\n" + "="*60)
            print("VIDEO BİLGİLERİ")
            print("="*60)
            for key, value in info.items():
                print(f"{key}: {value}")
            print("="*60 + "\n")
            
            return info
            
        except Exception as e:
            print(f"⚠️  Video bilgisi alınırken hata: {e}")
            print("⚠️  Temel bilgilerle devam ediliyor...")
            # Temel bilgileri döndür
            return {
                "Video ID": self.video_id,
                "URL": self.url,
                "Başlık": "Bilgi alınamadı",
                "Kanal": "Bilgi alınamadı",
                "Görüntülenme": "N/A",
                "Süre (saniye)": 0,
                "Süre (dk:sn)": "N/A",
                "Yayın Tarihi": "N/A",
                "Açıklama": "N/A",
                "Derecelendirme": "N/A",
                "Thumbnail URL": f"https://img.youtube.com/vi/{self.video_id}/maxresdefault.jpg"
            }
    
    def download_video(self):
        """Videoyu indir"""
        try:
            print("📥 Video indiriliyor...")
            
            # Eğer YouTube objesi yoksa oluştur
            if not self.yt:
                self.yt = YouTube(
                    self.url,
                    use_oauth=False,
                    allow_oauth_cache=False
                )
            
            stream = self.yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if stream:
                self.video_path = stream.download(output_path=str(self.project_dir), filename="video.mp4")
                print(f"✓ Video indirildi: {self.video_path}")
                return True
            else:
                print("⚠️  Uygun video stream'i bulunamadı!")
                return False
                
        except Exception as e:
            print(f"⚠️  Video indirilirken hata: {e}")
            print("⚠️  Video indirme atlanıyor, transkript analizi yapılacak...")
            return False
    
    def get_transcript(self):
        """Video transkriptini al"""
        try:
            print("📝 Transkript alınıyor...")
            
            # API instance oluştur
            api = YouTubeTranscriptApi()
            
            # Önce mevcut transkriptleri listele
            try:
                transcript_list = api.list(self.video_id)
                print(f"✓ Mevcut transkriptler tespit edildi")
                
                available_transcripts = []
                for transcript in transcript_list:
                    trans_type = "otomatik" if transcript.is_generated else "manuel"
                    available_transcripts.append(f"{transcript.language} ({trans_type})")
                
                if available_transcripts:
                    print(f"  Mevcut diller: {', '.join(available_transcripts[:5])}")
                
            except Exception as e:
                print(f"⚠️  Transkript listesi alınamadı: {e}")
                transcript_list = None
            
            transcript_data = None
            
            # 1. Direkt fetch dene (varsayılan dil)
            try:
                transcript_data = api.fetch(self.video_id)
                print(f"✓ Varsayılan transkript bulundu")
            except Exception as e:
                print(f"  Varsayılan transkript hatası: {str(e)[:100]}")
            
            # 2. Manuel Türkçe/İngilizce transkript dene
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    if transcript.language_code in ['tr', 'en'] and not transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            print(f"✓ {transcript.language} manuel transkript bulundu")
                            break
                        except:
                            continue
            
            # 3. Otomatik İngilizce transkript dene
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    if transcript.language_code == 'en' and transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            print(f"✓ {transcript.language} otomatik transkript bulundu")
                            break
                        except:
                            continue
            
            # 4. Herhangi bir transkript dene
            if not transcript_data and transcript_list:
                for transcript in transcript_list:
                    try:
                        transcript_data = transcript.fetch()
                        print(f"✓ {transcript.language} transkript bulundu")
                        break
                    except:
                        continue
            
            if not transcript_data:
                print("❌ Hiçbir transkript alınamadı!")
                print("   Bu video için transkript devre dışı olabilir.")
                return None, [], []
            
            # Tam metni oluştur
            full_text = " ".join([entry.text for entry in transcript_data])
            
            # Cümlelere ayır
            try:
                sentences = sent_tokenize(full_text, language='english')
            except:
                try:
                    sentences = sent_tokenize(full_text, language='turkish')
                except:
                    # Basit nokta ile ayırma
                    sentences = [s.strip() for s in full_text.split('.') if s.strip()]
            
            # Zaman damgalarıyla birlikte kaydet
            transcript_with_time = []
            for entry in transcript_data:
                transcript_with_time.append({
                    'start': entry.start,
                    'duration': entry.duration,
                    'text': entry.text
                })
            
            print(f"✓ Toplam {len(transcript_data)} transkript segmenti alındı")
            print(f"✓ Toplam {len(sentences)} cümle tespit edildi")
            
            return full_text, transcript_with_time, sentences
            
        except Exception as e:
            print(f"❌ Transkript alınırken kritik hata: {e}")
            import traceback
            print(traceback.format_exc())
            return None, [], []
    
    def extract_frames(self, interval_seconds=30):
        """
        Videodan belirli aralıklarla frame çıkar
        
        Args:
            interval_seconds: Frame çıkarma aralığı (saniye)
        """
        if not self.video_path or not os.path.exists(self.video_path):
            print("❌ Video dosyası bulunamadı!")
            return []
        
        try:
            print(f"🖼️  Frame'ler çıkarılıyor (her {interval_seconds} saniyede bir)...")
            
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
            print(f"✓ {len(extracted_frames)} frame çıkarıldı ve kaydedildi")
            
            return extracted_frames
            
        except Exception as e:
            print(f"❌ Frame çıkarılırken hata: {e}")
            return []
    
    def extract_sentence_frames(self, transcript_with_time, sentences):
        """
        Her cümle veya soru bitiminde frame çıkar
        """
        if not self.video_path or not os.path.exists(self.video_path):
            print("❌ Video dosyası bulunamadı!")
            return []
        
        try:
            print("🖼️  Cümle bitimlerinde frame'ler çıkarılıyor...")
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            sentence_frames = []
            
            # Cümle sonlarını tespit et (. ? ! ile bitenler)
            sentence_end_times = []
            current_text = ""
            
            for entry in transcript_with_time:
                current_text += " " + entry['text']
                text = current_text.strip()
                
                # Cümle veya soru bitişi kontrolü
                if text.endswith('.') or text.endswith('?') or text.endswith('!'):
                    sentence_end_times.append({
                        'time': entry['start'] + entry['duration'],
                        'text': text
                    })
                    current_text = ""
            
            # Her cümle bitiminde frame çıkar
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
            print(f"✓ {len(sentence_frames)} cümle bitimi frame'i çıkarıldı")
            
            return sentence_frames
            
        except Exception as e:
            print(f"❌ Cümle frame'leri çıkarılırken hata: {e}")
            return []
    
    def save_text_data(self, video_info, full_text, transcript_with_time, sentences, sentence_frames):
        """Metin verilerini kaydet"""
        try:
            print("💾 Metin verileri kaydediliyor...")
            
            # 1. Video bilgilerini kaydet
            info_path = self.text_dir / "video_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("VIDEO BİLGİLERİ\n")
                f.write("="*60 + "\n\n")
                for key, value in video_info.items():
                    f.write(f"{key}: {value}\n")
            
            # 2. Tam transkript metni
            text_path = self.text_dir / "full_transcript.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # 3. Zaman damgalı transkript
            timed_path = self.text_dir / "timed_transcript.txt"
            with open(timed_path, 'w', encoding='utf-8') as f:
                for entry in transcript_with_time:
                    minutes = int(entry['start'] // 60)
                    seconds = int(entry['start'] % 60)
                    f.write(f"[{minutes:02d}:{seconds:02d}] {entry['text']}\n")
            
            # 4. Cümleler
            sentences_path = self.text_dir / "sentences.txt"
            with open(sentences_path, 'w', encoding='utf-8') as f:
                for idx, sentence in enumerate(sentences, 1):
                    f.write(f"{idx}. {sentence}\n\n")
            
            # 5. Cümle frame bilgileri
            if sentence_frames:
                frames_info_path = self.text_dir / "sentence_frames_info.txt"
                with open(frames_info_path, 'w', encoding='utf-8') as f:
                    f.write("CÜMLE BİTİMİ FRAME'LERİ\n")
                    f.write("="*60 + "\n\n")
                    for frame_info in sentence_frames:
                        f.write(f"Dosya: {frame_info['filename']}\n")
                        f.write(f"Zaman: {frame_info['time_formatted']}\n")
                        f.write(f"Cümle: {frame_info['sentence']}\n")
                        f.write("-"*60 + "\n\n")
            
            # 6. JSON formatında tüm veri
            json_path = self.text_dir / "analysis_data.json"
            analysis_data = {
                'video_info': video_info,
                'transcript': transcript_with_time,
                'sentences': sentences,
                'sentence_frames': sentence_frames
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Tüm metin verileri kaydedildi: {self.text_dir}")
            
        except Exception as e:
            print(f"❌ Metin verileri kaydedilirken hata: {e}")
    
    def generate_summary_report(self, video_info, full_text, sentence_frames, interval_frames):
        """Özet rapor oluştur"""
        try:
            report_path = self.project_dir / "SUMMARY_REPORT.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("YOUTUBE VİDEO ANALİZ RAPORU\n")
                f.write("="*70 + "\n\n")
                
                f.write("📊 VİDEO İSTATİSTİKLERİ\n")
                f.write("-"*70 + "\n")
                f.write(f"Video ID: {self.video_id}\n")
                f.write(f"Başlık: {video_info.get('Başlık', 'N/A')}\n")
                f.write(f"Kanal: {video_info.get('Kanal', 'N/A')}\n")
                f.write(f"Görüntülenme: {video_info.get('Görüntülenme', 'N/A'):,}\n")
                f.write(f"Süre: {video_info.get('Süre (dk:sn)', 'N/A')}\n")
                f.write(f"Yayın Tarihi: {video_info.get('Yayın Tarihi', 'N/A')}\n\n")
                
                f.write("📝 METİN ANALİZİ\n")
                f.write("-"*70 + "\n")
                if full_text:
                    words = full_text.split()
                    f.write(f"Toplam Kelime Sayısı: {len(words):,}\n")
                    f.write(f"Toplam Karakter Sayısı: {len(full_text):,}\n")
                    f.write(f"Ortalama Kelime Uzunluğu: {sum(len(word) for word in words) / len(words):.2f}\n\n")
                
                f.write("🖼️  GÖRSEL ANALİZİ\n")
                f.write("-"*70 + "\n")
                f.write(f"Cümle Bitimi Frame Sayısı: {len(sentence_frames)}\n")
                f.write(f"Düzenli Aralık Frame Sayısı: {len(interval_frames)}\n")
                f.write(f"Toplam Frame Sayısı: {len(sentence_frames) + len(interval_frames)}\n\n")
                
                f.write("📁 ÇIKTI DOSYALARI\n")
                f.write("-"*70 + "\n")
                f.write(f"Proje Klasörü: {self.project_dir}\n")
                f.write(f"Görseller Klasörü: {self.images_dir}\n")
                f.write(f"Metinler Klasörü: {self.text_dir}\n")
                f.write(f"Video Dosyası: {self.video_path}\n\n")
                
                f.write("="*70 + "\n")
                f.write(f"Analiz Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n")
            
            print(f"\n✓ Özet rapor oluşturuldu: {report_path}")
            
        except Exception as e:
            print(f"❌ Rapor oluşturulurken hata: {e}")
    
    def analyze(self, extract_interval_frames=True, interval_seconds=30):
        """
        Tam analiz yap
        
        Args:
            extract_interval_frames: Düzenli aralıklarla frame çıkar (True/False)
            interval_seconds: Frame çıkarma aralığı (saniye)
        """
        print("\n" + "="*70)
        print("🎬 YOUTUBE VİDEO ANALİZİ BAŞLIYOR")
        print("="*70 + "\n")
        
        # 1. Video bilgilerini al
        video_info = self.get_video_info()
        
        # 2. Videoyu indir
        if not self.download_video():
            print("⚠️  Video indirilemedi, sadece transkript analizi yapılacak")
        
        # 3. Transkripti al
        full_text, transcript_with_time, sentences = self.get_transcript()
        
        if not full_text:
            print("⚠️  Transkript alınamadı!")
            
            # Video varsa sadece frame analizi yap
            if self.video_path and os.path.exists(self.video_path):
                print("📊 Video mevcut, sadece frame analizi yapılacak...")
                interval_frames = []
                if extract_interval_frames:
                    interval_frames = self.extract_frames(interval_seconds)
                
                # Minimal rapor oluştur
                self.generate_summary_report(video_info, "", [], interval_frames)
                print("\n" + "="*70)
                print("✅ ANALİZ TAMAMLANDI (Sadece Video Frame'leri)!")
                print("="*70)
                print(f"\n📁 Tüm çıktılar şurada: {self.project_dir}")
                print(f"🖼️  Görseller: {self.images_dir}")
                print("\n")
                return
            else:
                print("❌ Ne transkript ne de video mevcut, analiz sonlandırılıyor!")
                return
        
        # 4. Frame'leri çıkar
        sentence_frames = []
        interval_frames = []
        
        if self.video_path and os.path.exists(self.video_path):
            # Cümle bitimlerinde frame'ler
            sentence_frames = self.extract_sentence_frames(transcript_with_time, sentences)
            
            # Düzenli aralıklarla frame'ler (isteğe bağlı)
            if extract_interval_frames:
                interval_frames = self.extract_frames(interval_seconds)
        else:
            print("⚠️  Video dosyası yok, sadece transkript analizi yapıldı")
        
        # 5. Metin verilerini kaydet
        self.save_text_data(video_info, full_text, transcript_with_time, sentences, sentence_frames)
        
        # 6. Özet rapor oluştur
        self.generate_summary_report(video_info, full_text, sentence_frames, interval_frames)
        
        print("\n" + "="*70)
        print("✅ ANALİZ TAMAMLANDI!")
        print("="*70)
        print(f"\n📁 Tüm çıktılar şurada: {self.project_dir}")
        print(f"🖼️  Görseller: {self.images_dir}")
        print(f"📝 Metinler: {self.text_dir}")
        print("\n")


def main():
    """Ana fonksiyon"""
    print("\n" + "="*70)
    print("🎥 YOUTUBE VİDEO ANALİZ ARACI")
    print("="*70 + "\n")
    
    # Kullanıcıdan URL al
    url = input("YouTube video URL'sini girin: ").strip()
    
    if not url:
        print("❌ URL boş olamaz!")
        return
    
    # Ayarlar
    print("\n⚙️  Ayarlar:")
    extract_interval = input("Düzenli aralıklarla frame çıkarılsın mı? (E/H, varsayılan: E): ").strip().upper()
    extract_interval_frames = extract_interval != 'H'
    
    interval_seconds = 30
    if extract_interval_frames:
        interval_input = input("Frame çıkarma aralığı (saniye, varsayılan: 30): ").strip()
        if interval_input.isdigit():
            interval_seconds = int(interval_input)
    
    # Analizi başlat
    try:
        analyzer = YouTubeVideoAnalyzer(url)
        analyzer.analyze(
            extract_interval_frames=extract_interval_frames,
            interval_seconds=interval_seconds
        )
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
