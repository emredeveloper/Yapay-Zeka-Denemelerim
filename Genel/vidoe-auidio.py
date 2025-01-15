import csv
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os
from pydub import AudioSegment
import math
from tqdm import tqdm

def extract_audio_and_transcribe(video_path, output_csv):
    # Videoyu yükle
    video = VideoFileClip(video_path)
    
    # Geçici ses dosyası oluştur
    temp_audio = "temp_audio.wav"
    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
    
    # Speech recognition için recognizer oluştur
    recognizer = sr.Recognizer()
    
    # Ses dosyasını yükle
    audio = AudioSegment.from_wav(temp_audio)
    
    # Sesi metne çevir ve zaman damgalarını kaydet
    timestamps = []
    duration = int(video.duration)
    chunk_size = 3000  # Her 5 saniyelik bölümü işle (milisaniye cinsinden)
    overlap = 1500  # 2 saniyelik örtüşme
    
    # İlerleme çubuğu oluştur
    progress_bar = tqdm(total=duration, desc="Video işleniyor", unit="saniye")
    
    for start_time in range(0, duration * 1000, chunk_size - overlap):
        end_time = min(start_time + chunk_size, duration * 1000)
        chunk = audio[start_time:end_time]
        
        # Geçici chunk dosyası oluştur
        chunk.export("temp_chunk.wav", format="wav")
        
        with sr.AudioFile("temp_chunk.wav") as source:
            audio_chunk = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_chunk, language="tr-TR")
            
            # Zaman damgasını hesapla
            time_seconds = math.floor(start_time / 1000)
            minutes = time_seconds // 60
            seconds = time_seconds % 60
            time_stamp = f"{minutes}:{seconds:02d}"
            
            # Metni kaydet
            if text.strip():
                timestamps.append((time_stamp, text.strip()))
                print(f"Tanınan metin ({time_stamp}): {text.strip()}")
        
        except sr.UnknownValueError:
            print(f"Uyarı: {time_stamp} için metin tanınamadı.")
        except sr.RequestError as e:
            print(f"Google Speech Recognition hatası; {e}")
        
        # İlerleme çubuğunu güncelle
        progress_bar.update((chunk_size - overlap) // 1000)
    
    progress_bar.close()
    
    # CSV dosyasına kaydet
    if timestamps:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dakika:Saniye", "Metin"])
            writer.writerows(timestamps)
        print(f"İşlem tamamlandı. Sonuçlar {output_csv} dosyasına kaydedildi.")
    else:
        print("Uyarı: Hiç metin tanınamadı. CSV dosyası oluşturulmadı.")
    
    # Geçici dosyaları sil ve kaynakları serbest bırak
    os.remove(temp_audio)
    os.remove("temp_chunk.wav")
    video.close()

# Kullanım örneği
video_path = "apple.mp4"  # Video dosyanızın adını buraya yazın
output_csv = "ses_metinleri_ve_zaman_damgalari.csv"
extract_audio_and_transcribe(video_path, output_csv)