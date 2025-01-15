import gradio as gr
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from transformers import pipeline

# Global değişkenler
frames = []
transcriptions = []
vectors = None
vectorizer = None

def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")
    return "temp_audio.wav"

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    transcriptions = []
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            transcriptions.append((0, text))  # Basitlik için sadece başlangıç zamanını 0 olarak alıyoruz
        except sr.UnknownValueError:
            print("Google Speech Recognition anlamadı")
        except sr.RequestError as e:
            print(f"Google Speech Recognition servis hatası; {e}")
    
    return transcriptions

def vectorize_text(transcriptions):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([t[1] for t in transcriptions])
    return vectors, vectorizer

def find_similar_moment(query, vectors, vectorizer, transcriptions):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)
    most_similar_idx = similarities.argmax()
    return transcriptions[most_similar_idx][0], most_similar_idx

def generate_response(query, context):
    generator = pipeline('text-generation', model='gpt2')
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return response.split("Answer:")[1].strip()

def process_video(video_path):
    global frames, transcriptions, vectors, vectorizer
    
    if not os.path.exists(video_path):
        return "Hata: Belirtilen video dosyası bulunamadı."
    
    frames = extract_frames(video_path)
    audio_path = extract_audio(video_path)
    transcriptions = transcribe_audio(audio_path)
    vectors, vectorizer = vectorize_text(transcriptions)
    
    os.remove(audio_path)
    return "Video işlendi ve hazır!"

def chat_with_video(query):
    global vectors, vectorizer, transcriptions, frames
    
    if vectors is None or vectorizer is None or not transcriptions or not frames:
        return "Lütfen önce bir video işleyin.", None
    
    timestamp, frame_idx = find_similar_moment(query, vectors, vectorizer, transcriptions)
    context = transcriptions[frame_idx][1]
    response = generate_response(query, context)
    return response, frames[frame_idx]

# Gradio arayüzü
with gr.Blocks() as demo:
    gr.Markdown("# Video Sohbet Botu")
    
    with gr.Tab("Video İşle"):
        video_path_input = gr.Textbox(label="Video Dosya Yolu")
        process_button = gr.Button("Videoyu İşle")
        output_text = gr.Textbox()
        
    with gr.Tab("Sohbet Et"):
        query_input = gr.Textbox(label="Sorunuzu girin")
        chat_button = gr.Button("Sor")
        response_output = gr.Textbox(label="Cevap")
        frame_output = gr.Image(label="İlgili Kare")
    
    process_button.click(process_video, inputs=video_path_input, outputs=output_text)
    chat_button.click(chat_with_video, inputs=query_input, outputs=[response_output, frame_output])

demo.launch()