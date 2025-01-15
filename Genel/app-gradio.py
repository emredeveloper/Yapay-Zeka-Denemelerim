import gradio as gr
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "AIzaSyARLSyuSqD79Lvct7gc203X7RvXxa3CuHo"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_video_id(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:shorts\/)?([^\s&]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_video_details(video_id):
    try:
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()
        if 'items' in response:
            video = response['items'][0]
            return {
                'id': video_id,
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'tags': video['snippet'].get('tags', []),
                'view_count': int(video['statistics']['viewCount']),
                'like_count': int(video['statistics'].get('likeCount', '0')),
                'comment_count': int(video['statistics'].get('commentCount', '0')),
                'published_at': video['snippet']['publishedAt'],
                'channel_title': video['snippet']['channelTitle'],
                'thumbnail': video['snippet']['thumbnails']['default']['url']
            }
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
    return None

def search_similar_shorts(video_details, max_results=5, lang=None):
    query = f"{video_details['title']} {' '.join(video_details['tags'])}"
    try:
        request = youtube.search().list(
            q=query,
            type="video",
            videoDuration="short",
            part="id,snippet",
            maxResults=max_results,
            relevanceLanguage=lang
        )
        response = request.execute()
        similar_videos = []
        for item in response['items']:
            video_id = item['id']['videoId']
            details = get_video_details(video_id)
            if details:
                similar_videos.append(details)
        return similar_videos
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
    return []

def calculate_similarity(original_text, similar_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_text] + similar_texts)
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

def recommend_shorts(url, lang, sort_by):
    video_id = extract_video_id(url)
    if not video_id:
        return "Geçersiz YouTube Shorts URL'si"

    video_details = get_video_details(video_id)
    if not video_details:
        return "Video detayları alınamadı"

    similar_videos = search_similar_shorts(video_details, lang=lang)
    if not similar_videos:
        return "Benzer videolar bulunamadı"

    original_text = f"{video_details['title']} {video_details['description']} {' '.join(video_details['tags'])}"
    similar_texts = [f"{v['title']} {v['description']} {' '.join(v['tags'])}" for v in similar_videos]
    similarities = calculate_similarity(original_text, similar_texts)

    for video, similarity in zip(similar_videos, similarities):
        video['similarity'] = similarity
        video['engagement_rate'] = (video['like_count'] + video['comment_count']) / video['view_count'] if video['view_count'] > 0 else 0

    if sort_by == "views":
        similar_videos.sort(key=lambda x: x['view_count'], reverse=True)
    elif sort_by == "likes":
        similar_videos.sort(key=lambda x: x['like_count'], reverse=True)
    elif sort_by == "engagement":
        similar_videos.sort(key=lambda x: x['engagement_rate'], reverse=True)
    else:  # default: relevance (similarity)
        similar_videos.sort(key=lambda x: x['similarity'], reverse=True)

    result = f"Orijinal Video:\nBaşlık: {video_details['title']}\n"
    result += f"Kanal: {video_details['channel_title']}\n"
    result += f"Yayın Tarihi: {video_details['published_at']}\n"
    result += f"Görüntülenme: {video_details['view_count']}\n"
    result += f"Beğeni: {video_details['like_count']}\n"
    result += f"Yorum: {video_details['comment_count']}\n"
    result += f"Etiketler: {', '.join(video_details['tags'])}\n\n"
    result += "Önerilen Benzer Shorts:\n\n"

    for video in similar_videos:
        result += f"Başlık: {video['title']}\n"
        result += f"Kanal: {video['channel_title']}\n"
        result += f"Yayın Tarihi: {video['published_at']}\n"
        result += f"Görüntülenme: {video['view_count']}\n"
        result += f"Beğeni: {video['like_count']}\n"
        result += f"Yorum: {video['comment_count']}\n"
        result += f"Benzerlik Skoru: {video['similarity']:.4f}\n"
        result += f"Etkileşim Oranı: {video['engagement_rate']:.4f}\n"
        result += f"URL: https://www.youtube.com/shorts/{video['id']}\n\n"

    return result

iface = gr.Interface(
    fn=recommend_shorts,
    inputs=[
        gr.Textbox(label="YouTube Shorts URL"),
        gr.Dropdown(choices=["en", "tr", "es", "fr", "de"], label="Dil (isteğe bağlı)"),
        gr.Radio(["relevance", "views", "likes", "engagement"], label="Sıralama Kriteri", value="relevance")
    ],
    outputs=gr.Textbox(label="Öneriler"),
    title="Gelişmiş YouTube Shorts Öneri Sistemi",
    description="Bir YouTube Shorts URL'si girin ve benzer içeriklere sahip diğer Shorts videolarını alın. İsteğe bağlı olarak dil ve sıralama kriteri seçebilirsiniz."
)

iface.launch()