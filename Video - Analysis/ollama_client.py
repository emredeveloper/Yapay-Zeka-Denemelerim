"""
Ollama API integration
- granite3.1:2b: Text-based Q&A and analysis
- qwen2.5-vl:3b: Visual analysis and description
"""

import requests
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Union


class OllamaClient:
    """Ollama API client"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.text_model = "granite4:tiny-h"  # User's suggested model
        self.vision_model = "qwen2.5vl:3b"   # Mevcut vision model
    
    def check_connection(self) -> bool:
        """Check Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List installed models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def generate_text(self, 
                     prompt: str, 
                     system: Optional[str] = None,
                     context: Optional[str] = None,
                     stream: bool = False) -> Union[str, Dict]:
        """Text generation (granite3.1:2b)"""
        try:
            # Create prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
            
            payload = {
                "model": self.text_model,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                if stream:
                    return response
                else:
                    result = response.json()
                    return result.get('response', '')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_image(self, 
                     image_path: str, 
                     question: str = "What do you see in this image? Explain in detail.",
                     stream: bool = False) -> Union[str, Dict]:
        """Image analysis (qwen2.5-vl:3b)"""
        try:
            # Encode image
            image_b64 = self._encode_image(image_path)
            
            payload = {
                "model": self.vision_model,
                "prompt": question,
                "images": [image_b64],
                "stream": stream,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                if stream:
                    return response
                else:
                    result = response.json()
                    return result.get('response', '')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_multiple_images(self,
                               image_paths: List[str],
                               question: str) -> List[Dict[str, str]]:
        """Multiple image analysis"""
        results = []
        for img_path in image_paths:
            analysis = self.analyze_image(img_path, question)
            results.append({
                'image': img_path,
                'analysis': analysis
            })
        return results
    
    def answer_question_with_context(self,
                                    question: str,
                                    transcript: str,
                                    video_info: Dict,
                                    relevant_frames: Optional[List[str]] = None) -> str:
        """Answer questions in video context"""
        
        # System prompt - better English
        system = """You are a video analysis assistant. You answer users' questions based on the provided video transcript and video information.
        
        Rules:
        1. Use only the provided transcript and information
        2. Give clear, understandable answers in English
        3. If information is not in transcript, say clearly
        4. If possible, specify timestamps (e.g. "at 5 minutes...")
        5. Give short and concise answers, don't go into unnecessary detail"""
        
        # Clean and limit transcript
        clean_transcript = transcript.strip()
        if len(clean_transcript) > 8000:
            clean_transcript = clean_transcript[:8000] + "... (transkript devam ediyor)"
        
        # Context preparation - more structured
        context = f"""=== VIDEO INFORMATION ===
Title: {video_info.get('title', 'Unknown')}
Channel: {video_info.get('channel', 'Unknown')}
Duration: {video_info.get('duration', 'Unknown')}
Views: {video_info.get('views', 'Unknown')}

=== VIDEO TRANSCRIPT ===
{clean_transcript}
"""
        
        # If there are images, analyze them too
        if relevant_frames and len(relevant_frames) > 0:
            context += "\n\n=== VISUAL ANALYSES ===\n"
            for i, frame_path in enumerate(relevant_frames[:2], 1):  # First 2 frames
                try:
                    img_analysis = self.analyze_image(
                        frame_path, 
                        "What does this video frame show? Explain briefly (max 2 sentences)."
                    )
                    context += f"Image {i}: {img_analysis}\n"
                except Exception as e:
                    context += f"Image {i}: Could not be analyzed\n"
        
        return self.generate_text(question, system=system, context=context)
    
    def semantic_search(self,
                       query: str,
                       transcript_segments: List[Dict],
                       top_k: int = 5) -> List[Dict]:
        """Semantic search (simple version)"""
        # Find relevant segments using LLM
        segments_text = "\n".join([
            f"[{i}] {seg['text']} (Time: {seg['start']:.1f}s)"
            for i, seg in enumerate(transcript_segments[:100])  # First 100 segments
        ])
        
        prompt = f"""
Find the {top_k} most relevant segments from the transcript segments below for "{query}".
Write only the segment numbers separated by commas (example: 1,5,12,20,45).

Segments:
{segments_text}

Most relevant {top_k} segment numbers:"""
        
        try:
            response = self.generate_text(prompt)
            # Remove numbers
            numbers = [int(n.strip()) for n in response.split(',') if n.strip().isdigit()]
            
            results = []
            for idx in numbers[:top_k]:
                if idx < len(transcript_segments):
                    results.append(transcript_segments[idx])
            
            return results
        except:
            # Basit text matching fallback
            results = []
            query_lower = query.lower()
            for seg in transcript_segments:
                if query_lower in seg['text'].lower():
                    results.append(seg)
                    if len(results) >= top_k:
                        break
            return results


# Test fonksiyonu
def test_ollama():
    """Test Ollama connection"""
    client = OllamaClient()
    
    print("🔍 Ollama Connection Test")
    print("="*50)
    
    if client.check_connection():
        print("✅ Successfully connected to Ollama!")
        
        models = client.list_models()
        print(f"\n📦 Installed Models ({len(models)}):")
        for model in models:
            print(f"  - {model}")
        
        # Model check
        if client.text_model in models:
            print(f"\n✅ Text model available: {client.text_model}")
        else:
            print(f"\n⚠️  Text model missing: {client.text_model}")
            print(f"   To download: ollama pull {client.text_model}")
        
        if client.vision_model in models:
            print(f"✅ Vision model available: {client.vision_model}")
        else:
            print(f"⚠️  Vision model missing: {client.vision_model}")
            print(f"   To download: ollama pull {client.vision_model}")
        
        # Simple test
        if client.text_model in models:
            print("\n🧪 Test Question...")
            response = client.generate_text("Hello! How are you?")
            print(f"📝 Response: {response[:100]}...")
        
    else:
        print("❌ Could not connect to Ollama!")
        print("   Is Ollama running? Check: ollama list")
    
    print("="*50)


if __name__ == "__main__":
    test_ollama()
