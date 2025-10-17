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
    
    def generate_learning_content(self, text: str, sample_sentences: list = None) -> dict:
        """
        Generate all learning content in ONE API call (FAST!)
        
        Args:
            text: Full transcript text
            sample_sentences: List of sample sentence dicts with 'text' and 'start' keys
            
        Returns:
            Dictionary with vocabulary, sentences, grammar_patterns, and quiz_questions
        """
        # Limit text size
        text_sample = text[:3000]
        
        # Prepare sample sentences
        if sample_sentences:
            sentences_text = "\n".join([f"{i+1}. {s['text']}" for i, s in enumerate(sample_sentences[:5])])
        else:
            sentences_text = "No sentences provided"
        
        prompt = f"""You are an English teacher creating learning materials from a video transcript.

TRANSCRIPT:
{text_sample}

SAMPLE SENTENCES FROM VIDEO:
{sentences_text}

Create a comprehensive learning package with:
1. 15 important vocabulary words (intermediate level)
2. Analysis of the 5 sample sentences 
3. 5 grammar patterns found in the text
4. 8 quiz questions

Return ONLY valid JSON without any control characters or special formatting:

{{
  "vocabulary": [
    {{
      "word": "example",
      "pos": "noun",
      "definition": "a thing characteristic of its kind",
      "turkish": "ornek",
      "example": "sentence from text"
    }}
  ],
  "sentences": [
    {{
      "sentence": "original sentence",
      "turkish": "Turkish translation",
      "structure": "Subject + Verb + Object",
      "grammar_points": ["Present Simple"],
      "word_breakdown": [
        {{"word": "word1", "turkish": "kelime1", "role": "subject"}}
      ]
    }}
  ],
  "grammar_patterns": [
    {{
      "pattern": "Present Perfect Tense",
      "explanation": "Used for actions in the past with present relevance",
      "turkish_explanation": "Gecmiste yapilan ama simdiyle baglantisi olan eylemler icin",
      "example": "I have finished my homework",
      "usage": "Have/Has + past participle"
    }}
  ],
  "quiz_questions": [
    {{
      "type": "multiple_choice",
      "question": "What does example mean?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Explanation in English and Turkish"
    }}
  ]
}}

IMPORTANT: Return clean JSON only, no markdown, no code blocks, no newlines in strings."""

        try:
            print("🚀 Generating all learning content in ONE call...")
            response = self.generate_text(prompt)
            
            import json
            import re
            
            # Clean response - remove markdown code blocks
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```[a-z]*\n', '', cleaned)
                cleaned = re.sub(r'\n```$', '', cleaned)
            
            # Replace control characters
            cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Fix multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Additional cleanup
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                
                result = json.loads(json_str)
                
                # Ensure all keys exist
                if 'vocabulary' not in result:
                    result['vocabulary'] = []
                if 'sentences' not in result:
                    result['sentences'] = []
                if 'grammar_patterns' not in result:
                    result['grammar_patterns'] = []
                if 'quiz_questions' not in result:
                    result['quiz_questions'] = []
                
                print(f"✅ Generated: {len(result['vocabulary'])} words, {len(result['sentences'])} sentences, {len(result['grammar_patterns'])} patterns, {len(result['quiz_questions'])} questions")
                return result
            else:
                print(f"❌ Could not find JSON in response")
                return self._empty_learning_content()
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response preview: {response[:500]}")
            return self._empty_learning_content()
        except Exception as e:
            print(f"❌ Error generating learning content: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_learning_content()
    
    def _empty_learning_content(self) -> dict:
        """Return empty learning content structure"""
        return {
            'vocabulary': [],
            'sentences': [],
            'grammar_patterns': [],
            'quiz_questions': []
        }
    
    def extract_vocabulary(self, text: str, level: str = 'intermediate', max_words: int = 20) -> list:
        """
        Extract important vocabulary words from text with Turkish translations
        
        Args:
            text: Text to extract vocabulary from
            level: Difficulty level (beginner, intermediate, advanced)
            max_words: Maximum number of words to extract
            
        Returns:
            List of dictionaries with word, definition, turkish_translation, example
        """
        prompt = f"""Analyze this English text and extract the {max_words} most important vocabulary words for a {level} level learner.

Text: {text[:2000]}

For each word provide:
1. The word itself
2. Part of speech (noun, verb, adjective, etc.)
3. English definition (simple and clear)
4. Turkish translation
5. Example sentence from the text (if available)

Return ONLY a JSON array in this exact format:
[
  {{
    "word": "example",
    "pos": "noun",
    "definition": "something that is typical of its kind",
    "turkish": "örnek",
    "example": "This is an example sentence."
  }}
]

JSON array:"""

        try:
            response = self.generate_text(prompt)
            # Extract JSON from response
            import json
            import re
            
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                vocabulary = json.loads(json_match.group())
                return vocabulary[:max_words]
            else:
                print(f"Could not parse vocabulary JSON: {response[:200]}")
                return []
                
        except Exception as e:
            print(f"Error extracting vocabulary: {e}")
            return []
    
    def analyze_sentence(self, sentence: str) -> dict:
        """
        Analyze sentence structure with grammar breakdown and Turkish translation
        
        Args:
            sentence: English sentence to analyze
            
        Returns:
            Dictionary with sentence analysis
        """
        prompt = f"""Analyze this English sentence for language learners:

Sentence: {sentence}

Provide:
1. Turkish translation
2. Grammar structure (e.g., "Subject + Present Perfect + Object")
3. Key grammar points used (e.g., "Present Perfect Tense")
4. Word-by-word breakdown with Turkish meanings

Return ONLY a JSON object in this exact format:
{{
  "sentence": "{sentence}",
  "turkish": "Turkish translation here",
  "structure": "Grammar structure pattern",
  "grammar_points": ["Point 1", "Point 2"],
  "word_breakdown": [
    {{"word": "word1", "turkish": "kelime1", "role": "subject"}},
    {{"word": "word2", "turkish": "kelime2", "role": "verb"}}
  ]
}}

JSON object:"""

        try:
            response = self.generate_text(prompt)
            import json
            import re
            
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                print(f"Could not parse sentence analysis JSON")
                return {
                    "sentence": sentence,
                    "turkish": "Translation unavailable",
                    "structure": "Unknown",
                    "grammar_points": [],
                    "word_breakdown": []
                }
                
        except Exception as e:
            print(f"Error analyzing sentence: {e}")
            return {
                "sentence": sentence,
                "turkish": "Error occurred",
                "structure": "Unknown",
                "grammar_points": [],
                "word_breakdown": []
            }
    
    def extract_grammar_patterns(self, text: str) -> list:
        """
        Extract and explain grammar patterns from text
        
        Args:
            text: Text to analyze for grammar patterns
            
        Returns:
            List of grammar patterns with explanations
        """
        prompt = f"""Analyze this English text and identify the 5 most important grammar patterns for language learners.

Text: {text[:2000]}

For each pattern provide:
1. Grammar pattern name (e.g., "Present Perfect Tense")
2. Simple explanation in English
3. Turkish explanation
4. Example from the text
5. General usage rule

Return ONLY a JSON array in this exact format:
[
  {{
    "pattern": "Grammar pattern name",
    "explanation": "Simple explanation",
    "turkish_explanation": "Türkçe açıklama",
    "example": "Example sentence from text",
    "usage": "When and how to use this"
  }}
]

JSON array:"""

        try:
            response = self.generate_text(prompt)
            import json
            import re
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group())
                return patterns[:5]
            else:
                print(f"Could not parse grammar patterns JSON")
                return []
                
        except Exception as e:
            print(f"Error extracting grammar patterns: {e}")
            return []
    
    def generate_quiz(self, vocabulary: list, sentences: list, count: int = 10) -> list:
        """
        Generate quiz questions from vocabulary and sentences
        
        Args:
            vocabulary: List of vocabulary items
            sentences: List of example sentences
            count: Number of questions to generate
            
        Returns:
            List of quiz questions with answers
        """
        print(f"🎯 generate_quiz called with {len(vocabulary)} words, {len(sentences)} sentences, count={count}")
        
        # Prepare vocabulary summary
        vocab_text = "\n".join([f"- {v.get('word', '')}: {v.get('definition', '')}" 
                                for v in vocabulary[:10]])
        
        # Prepare sentence samples
        sentence_text = "\n".join([f"- {s}" for s in sentences[:5]])
        
        print(f"📝 Vocab text length: {len(vocab_text)}, Sentence text length: {len(sentence_text)}")
        
        prompt = f"""Create {count} English learning quiz questions based on this vocabulary and sentences.

Vocabulary:
{vocab_text}

Example Sentences:
{sentence_text}

Create diverse question types:
1. Multiple choice (meaning of word)
2. Fill in the blank
3. Choose correct word form
4. Turkish to English translation
5. Grammar questions

Return ONLY a JSON array in this exact format:
[
  {{
    "type": "multiple_choice",
    "question": "What does 'example' mean?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Brief explanation in English with Turkish (Türkçe açıklama)"
  }}
]

JSON array:"""

        try:
            print("🤖 Calling AI to generate quiz...")
            response = self.generate_text(prompt)
            print(f"✅ AI response received, length: {len(response)} characters")
            print(f"📄 Response preview: {response[:200]}...")
            
            import json
            import re
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"✅ Found JSON array, length: {len(json_str)} characters")
                questions = json.loads(json_str)
                print(f"✅ Parsed {len(questions)} questions")
                
                # Validate question structure
                valid_questions = []
                for i, q in enumerate(questions):
                    if all(key in q for key in ['question', 'options', 'correct_answer', 'explanation']):
                        valid_questions.append(q)
                    else:
                        print(f"⚠️  Question {i} missing required fields: {q.keys()}")
                
                print(f"✅ Returning {len(valid_questions)} valid questions")
                return valid_questions[:count]
            else:
                print(f"❌ Could not find JSON array in response")
                print(f"Full response: {response}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Failed to parse: {json_str if 'json_str' in locals() else 'No JSON found'}")
            return []
        except Exception as e:
            print(f"❌ Error generating quiz: {e}")
            import traceback
            traceback.print_exc()
            return []


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
