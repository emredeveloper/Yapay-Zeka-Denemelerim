# 🎥 YouTube Video Analyzer + 🤖 AI

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║  🚀 AI-Powered YouTube Video Analysis                            ║
║  💬 Q&A | 🔍 Smart Search | 🖼️ Visual Analysis                  ║
║  ⚡ 70% Faster | 🔒 100% Local & Secure                        ║
╚══════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **🆕 New!** Now with AI-powered video analysis, smart search, and Q&A features!

A comprehensive Python tool that extracts transcripts, visual frames, and statistical information from YouTube videos **+ Local AI integration**.

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd Video-Analysis

# 2. Install Python packages
pip install -r requirements.txt

# 3. Install Ollama and download models
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b

# 4. Start the Flask application
python app_flask.py

# 5. Open in browser
# http://localhost:5000
```

**First use:** Enter video URL → Analyze → **AI Video Q&A** button → Enjoy! 🎉

## 🚀 What Can You Do?

- 🎬 Automatically analyze YouTube videos
- 💬 **Chat with AI about video content** (Turkish!)
- 🔍 **Smart search**: "Show me code", "Is there a graph?" queries
- 🖼️ **Perform visual analysis on each frame with AI**
- ⏯️ **Click on search results, video starts from that moment!**
- 📊 Detailed statistics and reports
- 🌐 Modern web interface (Flask)

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask 3.0+
- **AI/LLM**: Ollama (Local)
  - `granite4:tiny-h` → Text Q&A (2B parameters)
  - `qwen2.5vl:3b` → Visual Analysis (Vision-Language)
- **Video Processing**: OpenCV, PyTube
- **Transkript**: youtube-transcript-api
- **NLP**: NLTK
- **Frontend**: Bootstrap 5, jQuery
- **API**: YouTube IFrame API

## 🌟 Features

### 🤖 **NEW!** AI Features (Local Ollama)
- **🎯 Smart Video Question-Answer**: Ask questions in natural language about video content, AI will answer you!
  - � Real-time chat interface
  - 🧠 Automatic context extraction from video transcript
  - 🇹🇷 Multi-language support optimized
  - ⚡ Fast response time (granite4:tiny-h model)
  
- **🔍 Advanced Smart Search**: 3 different search modes!
  - �📝 **Text Search**: Keyword search within transcript
  - 🖼️ **Visual Search**: "Show me code", "Is there a graph?" queries
  - 🎨 **Hibrit Search**: Text + Visual combination (Optimized!)
  - ⏱️ Search time reduced from 60 seconds to 10-30 seconds
  - 🎯 Smart frame sampling strategy
  
- **👁️ Frame Visual Analysis**: Analyze each frame with AI
  - 🔬 Frame analysis with one click
  - 🎨 Visual content detection (qwen2.5vl:3b vision model)
  - 💡 Ability to ask special questions ("What's in this image?")
  
- **🎬 Smart Video Navigation**:
  - 🖱️ Click on search results, video opens automatically at that moment!
  - ⏯️ YouTube iframe integration
  - ⚡ Autoplay with no time wasted
  - 🎯 Direct timestamp detection from frames

### 📝 Text Analysis
- **Full Transcript**: Extracts all speech text from the video
- **Timestamped Transcript**: Shows time for each sentence
- **Sentence Parsing**: Splits text into sentences
- **Multi-language Support**: Turkish and English transcript support

### 🖼️ Visual Analysis
- **Sentence End Frames**: Automatically extracts frames at the end of each sentence or question (. ? !)
- **Regular Interval Frames**: Extracts frames at specified second intervals
- **Organized Folder Structure**: All images stored in separate folders
- **🆕 AI Frame Analysis**: Ability to analyze each frame with vision model

### 📊 Video Statistics
- Video title, channel name
- View count
- Video duration
- Publication date
- Word and character count
- Frame statistics

### 🌐 Web Interface (Flask)
- **Responsive Design**: Modern interface with Bootstrap 5
- **Real-time Chat**: Interactive conversation with AI
- **Frame Gallery**: Visual browsing of all frames
- **Smart Search Panel**: Text, visual and hybrid search options
- **Embedded Video Player**: Watch YouTube videos directly in the interface
- **Toast Notifications**: User-friendly feedback

## 📦 Installation

### 1️⃣ Install Python Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install flask opencv-python youtube-transcript-api pytube nltk requests
```

### 2️⃣ Install Ollama (For AI Features)

**Windows:**
1. Download from [Ollama.com](https://ollama.com/download)
2. Install and it will start automatically

**Download Models:**
```bash
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b
```

### 3️⃣ Start Flask Web Interface

```bash
python app_flask.py
```

Open in browser: `http://localhost:5000`

## 🚀 Usage

### 🌐 Web Interface (Recommended)

1. Start the Flask application: `python app_flask.py`
2. Go to `http://localhost:5000` in browser
3. Enter YouTube URL and configure settings
4. Let the video be analyzed!
5. **🆕 AI Video Q&A** button to:
   - Ask questions about the video 💬
   - Perform smart search 🔍
   - Analyze frames with AI 🖼️
   - Click on search results to watch video from that moment ▶️

### 💻 Command Line Usage

```bash
python youtube-app.py
```

The program will ask you for:
1. YouTube video URL
2. Extract frames at regular intervals? (Y/N)
3. Frame extraction interval (seconds)

### Example

```
Enter YouTube video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ

⚙️  Settings:
Extract frames at regular intervals? (Y/N, default: Y): Y
Frame extraction interval (seconds, default: 30): 20
```

## 📁 Output Structure

```
youtube_analysis/
└── {video_id}_{timestamp}/
    ├── images/
    │   ├── sentence_0000_time_00m15s.jpg
    │   ├── sentence_0001_time_00m32s.jpg
    │   ├── frame_0000_time_00m00s.jpg
    │   └── frame_0001_time_00m20s.jpg
    ├── text/
    │   ├── video_info.txt
    │   ├── full_transcript.txt
    │   ├── timed_transcript.txt
    │   ├── sentences.txt
    │   ├── sentence_frames_info.txt
    │   └── analysis_data.json
    ├── video.mp4
    └── SUMMARY_REPORT.txt
```

### File Descriptions

#### 📂 images/ folder
- `sentence_XXXX_*.jpg`: Frames extracted at sentence endings
- `frame_XXXX_*.jpg`: Frames extracted at regular intervals

#### 📂 text/ folder
- `video_info.txt`: General information about the video
- `full_transcript.txt`: Full transcript text
- `timed_transcript.txt`: Timestamped transcript
- `sentences.txt`: Text divided into sentences
- `sentence_frames_info.txt`: Information about each sentence frame
- `analysis_data.json`: All data in JSON format

#### 📄 Other files
- `video.mp4`: Downloaded video file
- `SUMMARY_REPORT.txt`: General summary report

## 💡 Programmatic Usage

```python
from youtube_app import YouTubeVideoAnalyzer

# Create analyzer
analyzer = YouTubeVideoAnalyzer(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_base_dir="my_analysis"
)

# Run analysis
analyzer.analyze(
    extract_interval_frames=True,  # Extract frames at regular intervals
    interval_seconds=30             # Frame every 30 seconds
)
```

## 🔧 Customization

### Change Frame Extraction Interval

```python
analyzer.analyze(interval_seconds=15)  # Every 15 seconds
```

### Sentence Frames Only

```python
analyzer.analyze(extract_interval_frames=False)  # Disable regular frames
```

### Different Output Directory

```python
analyzer = YouTubeVideoAnalyzer(
    url="your_url",
    output_base_dir="custom_output_folder"
)
```

## 📊 Sample Output

### Video Information
```
Title: Sample Video Title
Channel: Channel Name
Views: 1,234,567
Duration: 10:45
Publish Date: 2024-01-15
```

### Text Analysis
```
Total Word Count: 1,234
Total Character Count: 7,890
Sentence Count: 56
```

### Visual Analysis
```
Sentence End Frame Count: 56
Regular Interval Frame Count: 21
Total Frame Count: 77
```

## ⚠️ Important Notes

1. **Transcript Status**: Some videos may not have transcripts
2. **Video Download**: Some videos cannot be downloaded due to copyright
3. **Internet Connection**: Stable internet is required for video download
4. **Disk Space**: Long videos can take up a lot of space
5. **🆕 Ollama Requirements**: Ollama server must be running for AI features
   - Minimum 8GB RAM recommended
   - Much faster with GPU
6. **🆕 First Use**: Models are downloaded on first use (total ~4GB)

## 🐛 Error Solutions

### "Transcript not found" error
- Video may not have a transcript
- Video may be private or restricted

### "Video could not be downloaded" error
- Check your internet connection
- Video may be copyright protected
- Try a different video URL

### 🆕 "Cannot connect to Ollama" error
```bash
# Start Ollama service
ollama serve
```
Or make sure Ollama application is running on Windows.

### 🆕 "Model not found" error
```bash
# Download required models
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b
```

### 🆕 Slow Search Results
- **Solution 1**: Use "Text" or "Hybrid" search mode (faster)
- **Solution 2**: Test with shorter videos
- **Solution 3**: Ollama will automatically use GPU if available

### NLTK data error
Downloaded automatically on first use, to download manually:
```python
import nltk
nltk.download('punkt')
```

## 🎯 Performance Tips

### Increasing Search Speed
- **Hybrid Search**: Search text first, then analyze only relevant frames (60-70% faster!)
- **Text Search**: Search only within transcript (instant results)
- **Visual Search**: Optimized with 10 frame sampling (~20-30 seconds)

### Improving AI Response Quality
- Ask specific questions: Instead of "What's being discussed in the video?", ask "What technology is the video talking about?"
- Use frame analysis: Instead of "What's in this image?", ask "What programming language is the code in this image written in?"

## 📝 Notes

- Frame extraction process can take time depending on video length
- High-resolution videos take up more disk space
- Sentence detection is done with NLTK library (Turkish and English supported)
- **🆕 AI features run completely local** - your data doesn't go out!
- **🆕 First model download** may take a few minutes (only once)

## 🎬 Feature Demonstrations

### 💬 AI Video Q&A
```
👤 "What topics are discussed in this video?"
🤖 "The video covers artificial intelligence, machine learning, and deep learning topics..."

👤 "What's being discussed at 5 minutes?"
🤖 "At 5 minutes, neural network architectures and activation functions are explained in detail..."
```

### 🔍 Smart Search Examples
- **Text Search**: "machine learning" → List all mentioned moments
- **Visual Search**: "code image" → Find all moments with code on screen
- **Hybrid Search**: "python code" → Find moments with both "python" word and code visuals

### 🖼️ Frame Analysis
```
🖼️ Click on frame → AI Analyze
🤖 "This image shows a for loop and list comprehension example written in Python. 
    The code uses the enumerate() function..."
```

### ⏯️ Smart Video Navigation
```
🔍 Search → Click on result card → 🎬 Video automatically starts from that moment!
"Exactly the scene I wanted!" ✅
```

## 🌟 Why This Tool?

| Feature | Traditional Method | This Tool |
|---------|-------------------|---------|
| Video Search | Manual watching, note taking | AI-powered, automatic timestamps |
| Frame Extraction | Manual screenshots | Automatic, timestamped |
| Content Analysis | Watch and take notes | Ask AI, get direct answers |
| Visual Search | Impossible | Search "show graph"! |
| Privacy | Send data to cloud services | 100% local, secure |
| Speed | Hours | Minutes |

## 📊 Performance Comparison

**Previous Version vs New Version (AI-Featured)**

- ❌ **Old**: No frame analysis
- ✅ **New**: AI-powered visual content analysis

- ❌ **Old**: Manual transcript reading
- ✅ **New**: Ask "What's being discussed?", AI answers

- ❌ **Old**: No visual search
- ✅ **New**: Search "show graph", AI finds it

- ⏱️ **Search Duration**: 60-100 seconds → 10-30 seconds (**70% faster!**)

## 🤝 Contributing

You can send pull requests on GitHub for your suggestions and contributions.

### Development Ideas
- [ ] Batch video processing
- [ ] Video comparison feature
- [ ] Bookmark/favorite system
- [ ] Export Q&A history
- [ ] Real-time progress with WebSocket
- [ ] More LLM model support
- [ ] Video summarization feature

## 🎓 Usage Scenarios

### 🎯 Education & Learning
- Analyze lecture videos
- Quickly find specific topics: "Where is machine learning discussed in this video?"
- Extract code examples: "Show Python code examples"

### 📊 Content Analysis
- Analyze long interviews
- Identify main topics
- Bookmark important moments

### 🔬 Research
- Analyze technical presentations
- Extract graphs and diagrams: "Show graphs"
- Find moments where specific terms are mentioned

### 🎬 Content Creation
- Extract video scripts
- Use important frames for thumbnails
- Categorize video content

## 📄 License

This project is licensed under the MIT license.

---

<div align="center">

### 🌟 Did you like this project?

⭐ **Give a star** - Support the project's development!  
🐛 **Open an issue** - Report if you find errors  
🤝 **Send a PR** - Contribute  
💬 **Share** - Recommend to your friends  

---

**Producer:** [Emre Developer](https://github.com/emredeveloper)  
**Technology:** Python 🐍 | Flask 🌶️ | Ollama 🤖 | OpenCV 📹  
**Version:** 2.0 (AI-Powered) 🚀  

---

*"Analyze your videos smarter with AI!"* ✨

</div>
