# Artificial Intelligence Trials & Experiments

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

**A comprehensive collection of AI/ML experiments, tools, and applications**

[Features](#-features) • [Projects](#-project-directories) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📖 Overview

This repository contains a diverse collection of artificial intelligence, machine learning, and data science projects. Each project demonstrates different aspects of modern AI technology, from Large Language Models (LLMs) and knowledge distillation to geospatial analysis and video processing.

The projects range from experimental research code to full-featured web applications, all built with Python and modern AI frameworks.

## ✨ Features

- 🤖 **Large Language Models (LLMs)**: Knowledge distillation, model training, and agentic flows
- 🗺️ **Geospatial Analysis**: Interactive mapping and geographic data visualization
- 🎥 **Video Analysis**: AI-powered YouTube video analysis with transcript extraction
- 📊 **Machine Learning**: Regression models, statistics, and data analysis
- 🔍 **RAG & Embeddings**: Retrieval-Augmented Generation and document embeddings
- 🎮 **Interactive Applications**: Web-based interfaces with Flask and Gradio
- 🧪 **Experimental Research**: Cutting-edge AI techniques and methodologies

## 📁 Project Directories

### 🤖 [LLM](./LLM/) - Large Language Model Experiments

**Purpose**: Knowledge distillation techniques and LLM training implementations.

**Key Features**:
- **Soft-Label Distillation**: Full probability distribution transfer from teacher to student models
- **Hard-Label Distillation**: Memory-efficient one-hot encoding approach (DeepSeek-R1 style)
- **Co-Distillation**: Simultaneous training of teacher and student models (Llama 4 style)
- Quiz generation with AI agents
- JSON-based LLM interactions
- Vision-Language Model (VLM) experiments with LM Studio

**Technologies**: Hugging Face Transformers, PyTorch, TensorBoard, LM Studio

**Files**:
- `soft_label_distillation.py` - Soft-label knowledge distillation
- `hard_label_distillation.py` - Hard-label knowledge distillation
- `co_distillation.py` - Co-distillation implementation
- `run_all_distillation.py` - Run all distillation methods
- `quiz_agents.py` - AI-powered quiz generation
- `turkce-quiz-uretici.py` - Turkish quiz generator
- `json-with-gemini.py` - Gemini API JSON interactions
- `lmstudio-vlm.ipynb` - Vision-Language Model experiments

---

### 🗺️ [Geospatial Analysis](./Geospatial%20Analysis/) - Geographic Data Visualization

**Purpose**: Interactive mapping and geospatial analysis of Turkish cities.

**Key Features**:
- **Interactive Maps**: Folium-based interactive map visualization
- **City Data**: Population, area, region, plate code, and elevation data for 20+ Turkish cities
- **Distance Calculation**: Haversine formula for calculating distances between cities
- **AI Agent Integration**: LM Studio with tool-based agentic flows
- **Visual Analysis**: Vision-Language Model (VLM) for map image analysis
- **Wikipedia Integration**: City information from Wikipedia API
- **Map Styles**: OpenStreetMap, Satellite, Terrain, and Dark themes
- **Statistics & Analytics**: Detailed demographic and geographic statistics

**Technologies**: Flask, Folium, GeoPandas, LM Studio, Matplotlib, PIL

**Files**:
- `app_flask.py` - Main Flask web application
- `app_gradio.py` - Gradio-based interface (legacy)
- `app.py` - Initial geospatial analysis script
- `requirements.txt` - Python dependencies

**API Endpoints**:
- `POST /api/create_map` - Create interactive map
- `POST /api/agent/chat` - AI agent chat interface
- `POST /api/agent/analyze_image` - Visual map analysis
- `GET /api/city_info/<city_name>` - City information
- `POST /api/statistics` - Geographic statistics

---

### 🎥 [Video - Analysis](./Video%20-%20Analysis/) - YouTube Video Analyzer

**Purpose**: AI-powered YouTube video analysis with transcript extraction and visual frame analysis.

**Key Features**:
- **Video Analysis**: Automatic transcript extraction and frame analysis
- **AI Q&A**: Chat with AI about video content using local Ollama models
- **Smart Search**: Text, visual, and hybrid search modes
- **Frame Extraction**: Automatic frame extraction at sentence endings and regular intervals
- **Visual Analysis**: Vision-Language Model for frame content analysis
- **Smart Navigation**: Click on search results to jump to specific video moments
- **Statistics**: Detailed video statistics and reports

**Technologies**: Flask, Ollama, OpenCV, PyTube, NLTK, YouTube Transcript API

**Models Used**:
- `granite4:tiny-h` - Text Q&A (2B parameters)
- `qwen2.5vl:3b` - Visual analysis (Vision-Language)

**Files**:
- `app_flask.py` - Flask web application
- `youtube_app.py` - Core video analysis class
- `ollama_client.py` - Ollama API client
- `requirements.txt` - Python dependencies

**Features**:
- 70% faster search compared to previous versions
- 100% local processing (no cloud services)
- Real-time chat interface
- Frame gallery with AI analysis
- Smart video navigation

---

### 📊 [ML](./ML/) - Machine Learning & Statistics

**Purpose**: Machine learning models, statistical analysis, and data processing.

**Key Features**:
- **Regression Models**: Multiple regression model implementations
- **Statistical Analysis**: Advanced statistical methods and optimizations
- **Data Processing**: Efficient data processing with Numba
- **Feature Importance**: Model feature importance analysis
- **Random Sparse Projection**: Dimensionality reduction techniques
- **Top-K Accuracy**: Custom accuracy scoring metrics

**Technologies**: scikit-learn, NumPy, Pandas, Numba, Matplotlib

**Files**:
- `app.py` - Machine learning web application
- `7regressionmodel.ipynb` - Multiple regression models
- `numba_ml.py` - Numba-accelerated ML operations
- `PPscore.ipynb` - Predictive Power Score analysis
- `Random Sparse Projection.ipynb` - Sparse projection techniques
- `Top-K-Accuracy-Score.ipynb` - Custom accuracy metrics
- `ydflearn.ipynb` - YDF (Yggdrasil Decision Forests) learning

---

### 🔍 [Embedding - Doc - RAG](./Embedding%20-%20Doc%20-%20RAG/) - Retrieval-Augmented Generation

**Purpose**: Document embeddings, RAG (Retrieval-Augmented Generation), and semantic search.

**Key Features**:
- **ModernBERT**: Modern BERT-based embeddings
- **Document Embeddings**: Text embedding generation
- **RAG Implementation**: Retrieval-Augmented Generation systems
- **Semantic Search**: Vector-based semantic search

**Technologies**: Transformers, ChromaDB, Sentence-Transformers

**Files**:
- `modernbert-app.py` - ModernBERT application

---

### 📈 [Graph](./Graph/) - Graph RAG

**Purpose**: Graph-based Retrieval-Augmented Generation.

**Key Features**:
- **Graph RAG**: Knowledge graph construction and traversal
- **Improved Graph RAG**: Enhanced graph-based RAG implementation

**Files**:
- `graph_rag.py` - Basic graph RAG implementation
- `run_improved_graph_rag.py` - Improved graph RAG runner

---

### 🧪 [Trials](./Trials/) - Experimental Research

**Purpose**: Cutting-edge AI research and experimental implementations.

**Key Features**:
- **Active Learning**: Active learning with Gemma 3
- **Code Analysis**: Gemini 2.5 Pro code analysis
- **Graph Agents**: Graph-based AI agents
- **Knowledge Distillation**: Various distillation techniques
- **Anomaly Detection**: PyOD-based anomaly detection
- **Semantic Search**: Semantic frequency and BM25 hybrid search
- **RL RAG**: Reinforcement Learning for RAG

**Files**:
- `Active_Learning_Gemma_3.ipynb` - Active learning experiments
- `gemini2-5-pro-code-analysis.py` - Code analysis with Gemini
- `graph_agent.py` - Graph-based agents
- `knowledge_distil.ipynb` - Knowledge distillation research
- `anomaly_Detect_wtih_pyod.py` - Anomaly detection
- `semantic_frekansbm25.ipynb` - Hybrid search techniques
- `simple_rag_rl.py` - RL-based RAG

---

### 📝 [Data Sentetic](./Data%20Sentetic/) - Synthetic Data Generation

**Purpose**: Synthetic data generation and data augmentation.

**Key Features**:
- **Synthetic Data**: Generate synthetic datasets
- **DataCamp App**: DataCamp-style data generation
- **HG App**: Historical data generation

**Files**:
- `datacamp_app.ipynb` - DataCamp-style data generation
- `hg_app.ipynb` - Historical data generation

---

### 🔤 [Spacy](./Spacy/) - NLP with spaCy

**Purpose**: Natural Language Processing with spaCy.

**Key Features**:
- **Document Processing**: Text processing and analysis
- **NLP Pipelines**: spaCy NLP pipelines

**Files**:
- `spacydoc.ipynb` - spaCy document processing

---

### ⏱️ [Time Series](./Time%20Series/) - Time Series Analysis

**Purpose**: Time series forecasting and analysis.

**Key Features**:
- **Time Series Forecasting**: Transformer-based time series models
- **Amazon Forecast**: Amazon forecasting techniques
- **Transformer Models**: Time series transformers

**Files**:
- `transformer_timeseries.py` - Transformer-based time series
- `amazonfor.ipynb` - Amazon forecasting

---

### 🎮 [XOX LLM](./XOX%20LLM/) - Tic-Tac-Toe with LLM

**Purpose**: Interactive tic-tac-toe game with LLM integration.

**Key Features**:
- **Game Interface**: Web-based tic-tac-toe game
- **LLM Integration**: AI opponent with LLM

**Files**:
- `xox_flask.py` - Flask-based tic-tac-toe game
- `templates/index.html` - Game interface
- `static/style.css` - Game styles

---

### 📦 [Json](./Json/) - JSON Data Processing

**Purpose**: JSON-based data processing and ML applications.

**Key Features**:
- **JSON Format AI**: AI-powered JSON formatting
- **ML JSON Ollama**: Machine learning with JSON and Ollama
- **Data Processing**: CSV and JSON data processing

**Files**:
- `json-format-ai.py` - AI JSON formatter
- `ml-json-ollama.py` - ML with JSON and Ollama

---

### 🔧 [Genel](./Genel/) - General Utilities

**Purpose**: General-purpose utilities and experimental code.

**Key Features**:
- **Gradio Apps**: Various Gradio-based applications
- **CSV Processing**: CSV data processing
- **Graph RAG**: Graph-based RAG implementations
- **Transformer API**: Transformer model APIs
- **Video Processing**: Video and audio processing

**Files**:
- `app-gradio.py` - Gradio applications
- `csv-app.py` - CSV processing
- `graphrag.py` - Graph RAG
- `transformer-api.py` - Transformer API
- `vidoe-auidio.py` - Video/audio processing

---

### 🗄️ [Datasets](./Datasets/) - Sample Datasets

**Purpose**: Sample datasets for testing and development.

**Datasets**:
- `hotel_bookings.csv` - Hotel booking data
- `ner_dataset.csv` - Named Entity Recognition dataset
- `netflix_reviews.csv` - Netflix reviews
- `ufos.csv` - UFO sightings data

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Basic Setup

1. **Clone the repository**:
```bash
git clone https://github.com/emredeveloper/Artifical-Intelligence-Trials-Notebooks.git
cd Artifical-Intelligence-Trials-Notebooks
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies for specific project**:
Each project has its own `requirements.txt` file. Install dependencies as needed:

```bash
# For Geospatial Analysis
cd "Geospatial Analysis"
pip install -r requirements.txt

# For Video Analysis
cd "../Video - Analysis"
pip install -r requirements.txt

# For LLM
cd "../LLM"
pip install -r requirements.txt
```

### Optional: Install AI Models

#### LM Studio (for Geospatial Analysis)
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download model: `lms get qwen/qwen3-vl-4b`

#### Ollama (for Video Analysis)
1. Download and install [Ollama](https://ollama.com/)
2. Download models:
```bash
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b
```

## 💻 Usage

### Geospatial Analysis

```bash
cd "Geospatial Analysis"
python app_flask.py
# Open http://127.0.0.1:5000 in browser
```

### Video Analysis

```bash
cd "Video - Analysis"
python app_flask.py
# Open http://localhost:5000 in browser
```

### LLM Distillation

```bash
cd LLM
python run_all_distillation.py
# Or run individual methods
python soft_label_distillation.py
python hard_label_distillation.py
python co_distillation.py
```

### Machine Learning

```bash
cd ML
python app.py
# Or use Jupyter notebooks
jupyter notebook
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Flask**: Web framework for applications
- **Gradio**: Quick ML interface creation
- **Jupyter Notebooks**: Interactive development

### AI/ML Libraries
- **Hugging Face Transformers**: Pre-trained models
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Plotly**: Data visualization

### AI Services
- **LM Studio**: Local LLM hosting
- **Ollama**: Local AI model serving
- **Google Gemini**: Cloud AI API
- **OpenAI API**: GPT models (where applicable)

### Specialized Libraries
- **Folium**: Interactive mapping
- **GeoPandas**: Geospatial data processing
- **OpenCV**: Computer vision
- **NLTK**: Natural language processing
- **ChromaDB**: Vector database
- **spaCy**: NLP library

## 📊 Project Statistics

- **Total Projects**: 15+ major projects
- **Languages**: Python
- **Frameworks**: Flask, Gradio, Jupyter
- **AI Models**: LLMs, VLMs, Embeddings
- **Applications**: Web apps, CLIs, Notebooks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Emre Developer**
- GitHub: [@emredeveloper](https://github.com/emredeveloper)
- Repository: [Artifical-Intelligence-Trials-Notebooks](https://github.com/emredeveloper/Artifical-Intelligence-Trials-Notebooks)

## 🙏 Acknowledgments

- Hugging Face for transformer models
- LM Studio for local LLM hosting
- Ollama for local AI model serving
- All open-source contributors whose libraries made this possible

## 📚 Additional Resources

### Documentation
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [Ollama Documentation](https://ollama.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Flask Documentation](https://flask.palletsprojects.com/)

### Related Projects
- [Geospatial Analysis README](./Geospatial%20Analysis/README.md)
- [Video Analysis README](./Video%20-%20Analysis/README.md)
- [LLM README](./LLM/README.md)

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

Made with ❤️ by [Emre Developer](https://github.com/emredeveloper)

</div>

