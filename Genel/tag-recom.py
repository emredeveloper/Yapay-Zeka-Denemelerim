import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample questions and tags (you would replace this with your own dataset)
questions = [
    "What is the capital of France?",
    "How do I declare a variable in Python?",
    "What's the difference between RAM and ROM?",
    "Who wrote 'To Kill a Mockingbird'?",
    "How do I calculate the area of a circle?"
]

tags = [
    ["geography", "cities"],
    ["programming", "python", "variables"],
    ["computer-science", "hardware"],
    ["literature", "authors"],
    ["mathematics", "geometry"]
]

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([w for w in tokens if w not in stop_words and w.isalnum()])

preprocessed_questions = [preprocess(q) for q in questions]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

print(X)
# Prepare tags
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tags)

# Train the model
classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(X, y)

# Function to predict tags for a new question
def predict_tags(question):
    preprocessed = preprocess(question)
    vector = vectorizer.transform([preprocessed])
    predictions = classifier.predict(vector)
    return mlb.inverse_transform(predictions)[0]

# Test the system
new_question = "what is the math?"
predicted_tags = predict_tags(new_question)
print(f"Question: {new_question}")
print(f"Predicted tags: {predicted_tags}")