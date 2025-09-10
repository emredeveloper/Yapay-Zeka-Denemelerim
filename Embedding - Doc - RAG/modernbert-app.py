from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModel.from_pretrained("jhu-clsp/mmBERT-base")

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.numpy()

multilingual_texts = [
    "Artificial intelligence is transforming technology",
    "La inteligencia artificial está transformando la tecnología",
    "L'intelligence artificielle transforme la technologie", 
    "Yapay zeka teknolojiyi dönüştürüyor"
]

embeddings = get_embeddings(multilingual_texts)
similarities = cosine_similarity(embeddings)
print("Cross-lingual similarity matrix:")
print(similarities)


from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
model = AutoModelForMaskedLM.from_pretrained("jhu-clsp/mmBERT-base")

def predict_masked_token(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)
    predictions = outputs.logits[mask_indices]
    top_tokens = torch.topk(predictions, 5, dim=-1)
    
    return [tokenizer.decode(token) for token in top_tokens.indices[0]]

# Works across languages
texts = [
    "The capital of France is <mask>.",
    "La capital de España es <mask>.",
    "Die Hauptstadt von Deutschland ist <mask>."
]

for text in texts:
    predictions = predict_masked_token(text)
    print(f"Text: {text}")
    print(f"Predictions: {predictions}")
