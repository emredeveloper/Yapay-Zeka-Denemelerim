import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import requests
import io
from torch.utils.data import Dataset, DataLoader
import time

class LightningIndexer(nn.Module):
    """Basitleştirilmiş Lightning Indexer"""
    
    def __init__(self, d_model: int, num_indexer_heads: int = 4, indexer_dim: int = 32):
        super().__init__()
        self.num_indexer_heads = num_indexer_heads
        self.indexer_dim = indexer_dim
        
        self.query_proj = nn.Linear(d_model, num_indexer_heads * indexer_dim)
        self.key_proj = nn.Linear(d_model, num_indexer_heads * indexer_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        
        q_index = self.query_proj(hidden_states)
        k_index = self.key_proj(hidden_states)
        
        q_index = q_index.view(batch_size, seq_len, self.num_indexer_heads, self.indexer_dim)
        k_index = k_index.view(batch_size, seq_len, self.num_indexer_heads, self.indexer_dim)
        
        q_expanded = q_index.unsqueeze(2)
        k_expanded = k_index.unsqueeze(1)
        
        dot_product = torch.sum(q_expanded * k_expanded, dim=-1)
        activated = F.relu(dot_product)
        
        index_scores = torch.mean(activated, dim=-1)
        
        return index_scores

class SimpleSparseAttention(nn.Module):
    """Basit ve Çalışan Sparse Attention"""
    
    def __init__(self, d_model: int, num_heads: int, top_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.indexer = LightningIndexer(d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        with torch.no_grad():
            index_scores = self.indexer(x)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            index_scores = index_scores * causal_mask - 1e9 * (1 - causal_mask)
            
            top_k = min(self.top_k, seq_len)
            _, selected_indices = torch.topk(index_scores, k=top_k, dim=-1)
        
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        output = torch.zeros_like(Q)
        
        for batch_idx in range(batch_size):
            for head_idx in range(self.num_heads):
                for seq_idx in range(seq_len):
                    selected_idx = selected_indices[batch_idx, seq_idx]
                    K_selected = K[batch_idx, selected_idx, head_idx]
                    V_selected = V[batch_idx, selected_idx, head_idx]
                    
                    q = Q[batch_idx, head_idx, seq_idx]
                    scores = torch.matmul(q, K_selected.t()) * self.scale
                    attn_weights = F.softmax(scores, dim=-1)
                    output[batch_idx, head_idx, seq_idx] = torch.matmul(attn_weights, V_selected)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.wo(output)
        
        return output

class SimpleDSABlock(nn.Module):
    """Basit DSA Bloğu"""
    
    def __init__(self, d_model: int, num_heads: int, top_k: int = 8):
        super().__init__()
        self.attention = SimpleSparseAttention(d_model, num_heads, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class SimpleDSAModel(nn.Module):
    """Çalışan Basit DSA Modeli"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, top_k: int = 4, max_seq_len: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            SimpleDSABlock(d_model, num_heads, top_k)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len]
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return (loss, logits) if loss is not None else (logits,)

# 🗂️ GERÇEK VERİ SETİ İŞLEMLERİ

class TextDataset(Dataset):
    """Metin veri seti"""
    
    def __init__(self, texts, tokenizer, seq_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize ve padding
        tokens = self.tokenizer.encode(text)
        
        # Sequence uzunluğuna göre kes
        if len(tokens) > self.seq_length:
            start_idx = torch.randint(0, len(tokens) - self.seq_length, (1,)).item()
            tokens = tokens[start_idx:start_idx + self.seq_length]
        else:
            # Padding
            tokens = tokens + [0] * (self.seq_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels, input_ids ile aynı (causal LM)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class SimpleTokenizer:
    """Geliştirilmiş tokenizer"""
    
    def __init__(self):
        self.vocab_size = 10000
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        # Vocabulary oluştur
        self.vocab = self._create_vocab()
        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        
    def _create_vocab(self):
        """Vocabulary oluştur"""
        vocab = {0: '<PAD>', 1: '<UNK>'}
        
        # Karakter seti
        chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?,;:-\'"')
        for i, char in enumerate(chars):
            vocab[i + 2] = char
        
        # Yaygın kelimeler
        common_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on',
                       'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one',
                       'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can',
                       'said', 'there', 'each', 'which', 'do', 'how', 'their', 'if', 'will', 'up', 'other',
                       'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
                       'like', 'into', 'him', 'has', 'two', 'more', 'very', 'after', 'words', 'first', 'been',
                       'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made',
                       'may', 'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know', 'place',
                       'year', 'live', 'me', 'back', 'give', 'most', 'very', 'good', 'man', 'think', 'say',
                       'great', 'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too', 'means',
                       'old', 'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'also', 'around',
                       'form', 'three', 'small', 'set', 'put', 'end', 'why', 'again', 'turn', 'here', 'off',
                       'went', 'old', 'number', 'no', 'way', 'could', 'people', 'my', 'than', 'water', 'call',
                       'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great', 'where', 'help',
                       'deep', 'learning', 'neural', 'network', 'model', 'artificial', 'intelligence', 'language',
                       'natural', 'processing', 'machine', 'computer', 'data', 'algorithm', 'training', 'attention']
        
        for j, word in enumerate(common_words[:200]):  # İlk 200 kelime
            vocab[len(chars) + j + 2] = word
            
        return vocab
    
    def encode(self, text):
        """Metni token ID'lere çevir"""
        tokens = []
        text = text.lower().strip()
        words = text.split()
        
        for word in words:
            # Önce kelimeyi ara
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Kelime bulunamazsa karakter karakter
                for char in word:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.unk_token_id)
        
        return tokens[:128]  # Maksimum 128 token
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        if isinstance(token_ids, int):
            return self.vocab.get(token_ids, '<UNK>')
        elif isinstance(token_ids, torch.Tensor):
            tokens = [self.vocab.get(tid.item(), '<UNK>') for tid in token_ids]
        else:
            tokens = [self.vocab.get(tid, '<UNK>') for tid in token_ids]
        
        # Kelimeleri birleştir
        result = ''
        for i, token in enumerate(tokens):
            if token == '<PAD>':
                break
            elif len(token) == 1 and not token.isalnum():
                result += token
            elif i == 0 or tokens[i-1] in [' ', '.', '!', '?', ',', ';', ':', '-']:
                result += token
            else:
                result += ' ' + token
        
        return result.strip()

def load_wikitext_data():
    """WikiText-2 verisini yükle"""
    print("📥 WikiText-2 verisi yükleniyor...")
    
    try:
        # WikiText-2 raw text link
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        response = requests.get(url)
        response.raise_for_status()
        
        text = response.text
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
        
        print(f"✅ {len(paragraphs)} paragraf yüklendi")
        return paragraphs[:100]  # İlk 100 paragraf ile sınırlı
        
    except Exception as e:
        print(f"❌ WikiText yükleme hatası: {e}")
        print("🔧 Örnek veri oluşturuluyor...")
        return create_sample_data()

def create_sample_data():
    """Örnek veri oluştur"""
    sample_texts = [
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand and generate human language.",
        "Transformers have revolutionized the field of natural language processing in recent years.",
        "Sparse attention mechanisms allow models to handle longer sequences efficiently.",
        "The development of large language models has accelerated progress in AI research.",
        "Reinforcement learning involves training agents through rewards and punishments.",
        "Computer vision algorithms can identify objects and patterns in images and videos.",
        "The attention mechanism allows models to focus on relevant parts of the input.",
        "Generative AI can create new content such as text, images, and music.",
        "Transfer learning enables models to apply knowledge from one task to another."
    ] * 10  # 100 örnek oluştur
    
    return sample_texts

class DSATrainer:
    """DSA Eğitici"""
    
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.parameters()).device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, logits = self.model(input_ids, labels=labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            if batch_idx % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | LR: {current_lr:.2e}')
        
        avg_loss = total_loss / total_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Validation"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(input_ids, labels=labels)
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / total_batches
        self.val_losses.append(avg_loss)
        print(f'Validation Epoch {epoch} | Loss: {avg_loss:.4f}')
        return avg_loss
    
    def train(self, epochs=5):
        """Tam eğitim"""
        print("🚀 DSA Model Eğitimi Başlıyor...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f'🎯 Epoch {epoch} Tamamlandı | '
                  f'Time: {epoch_time:.1f}s | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f}')
            
            # En iyi modeli kaydet
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_dsa_model.pth')
                print(f'💾 Yeni en iyi model kaydedildi! (Loss: {val_loss:.4f})')
            
            print('-' * 60)

def test_trained_model(model, tokenizer, device):
    """Eğitilmiş modeli test et"""
    print("\n🧪 Model Testi...")
    
    test_sentences = [
        "The artificial intelligence",
        "Deep learning models",
        "Natural language"
    ]
    
    model.eval()
    
    for sentence in test_sentences:
        # Tokenize
        tokens = tokenizer.encode(sentence)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            loss, logits = model(input_ids, labels=input_ids)
            
            # En olası sonraki token'lar
            next_token_logits = logits[0, -1, :]
            top_tokens = torch.topk(next_token_logits, 5)
            
            print(f"\nInput: '{sentence}'")
            print(f"Tokenized: {tokens}")
            print(f"Decoded input: '{tokenizer.decode(tokens)}'")
            print(f"Loss: {loss.item():.4f}")
            print("Top 5 next token predictions:")
            for i, (score, idx) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                predicted_token = tokenizer.decode(idx.item())
                print(f"  {i+1}. '{predicted_token}' (Token {idx.item()}, score: {score:.3f})")

def main():
    """Ana eğitim fonksiyonu"""
    print("=" * 70)
    print("           DEEPSEEK SPARSE ATTENTION - GERÇEK VERİ EĞİTİMİ")
    print("=" * 70)
    
    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Kullanılan cihaz: {device}")
    
    # Veri yükle
    texts = load_wikitext_data()
    
    # Tokenizer
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"📊 Veri istatistikleri:")
    print(f"   - Paragraf sayısı: {len(texts)}")
    print(f"   - Vocabulary size: {vocab_size}")
    
    # Dataset oluştur
    dataset = TextDataset(texts, tokenizer, seq_length=64)  # Kısa sequence'ler
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: 8")
    print(f"   - Sequence length: 64")
    
    # Model
    model = SimpleDSAModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        top_k=8,
        max_seq_len=64
    ).to(device)
    
    print(f"🤖 Model oluşturuldu:")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Layers: 2")
    print(f"   - Heads: 4") 
    print(f"   - Top-k: 8")
    
    # Eğitici
    trainer = DSATrainer(model, train_loader, val_loader, learning_rate=1e-3)
    
    # Eğitim
    print("\n🔥 EĞİTİM BAŞLIYOR...")
    trainer.train(epochs=5)
    
    # Eğitim grafiği
    print("\n📈 Eğitim Sonuçları:")
    print(f"   - Son train loss: {trainer.train_losses[-1]:.4f}")
    print(f"   - Son val loss: {trainer.val_losses[-1]:.4f}")
    print(f"   - En iyi val loss: {min(trainer.val_losses):.4f}")
    
    # Model testi
    test_trained_model(model, tokenizer, device)
    
    print("\n🎉 EĞİTİM TAMAMLANDI!")
    print("💾 En iyi model 'best_dsa_model.pth' olarak kaydedildi")
    
def test_trained_model_with_text(model, tokenizer, device):
    """Eğitilmiş modeli metin üretimi ile test et"""
    print("\n🧪 Model Testi - Metin Üretimi...")
    
    # Basit bir token-to-text mapping oluşturalım
    vocab = {}
    for i in range(tokenizer.vocab_size):
        # Token ID'yi basit bir metne çevir
        if i == 0:
            vocab[i] = "[PAD]"
        else:
            # Token ID'yi karaktere çevir (basit bir yöntem)
            char_idx = (i - 1) % 26
            vocab[i] = chr(ord('a') + char_idx) + str(i // 26)
    
    test_sentences = [
        "The artificial intelligence",
        "Deep learning models", 
        "Natural language",
        "Machine learning is",
        "The future of AI"
    ]
    
    model.eval()
    
    for sentence in test_sentences:
        print(f"\n" + "="*50)
        print(f"Input: '{sentence}'")
        print("="*50)
        
        # Tokenize
        tokens = tokenizer.encode(sentence)
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generation
        with torch.no_grad():
            loss, logits = model(input_ids, labels=input_ids)
            
            print(f"Loss: {loss.item():.4f}")
            
            # Mevcut input'un token'larını göster
            print(f"\nInput tokens: {[vocab.get(t, '?') for t in tokens]}")
            
            # Sonraki token tahminleri
            next_token_logits = logits[0, -1, :]
            top_tokens = torch.topk(next_token_logits, 10)
            
            print(f"\nTop 10 next token predictions:")
            for i, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                token_text = vocab.get(token_id.item(), f"UNK_{token_id.item()}")
                print(f"  {i+1:2d}. '{token_text}' (ID: {token_id.item():4d}, score: {score:.3f})")
            
            # Kısa bir generation yapalım
            print(f"\nGenerated continuation:")
            generated = sentence
            current_input = input_ids.clone()
            
            for step in range(10):  # 10 token generate et
                with torch.no_grad():
                    logits = model(current_input)
                    next_token_logits = logits[0, -1, :]
                    
                    # Temperature sampling
                    temperature = 0.8
                    scaled_logits = next_token_logits / temperature
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    next_token_text = vocab.get(next_token.item(), f"UNK_{next_token.item()}")
                    generated += " " + next_token_text
                    
                    # Input'u güncelle
                    current_input = torch.cat([
                        current_input, 
                        next_token.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    # Maksimum uzunluk
                    if current_input.shape[1] >= 64:
                        break
            
            print(f"   '{generated}'")

def analyze_training_results(trainer, model, tokenizer, device):
    """Eğitim sonuçlarını analiz et"""
    print("\n📊 Detaylı Eğitim Analizi")
    print("="*60)
    
    # Loss trend'ini analiz et
    if len(trainer.train_losses) > 1:
        improvement = trainer.train_losses[0] - trainer.train_losses[-1]
        print(f"🏆 Toplam iyileşme: {improvement:.4f}")
        print(f"📉 Başlangıç loss: {trainer.train_losses[0]:.4f}")
        print(f"📈 Son loss: {trainer.train_losses[-1]:.4f}")
        
        # Overfitting kontrolü
        if len(trainer.val_losses) > 0:
            train_val_gap = abs(trainer.train_losses[-1] - trainer.val_losses[-1])
            print(f"🔍 Train-Val gap: {train_val_gap:.4f}")
            if train_val_gap > 0.5:
                print("   ⚠️  Potansiyel overfitting!")
            else:
                print("   ✅ Generalization iyi görünüyor")
    
    # Perplexity hesapla
    final_val_loss = trainer.val_losses[-1] if trainer.val_losses else trainer.train_losses[-1]
    perplexity = math.exp(final_val_loss)
    print(f"🎯 Final Perplexity: {perplexity:.2f}")
    
    # Model boyutu analizi
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🤖 Model Boyutu:")
    print(f"   - Toplam parametre: {total_params:,}")
    print(f"   - Eğitilebilir parametre: {trainable_params:,}")
    
    # Vocabulary analizi
    print(f"📚 Vocabulary size: {tokenizer.vocab_size}")

def demonstrate_sparse_attention_benefits():
    """Sparse Attention'ın faydalarını göster"""
    print("\n🎯 DSA (DeepSeek Sparse Attention) Avantajları")
    print("="*60)
    
    seq_lengths = [64, 128, 256, 512]
    top_k = 8
    
    print(f"Top-k değeri: {top_k}")
    print("\nGeleneksel vs Sparse Attention Karşılaştırması:")
    print("Sequence Length | Geleneksel O(L²) | Sparse O(L·k) | Kazanç")
    print("-" * 65)
    
    for L in seq_lengths:
        traditional_ops = L * L
        sparse_ops = L * top_k
        gain = traditional_ops / sparse_ops
        
        print(f"{L:>14} | {traditional_ops:>16} | {sparse_ops:>12} | {gain:>6.1f}x")
    
    print(f"\n💡 Sparse Attention ile:")
    print(f"   - 64 token'da {64*64/(64*8):.1f}x daha az işlem")
    print(f"   - 512 token'da {512*512/(512*8):.1f}x daha az işlem")
    print(f"   - Daha uzun sequence'ler için ideal")

def main_with_enhanced_analysis():
    """Geliştirilmiş analiz ile ana fonksiyon"""
    print("=" * 70)
    print("           DEEPSEEK SPARSE ATTENTION - GELİŞMİŞ ANALİZ")
    print("=" * 70)
    
    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Kullanılan cihaz: {device}")
    
    # Veri yükle
    texts = load_wikitext_data()
    
    # Tokenizer
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"📊 Veri istatistikleri:")
    print(f"   - Paragraf sayısı: {len(texts)}")
    print(f"   - Vocabulary size: {vocab_size}")
    
    # Dataset oluştur
    dataset = TextDataset(texts, tokenizer, seq_length=64)
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Model
    model = SimpleDSAModel(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        top_k=8,
        max_seq_len=64
    ).to(device)
    
    # Eğitici
    trainer = DSATrainer(model, train_loader, val_loader, learning_rate=1e-3)
    
    # Eğitim
    print("\n🔥 EĞİTİM BAŞLIYOR...")
    trainer.train(epochs=5)
    
    # Gelişmiş analiz
    analyze_training_results(trainer, model, tokenizer, device)
    
    # Metin üretimi testi
    test_trained_model_with_text(model, tokenizer, device)
    
    # Sparse attention avantajları
    demonstrate_sparse_attention_benefits()
    
    print("\n🎉 EĞİTİM VE ANALİZ TAMAMLANDI!")
    print("💾 En iyi model 'best_dsa_model.pth' olarak kaydedildi")

# Tokenizer'ı geliştirelim
class ImprovedTokenizer:
    """Geliştirilmiş tokenizer"""
    
    def __init__(self):
        self.vocab_size = 10000
        self.pad_token_id = 0
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        """Basit bir vocabulary oluştur"""
        vocab = {0: "[PAD]"}
        
        # Harfler ve sayılar
        chars = []
        chars.extend([chr(ord('a') + i) for i in range(26)])  # a-z
        chars.extend([chr(ord('A') + i) for i in range(26)])  # A-Z
        chars.extend([str(i) for i in range(10)])  # 0-9
        chars.extend([' ', '.', ',', '!', '?', '-', '"', "'", ':', ';'])
        
        for i, char in enumerate(chars):
            vocab[i + 1] = char
            
        # Kelimeler için ek token'lar
        common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                       'can', 'her', 'was', 'one', 'our', 'out', 'get', 'has',
                       'him', 'how', 'man', 'new', 'now', 'old', 'see', 'two',
                       'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say',
                       'she', 'too', 'use', 'that', 'with', 'this', 'from']
        
        for j, word in enumerate(common_words):
            vocab[len(chars) + j + 1] = word
            
        return vocab
    
    def decode(self, token_ids):
        """Token ID'leri metne çevir"""
        if isinstance(token_ids, int):
            return self.vocab.get(token_ids, f"UNK_{token_ids}")
        elif isinstance(token_ids, torch.Tensor):
            return [self.vocab.get(tid.item(), f"UNK_{tid.item()}") for tid in token_ids]
        else:
            return [self.vocab.get(tid, f"UNK_{tid}") for tid in token_ids]
    
    def encode(self, text):
        """Metni token ID'lere çevir"""
        tokens = []
        words = text.lower().split()
        
        for word in words:
            # Önce tam kelimeyi ara
            found = False
            for token_id, token_text in self.vocab.items():
                if token_text == word and token_id != 0:
                    tokens.append(token_id)
                    found = True
                    break
            
            # Kelime bulunamazsa karakter karakter
            if not found:
                for char in word:
                    for token_id, token_text in self.vocab.items():
                        if token_text == char and token_id != 0:
                            tokens.append(token_id)
                            break
        
        return tokens[:128]  # Maksimum 128 token

# Geliştirilmiş tokenizer ile test
def test_with_improved_tokenizer():
    """Geliştirilmiş tokenizer ile test"""
    print("\n🔤 Geliştirilmiş Tokenizer Testi...")
    
    tokenizer = ImprovedTokenizer()
    
    test_texts = [
        "hello world",
        "the quick brown fox",
        "deep learning model"
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Decoded: '{' '.join(decoded)}'")
        print()

if __name__ == "__main__":
    main()