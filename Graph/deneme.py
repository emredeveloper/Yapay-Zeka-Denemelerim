from train_tokenizer import ImprovedTurkishTokenizer  # Changed from improved_tokenizer

text = "Doğal Dil İşleme, bilgisayarların Türkçe dahil insan dillerini anlamasını sağlar."
tokenizer = ImprovedTurkishTokenizer.load("turkish_tokenizer_model_strict_case")
tokens = tokenizer.tokenize(text)
decoded = tokenizer.decode(tokens)
print(f"Original: {text}")
print(f"Decoded:  {decoded}")
print(f"Match:    {'✓' if decoded == text else '✗'}")