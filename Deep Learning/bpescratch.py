from huggingface import login
login()


corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(words_with_offsets)
    new_words = [word for word, offset in words_with_offsets]
    print(new_words)
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)