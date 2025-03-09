import ollama
response = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[
        {"role": "user", "content": "Explain Newton's second law of motion"},
    ],
)
print(response["message"]["content"])