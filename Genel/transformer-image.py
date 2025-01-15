import requests

API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512"
headers = {"Authorization": "Bearer "}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("turkey.png")

print(output)