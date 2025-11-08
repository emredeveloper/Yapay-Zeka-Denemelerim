from google import genai
from pydantic import BaseModel
from typing import List
import json

class Person(BaseModel):
    name: str
    age: int
    email: str

client = genai.Client(api_key="AIzaSyBDTJmH-oCCq9Td7G6g93_93yHH3gTcJkg")

# Liste için JSON Schema tanımı
list_schema = {
    "type": "array",
    "items": Person.model_json_schema(),
    "minItems": 3,
    "maxItems": 3
}

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give me 3 fictional persons with name, age and email.",
    config={
        "response_mime_type": "application/json",
        "response_json_schema": list_schema,
    },
)

print(response.text)      # JSON çıktısı olarak

# JSON'u parse edip Pydantic modellerine dönüştür
json_data = json.loads(response.text)
people = [Person(**person_data) for person_data in json_data]

for p in people:
    print(p.name, p.age, p.email)
