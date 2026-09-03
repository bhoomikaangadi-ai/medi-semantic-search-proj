import requests

text = "The patient was diagnosed with pneumonia."

print("Generating embedding...")

response = requests.post(
    "http://127.0.0.1:11434/api/embed",
    json={
        "model": "nomic-embed-text:latest",
        "input": text
    },
    timeout=120
)

response.raise_for_status()

data = response.json()

embedding = data["embeddings"][0]

print("Embedding generated!")
print("Type:", type(embedding))
print("Length:", len(embedding))
print("First 10 values:", embedding[:10])