import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# Load saved document embeddings and chunks
saved = np.load("medical_embeddings.npy")
with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Check the dengue symptom passage at list index 88
passage = chunks[88]["text"]

print("Saved embeddings shape:", saved.shape)
print("Chunks count:", len(chunks))
print("Checking passage index: 88")

# Generate its embedding again using the same method
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer(
    passage,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

with torch.no_grad():
    output = model(**inputs)

mask = inputs["attention_mask"].unsqueeze(-1).float()
embedding = (
    (output.last_hidden_state * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
).cpu().numpy()[0]

stored = saved[88]

cosine = np.dot(stored, embedding) / (
    np.linalg.norm(stored) * np.linalg.norm(embedding) + 1e-12
)

print("Stored vector norm:", np.linalg.norm(stored))
print("Recomputed vector norm:", np.linalg.norm(embedding))
print("Cosine similarity:", cosine)

if cosine > 0.999:
    print("RESULT: Stored and recomputed embeddings match.")
else:
    print("RESULT: Embeddings differ; investigate index generation.")