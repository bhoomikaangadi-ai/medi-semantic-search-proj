
import json
import faiss
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

with open(ROOT / "medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index(str(ROOT / "medical_faiss.index"))

question = "What are the common symptoms of dengue fever?"

print("Loading BioBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer(
    question,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

with torch.no_grad():
    output = model(**inputs)

mask = inputs["attention_mask"].unsqueeze(-1)
vector = (
    (output.last_hidden_state * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
).cpu().numpy().astype("float32")

if vector.shape[1] != index.d:
    raise ValueError(
        f"Embedding dimension {vector.shape[1]} != index dimension {index.d}"
    )

distances, indices = index.search(vector, min(20, index.ntotal))

print("\nTop 10 FAISS results:")
for rank, (idx, distance) in enumerate(
    zip(indices[0][:10], distances[0][:10]), start=1
):
    idx = int(idx)
    chunk = chunks[idx]
    print(f"\nRank: {rank} | List index: {idx} | Distance: {distance:.4f}")
    print("Chunk ID:", chunk.get("chunk_id"))
    print("Text:", chunk.get("text", "")[:350].replace("\n", " "))

target_index = 88
positions = np.where(indices[0] == target_index)[0]

print("\nTarget symptom passage:")
print("List index: 88")
print("Chunk ID:", chunks[88].get("chunk_id"))

if len(positions):
    rank = int(positions[0]) + 1
    print(f"Found in top 20 at rank {rank}.")
else:
    print("Not found in the top 20 FAISS results.")
