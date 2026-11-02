
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Existing medical dataset
with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk.get("text", "") for chunk in chunks]

# Test a biomedical sentence-embedding model
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

print("Loading biomedical retrieval model...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding medical passages...")
passage_embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

question = "What are the common symptoms of dengue fever?"

query_embedding = model.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# Normalized vectors: inner product gives cosine similarity
scores = passage_embeddings @ query_embedding[0]

top_indices = np.argsort(scores)[::-1][:10]

print("\nTop 10 results:")
for rank, idx in enumerate(top_indices, start=1):
    chunk = chunks[int(idx)]
    print(f"\nRank: {rank} | Index: {idx} | Score: {scores[idx]:.4f}")
    print("Chunk ID:", chunk.get("chunk_id"))
    print("Text:", chunk.get("text", "")[:500].replace("\n", " "))

# Check whether the known symptom passage was retrieved
target_index = 88
target_rank = int(np.where(top_indices == target_index)[0][0] + 1) \
    if target_index in top_indices else None

print("\nKnown symptom passage (index 88):")
if target_rank:
    print(f"Found at rank {target_rank}.")
else:
    print("Not in the top 10.")
