import faiss
import numpy as np
import json

index = faiss.read_index("medical_faiss.index")
embeddings = np.load("medical_embeddings.npy")

with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

target = 88
query_vector = embeddings[target:target + 1].astype("float32")

distances, indices = index.search(query_vector, 5)

print("Expected passage index:", target)
print("\nTop 5 results using the passage's own vector:")

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]), start=1
):
    idx = int(idx)
    print(
        f"Rank {rank}: index={idx}, "
        f"chunk_id={chunks[idx].get('chunk_id')}, "
        f"distance={distance:.6f}"
    )
    print(chunks[idx]["text"][:180].replace("\n", " "))
    print()