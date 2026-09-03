import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel


# ========================================
# LOAD BIOBERT
# ========================================

print("Loading BioBERT...")

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("BioBERT loaded.")


# ========================================
# LOAD FAISS INDEX
# ========================================

print("Loading FAISS index...")

index = faiss.read_index("medical_faiss.index")

print("FAISS vectors:", index.ntotal)


# ========================================
# LOAD MEDICAL CHUNKS
# ========================================

with open(
    "medical_chunks_source.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print("Medical chunks:", len(chunks))


# ========================================
# BIOBERT EMBEDDING FUNCTION
# ========================================

def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]

    embedding = embedding.numpy()

    return embedding


# ========================================
# KEYWORD RELEVANCE
# ========================================

def keyword_score(query, text):

    query_words = query.lower().split()

    text_lower = text.lower()

    score = 0

    for word in query_words:

        if len(word) > 3 and word in text_lower:

            score += 1

    return score


# ========================================
# SEARCH
# ========================================

query = input("\nEnter your medical question: ")

print("\nSearching FAISS...")

query_embedding = get_embedding(query)

distances, indices = index.search(
    query_embedding,
    15
)


# ========================================
# COLLECT RESULTS
# ========================================

results = []

for distance, idx in zip(
    distances[0],
    indices[0]
):

    chunk = chunks[idx]

    score = keyword_score(
        query,
        chunk["text"]
    )

    results.append(
        {
            "distance": float(distance),
            "keyword_score": score,
            "chunk": chunk
        }
    )


# ========================================
# SORT RESULTS
# ========================================

results.sort(
    key=lambda x: (
        -x["keyword_score"],
        x["distance"]
    )
)


# ========================================
# DISPLAY
# ========================================

print("\n")
print("=" * 80)
print("IMPROVED FAISS SEARCH RESULTS")
print("=" * 80)


for rank, result in enumerate(
    results[:5],
    start=1
):

    chunk = result["chunk"]

    print("\n")
    print(f"RESULT {rank}")
    print("-" * 80)

    print(
        "FAISS Distance:",
        result["distance"]
    )

    print(
        "Keyword Score:",
        result["keyword_score"]
    )

    print(
        "Source:",
        chunk["source"]
    )

    print(
        "Document:",
        chunk["document"]
    )

    print(
        "Chunk ID:",
        chunk["chunk_id"]
    )

    print("\nTEXT:")

    print(
        chunk["text"][:2000]
    )

    print("-" * 80)