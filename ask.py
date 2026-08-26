import os

# Windows OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import faiss
import torch
import ollama
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
OLLAMA_MODEL = "qwen2.5:7b"

# ==========================================
# LOAD BIOBERT
# ==========================================

print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ==========================================
# LOAD FAISS + DATA
# ==========================================

print("Loading FAISS...")

index = faiss.read_index("medical_faiss.index")

with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("medical_graph.json", encoding="utf-8") as f:
    graph = json.load(f)

print("System Ready!")

# ==========================================
# BIOBERT EMBEDDING
# ==========================================

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        output = model(**inputs)

    return output.last_hidden_state.mean(dim=1).numpy().astype("float32")

# ==========================================
# ONTOLOGY LOOKUP
# ==========================================

def get_related_concepts(chunk_id):

    chunk_name = f"chunk_{chunk_id}"

    concept_ids = [
        rel["target"]
        for rel in graph["relationships"]
        if rel["source"] == chunk_name
    ]

    names = []

    for node in graph["nodes"]:
        if node["id"] in concept_ids:
            names.append(node["name"])

    return list(dict.fromkeys(names))[:15]

# ==========================================
# HYBRID RERANKER
# ==========================================

def rerank_results(query, indices, distances):

    query_words = set(query.lower().split())

    ranked = []

    for idx, distance in zip(indices[0], distances[0]):

        chunk = chunks[idx]
        text = chunk["text"].lower()

        keyword_score = sum(word in text for word in query_words)

        concepts = get_related_concepts(chunk["chunk_id"])
        concept_text = " ".join(concepts).lower()

        ontology_score = sum(word in concept_text for word in query_words)

        faiss_score = 1 / (1 + float(distance))

        final_score = (
            faiss_score * 0.6
            + keyword_score * 0.3
            + ontology_score * 0.1
        )

        ranked.append((final_score, idx))

    ranked.sort(reverse=True)

    return [idx for _, idx in ranked[:5]]

# ==========================================
# SEARCH LOOP
# ==========================================

while True:

    query = input("\nAsk a medical question (or type exit): ")

    if query.lower() == "exit":
        break

    # BioBERT embedding
    q = embed(query)

    # Retrieve top-20 candidates from FAISS
    distances, indices = index.search(q, 20)

    # Hybrid reranking
    best_indices = rerank_results(query, indices, distances)

    evidence = []
    concepts = set()

    for idx in best_indices:

        c = chunks[idx]

        evidence.append(
            f"[Source: {c['source']} | {c['document']} | Chunk {c['chunk_id']}]\n{c['text']}"
        )

        concepts.update(get_related_concepts(c["chunk_id"]))

    prompt = f"""
You are an AI medical semantic search assistant.

Answer ONLY using the retrieved medical evidence.

Question:
{query}

Retrieved Evidence:
{chr(10).join(evidence)}

Related Medical Concepts:
{", ".join(sorted(concepts))}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n" + "=" * 70)
    print("AI SUMMARY")
    print("=" * 70)
    print(response["message"]["content"])

    print("\nSources:")

    for idx in best_indices:
        c = chunks[idx]
        print(
            f"- {c['source']} | {c['document']} | Chunk {c['chunk_id']}"
        )

    print("\nRelated Concepts:")
    print(", ".join(sorted(concepts)))