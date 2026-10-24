import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import faiss
import torch
import ollama
from transformers import AutoTokenizer, AutoModel

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
OLLAMA_MODEL = "qwen2.5:7b"

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Loading FAISS...")

index = faiss.read_index("medical_faiss.index")

with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("medical_graph.json", encoding="utf-8") as f:
    graph = json.load(f)

print("System Ready!")

# ==========================================================
# EMBEDDING
# ==========================================================

def embed(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        output = model(**inputs)

    return output.last_hidden_state.mean(dim=1).numpy().astype("float32")

# ==========================================================
# KNOWLEDGE GRAPH LOOKUP
# ==========================================================

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

    return list(dict.fromkeys(names))

# ==========================================================
# HYBRID RERANKING
# ==========================================================

def rerank_results(query, indices, distances):

    query_words = set(query.lower().split())

    ranked = []

    for idx, distance in zip(indices[0], distances[0]):

        chunk = chunks[idx]

        keyword_score = sum(word in chunk["text"].lower()
                            for word in query_words)

        ontology_score = sum(
            word in " ".join(get_related_concepts(chunk["chunk_id"])).lower()
            for word in query_words
        )

        faiss_score = 1 / (1 + float(distance))

        final = (
            faiss_score * 0.60
            + keyword_score * 0.30
            + ontology_score * 0.10
        )

        ranked.append((final, idx))

    ranked.sort(reverse=True)

    return ranked[:3]

# ==========================================================
# DISPLAY
# ==========================================================

def line():
    print("=" * 90)

def section(title):
    print(f"\n{title}")
    print("-" * 90)

# ==========================================================
# SEARCH LOOP
# ==========================================================

while True:

    query = input("\n🔍 Search Medical Information (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("\nThank you for using Medical Graph RAG.")
        break

    print("\nSearching WHO medical knowledge...")

    q = embed(query)

    distances, indices = index.search(q, 20)

    ranked = rerank_results(query, indices, distances)

    evidence = []
    concepts = set()

    for score, idx in ranked:

        c = chunks[idx]

        evidence.append(
            f"[WHO | {c['document']} | Chunk {c['chunk_id']}]\n{c['text']}"
        )

        concepts.update(get_related_concepts(c["chunk_id"]))

    prompt = f"""
You are an AI Medical Graph RAG Assistant.

Answer ONLY using the retrieved WHO medical evidence.

Format:
1. One-sentence direct answer.
2. 3-6 bullet points.
3. One-line conclusion.

Question:
{query}

Retrieved Evidence:
{chr(10).join(evidence)}

Related Concepts:
{", ".join(sorted(concepts))}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # ======================================================
    # OUTPUT
    # ======================================================

    print()
    line()
    print("🩺 MEDICAL GRAPH RAG SEARCH")
    line()

    print(f"\nQuery: {query}")

    # ------------------------------------------------------
    section("AI Overview")

    print(response["message"]["content"])

    # ------------------------------------------------------
    section("Top Relevant Documents")

    for rank, (score, idx) in enumerate(ranked, start=1):

        c = chunks[idx]

        print(f"[{rank}] WHO Medical Evidence")
        print(f"Document   : {c['document']}")
        print(f"Source     : {c['source']}")
        print(f"Chunk ID   : {c['chunk_id']}")
        print(f"Relevance  : {score:.3f}")
        print()

    # ------------------------------------------------------
    section("Knowledge Graph Concepts")

    if concepts:
        print(", ".join(sorted(concepts)[:25]))
    else:
        print("No linked concepts found.")

    # ------------------------------------------------------
    section("Related Images")

    print("Future Feature")
    print("• WHO medical illustrations")
    print("• Disease lifecycle diagrams")
    print("• Vector identification images")
    print("• Symptom reference images")

    # ------------------------------------------------------
    section("Related Videos")

    print("Future Feature")
    print("• WHO educational videos")
    print("• Disease awareness videos")
    print("• Prevention tutorials")

    # ------------------------------------------------------
    section("PubMed Research")

    print("Future Feature")
    print("• Latest PubMed research papers")
    print("• Clinical studies")
    print("• DOI and abstracts")

    # ------------------------------------------------------
    section("Kaggle Datasets")

    print("Future Feature")
    print("• Medical datasets")
    print("• Disease prediction datasets")
    print("• Research datasets")

    # ------------------------------------------------------
    section("Search Summary")

    print(f"Candidates searched : 20")
    print(f"Results shown       : {len(ranked)}")
    print(f"Knowledge concepts  : {len(concepts)}")

    line()