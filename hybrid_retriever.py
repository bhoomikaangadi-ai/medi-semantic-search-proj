import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from graph_retriever import find_entity, get_related
print("Loading FAISS...")

index = faiss.read_index("medical_faiss.index")

embeddings = np.load("medical_embeddings.npy")

with open("medical_chunks_source.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Chunks: {len(chunks)}")

model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
def semantic_search(query, top_k=3):

    query_vector = model.encode([query])

    distances, indices = index.search(
        np.array(query_vector).astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        results.append({
            "score": float(score),
            "chunk": chunks[idx]
        })

    return results
def graph_search(query):

    entity = find_entity(query)

    if entity is None:
        return None

    return {
        "entity": entity,
        "related": get_related(entity["id"])
    }
def hybrid_search(query):

    semantic = semantic_search(query)

    graph = graph_search(query)

    return {
        "query": query,
        "semantic_results": semantic,
        "graph_results": graph
    }
while True:

    print("\nAsk a medical question (exit to quit):")

    query = input("> ")

    if query.lower() == "exit":
        break

    result = hybrid_search(query)

    print("\n" + "="*70)
    print("SEMANTIC SEARCH")
    print("="*70)

    for i, item in enumerate(result["semantic_results"], 1):

        print(f"\nResult {i}")
        print(f"Score: {item['score']:.3f}")
        print(item["chunk"]["text"][:350])

    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH")
    print("="*70)

    if result["graph_results"] is None:

        print("No graph entity found.")
# ============================================================
# HYBRID RETRIEVER
# ============================================================

import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from graph_retriever import find_entity, get_related

print("=" * 70)
print("LOADING HYBRID RETRIEVER")
print("=" * 70)

# ------------------------------------------------------------
# Load FAISS
# ------------------------------------------------------------

index = faiss.read_index("medical_faiss.index")

embeddings = np.load("medical_embeddings.npy")

with open("medical_chunks_source.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Chunks: {len(chunks)}")

# ------------------------------------------------------------
# Load BioBERT model
# ------------------------------------------------------------

print("Loading BioBERT embeddings model...")

model = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb"
)

print("Model loaded.")


# ============================================================
# FAISS SEARCH
# ============================================================

def semantic_search(query, top_k=3):

    vector = model.encode([query])

    distances, indices = index.search(
        np.array(vector).astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        results.append({
            "score": float(score),
            "chunk": chunks[idx]
        })

    return results


# ============================================================
# GRAPH SEARCH
# ============================================================

def graph_search(query):

    query = query.lower()

    # Try whole query
    entity = find_entity(query)

    if entity:
        return entity, get_related(entity["id"])

    # Try individual words
    for word in query.split():

        entity = find_entity(word)

        if entity:
            return entity, get_related(entity["id"])

    return None, []


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query):

    semantic = semantic_search(query)

    entity, graph = graph_search(query)

    return {
        "semantic": semantic,
        "entity": entity,
        "graph": graph
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\nHybrid Medical Search")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nAsk: ").strip()

        if query.lower() == "exit":
            break

        result = hybrid_search(query)

        print("\n" + "=" * 70)
        print("SEMANTIC SEARCH")
        print("=" * 70)

        for i, item in enumerate(result["semantic"], 1):

            print(f"\nResult {i}")
            print(f"Score: {item['score']:.3f}")
            print(item["chunk"]["text"][:300], "...\n")

        print("=" * 70)
        print("KNOWLEDGE GRAPH")
        print("=" * 70)

        if result["entity"]:

            print(
                f"Entity: {result['entity']['canonical_name']} "
                f"({result['entity']['ontology_class']})\n"
            )

            for rel in result["graph"][:10]:

                print(
                    f"{rel['relation']:18} → "
                    f"{rel['entity']} ({rel['type']})"
                )

        else:

            print("No graph entity found.")
    else:

        print(
            f"Entity: {result['graph_results']['entity']['canonical_name']}"
        )

        for rel in result["graph_results"]["related"][:10]:

            print(
                f"{rel['relation']} → {rel['entity']} ({rel['type']})"
            )