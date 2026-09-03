import json
import spacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict

print("Loading SciSpacy...")

nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe(
    "scispacy_linker",
    config={
        "resolve_abbreviations": True,
        "linker_name": "mesh"
    }
)

print("Loading medical chunks...")

with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

# ============================================
# EXTRACT ONTOLOGY NODES & RELATIONSHIPS
# ============================================

print("\nExtracting ontology concepts...")

linker = nlp.get_pipe("scispacy_linker")

nodes = {}
relationships = []

for i, chunk in enumerate(chunks):

    doc = nlp(chunk["text"])

    chunk_node = f"chunk_{chunk['chunk_id']}"

    for ent in doc.ents:

        if not ent._.kb_ents:
            continue

        cui, score = ent._.kb_ents[0]

        concept = linker.kb.cui_to_entity[cui]

        nodes[cui] = {
            "id": cui,
            "name": concept.canonical_name,
            "semantic_type": concept.types,
            "source": chunk["source"]
        }

        relationships.append({
            "source": chunk_node,
            "target": cui,
            "type": "MENTIONS",
            "score": float(score),
            "document": chunk["document"]
        })

    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(chunks)} chunks")

# ============================================
# SAVE GRAPH
# ============================================

graph = {
    "nodes": list(nodes.values()),
    "relationships": relationships
}

with open(
    "medical_graph.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        graph,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n========================================")
print("GRAPH CREATED")
print("========================================")
print("Unique concepts:", len(nodes))
print("Relationships:", len(relationships))
print("Output: medical_graph.json")
print("========================================")