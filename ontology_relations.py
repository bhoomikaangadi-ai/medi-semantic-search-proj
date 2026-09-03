import json
import spacy
import scispacy
from scispacy.linking import EntityLinker

print("Loading SciSpacy...")

nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe(
    "scispacy_linker",
    config={
        "resolve_abbreviations": True,
        "linker_name": "umls"
    }
)

print("Loading chunks...")

with open("medical_chunks_source.json", encoding="utf-8") as f:
    chunks = json.load(f)

relations = []

# Relationship keywords (generic)
RELATION_PATTERNS = {
    "HAS_SYMPTOM": [
        "symptom",
        "develop",
        "present",
        "experience",
        "complain",
        "accompanied"
    ],
    "TREATED_WITH": [
        "treated with",
        "treatment",
        "therapy",
        "administer"
    ],
    "DIAGNOSED_BY": [
        "diagnosed by",
        "diagnosis",
        "tested",
        "pcr",
        "elisa",
        "microscopy"
    ],
    "TRANSMITTED_BY": [
        "transmitted by",
        "spread by",
        "mosquito",
        "vector"
    ]
}

for chunk in chunks:

    doc = nlp(chunk["text"])

    entities = []

    for ent in doc.ents:

        if ent._.kb_ents:

            entities.append(ent.text)

    entities = list(dict.fromkeys(entities))

    text = chunk["text"].lower()

    for rel_type, keywords in RELATION_PATTERNS.items():

        if any(k in text for k in keywords):

            for i in range(len(entities)-1):

                relations.append({
                    "source": entities[i],
                    "target": entities[i+1],
                    "type": rel_type,
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document"],
                    "source_type": chunk["source"]
                })

# Remove duplicates
seen = set()
unique = []

for r in relations:

    key = (
        r["source"],
        r["target"],
        r["type"]
    )

    if key not in seen:

        seen.add(key)
        unique.append(r)

with open(
    "medical_relations.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        unique,
        f,
        indent=2,
        ensure_ascii=False
    )

print("="*40)
print("GENERIC ONTOLOGY COMPLETE")
print("="*40)
print("Relationships:", len(unique))
print("Saved: medical_relations.json")