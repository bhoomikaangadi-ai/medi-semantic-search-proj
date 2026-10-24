# ============================================================
# BUILD MEDICAL KNOWLEDGE GRAPH (Improved)
# ============================================================

import json
import re
import spacy
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

ENTITY_FILE = "ontology_entities.json"
CHUNK_FILE = "medical_chunks_source.json"
OUTPUT_FILE = "medical_knowledge_graph.json"

print("Loading SpaCy...")
nlp = spacy.load("en_core_sci_sm")

print("Loading ontology entities...")

with open(ENTITY_FILE, "r", encoding="utf-8") as f:
    entity_data = json.load(f)

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

entities = entity_data["entities"]
mentions = entity_data["entity_mentions"]

# ============================================================
# BUILD NODE LOOKUP
# ============================================================

nodes = []
id_to_node = {}

for e in entities:

    node = {
        "id": e["cui"],
        "canonical_name": e["canonical_name"],
        "ontology_class": e["ontology_class"],
        "semantic_types": e["semantic_types"]
    }

    nodes.append(node)
    id_to_node[e["cui"]] = node

# ============================================================
# GROUP MENTIONS BY CHUNK
# ============================================================

chunk_mentions = defaultdict(list)

for m in mentions:
    chunk_mentions[m["chunk_id"]].append(m)

# ============================================================
# RELATION PATTERNS
# ============================================================

RELATION_PATTERNS = {
    ("Disease", "Symptom"): (
        "HAS_SYMPTOM",
        [
            "develop",
            "develops",
            "developed",
            "symptoms include",
            "accompanied by",
            "presents with",
            "characterized by"
        ]
    ),

    ("Disease", "Sign"): (
        "HAS_SIGN",
        [
            "sign",
            "bleeding",
            "shock",
            "hypotension",
            "hematocrit"
        ]
    ),

    ("Disease", "DiagnosticTest"): (
        "DIAGNOSED_BY",
        [
            "diagnosed by",
            "diagnosis",
            "diagnostic",
            "elisa",
            "pcr",
            "serology",
            "platelet"
        ]
    ),

    ("Disease", "Treatment"): (
        "TREATED_WITH",
        [
            "treated with",
            "management includes",
            "therapy",
            "fluid therapy",
            "rehydration",
            "transfusion"
        ]
    ),

    ("Disease", "Medication"): (
        "TREATED_WITH",
        [
            "treated with",
            "oral rehydration",
            "administered",
            "vaccine"
        ]
    ),

    ("Disease", "Pathogen"): (
        "CAUSED_BY",
        [
            "caused by",
            "caused due to",
            "transmitted by",
            "dengue virus",
            "plasmodium",
            "bacteria"
        ]
    ),

    ("Disease", "BodySystem"): (
        "AFFECTS",
        [
            "affects",
            "organ",
            "liver",
            "heart",
            "kidney"
        ]
    ),

    ("Disease", "PatientGroup"): (
        "MORE_COMMON_IN",
        [
            "children",
            "adults",
            "pregnant",
            "infants"
        ]
    ),
    ("Disease", "Complication"): (
    "HAS_COMPLICATION",
    [
        "progresses to",
        "may develop",
        "can lead to",
        "complication",
        "results in"
    ]
),
("Disease", "LaboratoryTest"): (
    "MONITORED_BY",
    [
        "monitor",
        "platelet count",
        "hematocrit",
        "blood test",
        "laboratory monitoring"
    ]
),
("Disease", "PreventionMethod"): (
    "PREVENTED_BY",
    [
        "prevent",
        "prevention",
        "mosquito net",
        "mosquito nets",
        "repellent",
        "vector control",
        "eliminate breeding sites"
    ]
),
("Disease", "GeographicRegion"): (
    "OCCURS_IN",
    [
        "occurs in",
        "found in",
        "endemic in",
        "common in",
        "tropical regions",
        "subtropical regions"
    ]
),
("Disease", "MosquitoSpecies"): (
    "TRANSMITTED_BY",
    [
        "transmitted by",
        "spread by",
        "aedes aegypti",
        "aedes albopictus"
    ]
),
("Disease", "VirusSerotype"): (
    "HAS_SEROTYPE",
    [
        "serotype",
        "denv-1",
        "denv-2",
        "denv-3",
        "denv-4"
    ]
)
}

# ============================================================
# IGNORE THESE SENTENCES
# ============================================================

IGNORE_PHRASES = [
    "differential diagnosis",
    "other diseases",
    "co-infection",
    "similar diseases",
    "compared with",
    "versus",
    "vs."
]

# ============================================================
# INVALID GRAPH ENTITIES
# ============================================================

INVALID_GRAPH_ENTITIES = {
    "clinical relevance",
    "development",
    "status",
    "result",
    "mechanism",
    "interaction",
    "purpose",
    "consideration",
    "observed",
    "system",
    "methods",
    "options",
    "target",
    "detected",
    "living",
    "pass",
    "follow-up status",
    "severe",
    "worse"
}

# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def valid_pair(source, target):

    if source["ontology_class"] != "Disease":
        return False

    if source["cui"] == target["cui"]:
        return False

    if normalize(target["canonical_name"]) in INVALID_GRAPH_ENTITIES:
        return False

    allowed = {
    "Symptom",
    "Sign",
    "WarningSign",
    "SevereManifestation",
    "Complication",
    "DiagnosticTest",
    "LaboratoryTest",
    "Treatment",
    "PreventionMethod",
    "Medication",
    "Pathogen",
    "MosquitoSpecies",
    "VirusSerotype",
    "Vector",
    "RiskFactor",
    "ClinicalPhase",
    "BodySystem",
    "PatientGroup",
    "GeographicRegion"
}

    return target["ontology_class"] in allowed

# ============================================================
# BUILD EDGES
# ============================================================

edges = []
seen = set()

print("Building sentence-level relationships...")

for chunk in chunks:

    chunk_id = chunk["chunk_id"]

    if chunk_id not in chunk_mentions:
        continue

    doc = nlp(chunk["text"])

    mentions_here = chunk_mentions[chunk_id]

    for sent in doc.sents:

        sentence = normalize(sent.text)

        if any(p in sentence for p in IGNORE_PHRASES):
            continue

        sentence_entities = []

        for m in mentions_here:

            mention = normalize(m["text"])

            if mention in sentence:
                sentence_entities.append(m)

        diseases = [
            e for e in sentence_entities
            if e["ontology_class"] == "Disease"
        ]

        for disease in diseases:

            for target in sentence_entities:

                if not valid_pair(disease, target):
                    continue

                pair = (
                    disease["ontology_class"],
                    target["ontology_class"]
                )

                if pair not in RELATION_PATTERNS:
                    continue

                relation, keywords = RELATION_PATTERNS[pair]

                # Require a trigger word
                if not any(k in sentence for k in keywords):
                    continue

                edge_key = (
                    disease["cui"],
                    target["cui"],
                    relation
                )

                if edge_key in seen:
                    continue

                seen.add(edge_key)

                edges.append({
                    "source": disease["cui"],
                    "target": target["cui"],
                    "relation": relation,
                    "chunk_id": chunk_id,
                    "sentence": sent.text.strip()
                })

# ============================================================
# SAVE GRAPH
# ============================================================

graph = {
    "metadata": {
        "nodes": len(nodes),
        "edges": len(edges),
        "method": "Sentence-level rule-based ontology graph"
    },
    "nodes": nodes,
    "edges": edges
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=4, ensure_ascii=False)

# ============================================================
# SUMMARY
# ============================================================

print("\nKnowledge Graph rebuilt successfully.")
print(f"Nodes : {len(nodes)}")
print(f"Edges : {len(edges)}")
print(f"Saved : {OUTPUT_FILE}")

# Relationship summary
relation_counts = defaultdict(int)

for edge in edges:
    relation_counts[edge["relation"]] += 1

print("\nRelationship Summary:")
for relation, count in sorted(relation_counts.items()):
    print(f"{relation}: {count}")