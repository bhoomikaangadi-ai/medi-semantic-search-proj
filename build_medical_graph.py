
import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\Projects\Medical-Graph-RAG-PROJ")

ENTITIES_FILE = BASE_DIR / "ontology_entities.json"
RELATIONS_FILE = BASE_DIR / "medical_relations.json"
CHUNKS_FILE = BASE_DIR / "medical_chunks_source.json"
OUTPUT_FILE = BASE_DIR / "medical_knowledge_graph.json"

EXCLUDED_PATHS = (
    "kaggle/dengue.csv",
    "medical_documents/kaggle/dengue.csv",
)

ALLOWED_RELATION_TYPES = {
    "HAS_WARNING_SIGN",
    "HAS_COMPLICATION",
    "HAS_RISK_FACTOR",
    "AFFECTS_BODY_SYSTEM",
    "HAS_LAB_FINDING",
    "DETECTED_BY",
    "DIAGNOSED_BY",
    "TRANSMITTED_BY",
    "HAS_TRANSMISSION_ROUTE",
    "PREVENTED_BY",
    "MANAGED_WITH",
    "TREATED_WITH",
    "REQUIRES_MONITORING",
    "CONTRAINDICATED_IN",
    "RECOMMENDED_FOR",
    "INDICATES",
    "CAUSES",
    "HAS_SYMPTOM",
    "HAS_SIGN",
    "ASSOCIATED_WITH",
    "AFFECTS_POPULATION",
    "OCCURS_IN_REGION",
    "HAS_SEVERITY",
    "FOLLOWED_BY",
}

# These are obvious document artifacts or overly generic terms.
# They are excluded only when encountered as standalone concepts.
ARTIFACT_NAMES = {
    "chapter",
    "page",
    "figure",
    "table",
    "section",
    "appendix",
    "highest",
    "lowest",
    "beginning",
    "ending",
    "north",
    "south",
    "east",
    "west",
    "evaluation",
    "assessment",
    "manager",
    "persistent",
    "policy",
    "attention",
    "evidence",
    "scientific study",
    "numerical phases",
    "levels",
    "response process",
    "specimen type",
}

# These concepts are commonly too generic to be useful as
# symptom/sign targets without additional context.
INVALID_SYMPTOM_NAMES = {
    "mmhg",
    "mm hg",
    "diastole",
    "systole",
    "highest",
    "lowest",
    "evaluation",
    "assessment",
    "levels",
    "manager",
    "policy",
    "attention",
    "persistent",
    "beginning",
    "ending",
    "chapter",
    "page",
}

# Explicit, conservative classification for recovered nodes.
# Review these mappings against your ontology_schema.json.
CLASS_OVERRIDES = {
    "accident and emergency department": "HealthcareSetting",
    "blood cells": "BodySystem",
    "shock": "Complication",
    "hospitalization": "HealthcareSetting",
    "culicidae": "MosquitoSpecies",
    "disease outbreaks": "EpidemiologicalEvent",
    "cessation of life": "Complication",
    "prophylactic treatment": "PreventionMethod",
    "endothelial cells": "BodySystem",
    "complement activation": "BodySystem",
    "patients": "PatientGroup",
    "woman": "PatientGroup",
    "hemorrhage": "Complication",
    "plasma": "BodySystem",
    "pulse pressure": "Sign",
    "rickettsia": "Pathogen",
    "rickettsia infections": "Disease",
    "intravenous fluid": "Treatment",
    "bolus of ingested food": "Unknown",
    "january": "Unknown",
    "north": "Unknown",
    "highest": "Unknown",
    "chapter": "Unknown",
    
    # Clinical signs and complications
    "oliguria": "Sign",
    "emergency situation": "Complication",
    "risk": "RiskFactor",
    "severities": "Unknown",
    "cautionary warning": "Sign",

    # Treatments and prevention
    "therapeutic hemodilution": "Treatment",
    "insecticides": "PreventionMethod",
    "pyrethroid resistance": "RiskFactor",
    "physical activity": "PreventionMethod",

    # Diseases and biological concepts
    "coronary arteriosclerosis": "Disease",
    "chronobiology disorders": "Disease",
    "disease": "Disease",
    "active immunity": "BodySystem",
    "antibody formation": "BodySystem",
    "memory b cells": "BodySystem",
    "antigens": "Pathogen",
    "antibodies, neutralizing": "BodySystem",
    "cross reactions": "Sign",
    "cell growth": "BodySystem",
    "pathogenicity aspects": "Pathogen",

    # Patient groups
    "males": "PatientGroup",
    "persons": "PatientGroup",
    "animals": "PatientGroup",
    "control groups": "PatientGroup",

    # Laboratory and diagnostic methods
    "reverse transcription": "LaboratoryTest",
    "nested polymerase chain reaction": "LaboratoryTest",
    "probes": "LaboratoryTest",
    "optical density units": "LaboratoryTest",
    "spectrophotometers": "LaboratoryTest",

    # Environmental and epidemiological concepts
    "home environment": "RiskFactor",
    "habitat": "RiskFactor",
    "spraying behavior": "PreventionMethod",
    "waste products": "Unknown",
    "sentinel": "Unknown",

    # Healthcare and public health
    "public health medicine (field)": "Unknown",
    "health communication": "PreventionMethod",
    "education, dental": "Unknown",
    "world health organization": "Unknown",

    # Other concepts that need contextual review
    "case (situation)": "Unknown",
    "dihydrofolic acid binding": "Unknown",
    "alcam protein, human": "Unknown",
    "system output": "Unknown",
    "nasogastric tube": "Unknown",
    "dimensions": "Unknown",
    "collections (publication)": "Unknown",
    "musca domestica (invertebrate)": "Unknown",
    "items located": "Unknown",
    "achievement": "Unknown",
    "locally": "Unknown",
    "levels (qualifier value)": "Unknown",
    "pair": "Unknown",
    "acute": "Unknown",
    "publications": "Unknown",
    "readiness": "Unknown",
    "data": "Unknown",
    "completeness": "Unknown",
    "density": "Unknown",
    "collection (action)": "Unknown",
    "site": "Unknown",
    "records": "Unknown",
    "location": "Unknown",
    "control act - action": "Unknown",
    "movement": "Unknown",
    "international": "Unknown",
    "observation in research": "Unknown",
     
# Additional provisional classifications
"nasogastric tube": "Treatment",
"musca domestica (invertebrate)": "Unknown",
"public health medicine (field)": "Unknown",
"world health organization": "Unknown",
"education, dental": "Unknown",
"resistance (psychotherapeutic)": "Unknown",
"acute": "Unknown",
"case (situation)": "Unknown",
"dihydrofolic acid binding": "Unknown",
"alcam protein, human": "Unknown",
"system output": "Unknown",
"dimensions": "Unknown",
"collections (publication)": "Unknown",
"items located": "Unknown",
"waste products": "Unknown",
"achievement": "Unknown",
"locally": "Unknown",
"levels (qualifier value)": "Unknown",
"pair": "Unknown",
"publications": "Unknown",
"readiness": "Unknown",
"data": "Unknown",
"completeness": "Unknown",
"density": "Unknown",
"sentinel": "Unknown",
"computing methodologies": "Unknown",
"collection (action)": "Unknown",
"site": "Unknown",
"records": "Unknown",
"location": "Unknown",
"control act - action": "Unknown",
"movement": "Unknown",
"international": "Unknown",
"observation in research": "Unknown",
"severities": "Unknown",

}

# Names with these classes are not automatically removed merely
# because their classes are not present in the original ontology.
# This lets the graph retain concepts that need later review.


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def normalize(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def get_cui(item):
    return str(item.get("cui") or item.get("id") or "").strip()


def get_entity_name(item):
    value = (
        item.get("canonical_name")
        or item.get("name")
        or item.get("text")
        or ""
    )
    return str(value).strip()


def get_entity_class(item):
    return (
        item.get("ontology_class")
        or item.get("class")
        or "Unknown"
    )


def get_list(data, *keys):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def get_chunk_id(item):
    value = item.get("chunk_id", item.get("id"))
    return None if value is None else str(value).strip()


def get_chunks(data):
    chunks = get_list(data, "chunks", "data", "documents")

    if not chunks:
        raise ValueError(
            "Could not find a chunk list in medical_chunks_source.json"
        )

    return chunks


def chunk_source_text(chunk):
    fields = (
        "source",
        "source_file",
        "file_path",
        "filename",
        "document",
        "document_name",
        "document_path",
    )

    values = [
        str(chunk.get(key, ""))
        for key in fields
        if chunk.get(key)
    ]

    return " ".join(values).replace("\\", "/").lower()


def is_excluded_path(path_text):
    path_text = str(path_text or "").replace("\\", "/").lower()

    if path_text.endswith("kaggle/dengue.csv"):
        return True

    return any(excluded in path_text for excluded in EXCLUDED_PATHS)


def is_excluded_chunk(chunk):
    return is_excluded_path(chunk_source_text(chunk))


# ============================================================
# ENTITY QUALITY AND CLASSIFICATION
# ============================================================

def is_artifact_name(name):
    normalized = normalize(name)

    if not normalized:
        return True

    if normalized in ARTIFACT_NAMES:
        return True

    # Remove standalone page/figure/table labels such as "Page 1".
    if re.fullmatch(
        r"(page|figure|table|section|chapter)\s*[-:#]?\s*\d*",
        normalized,
    ):
        return True

    return False


def classify_entity(name, existing_class="Unknown"):
    normalized = normalize(name)

    if normalized in CLASS_OVERRIDES:
        return CLASS_OVERRIDES[normalized]

    if existing_class and existing_class != "Unknown":
        return existing_class

    return "Unknown"


def is_invalid_symptom(name):
    normalized = normalize(name)

    if normalized in INVALID_SYMPTOM_NAMES:
        return True

    # Measurement units alone are not symptoms.
    if re.fullmatch(r"\d+(\.\d+)?\s*(mmhg|mg|ml|cm|kg)", normalized):
        return True

    return False


# ============================================================
# BUILD ENTITY NODES
# ============================================================

def build_nodes(entity_data):
    entities = get_list(entity_data, "entities")
    nodes_by_cui = {}
    diagnostics = Counter()

    for entity in entities:
        if not isinstance(entity, dict):
            diagnostics["invalid_entity_records"] += 1
            continue

        cui = get_cui(entity)
        name = get_entity_name(entity)

        if not cui or not name:
            diagnostics["entities_missing_id_or_name"] += 1
            continue

        if is_artifact_name(name):
            diagnostics["artifact_entities_skipped"] += 1
            continue

        ontology_class = classify_entity(
            name,
            get_entity_class(entity),
        )

        if cui not in nodes_by_cui:
            nodes_by_cui[cui] = {
                "id": cui,
                "cui": cui,
                "name": name,
                "canonical_name": name,
                "ontology_class": ontology_class,
                "semantic_types": entity.get("semantic_types", []),
                "best_score": entity.get("best_score"),
                "mentions": [],
                "chunk_ids": [],
                "node_origin": "ontology_entities",
            }

        node = nodes_by_cui[cui]

        for mention in entity.get("mentions", []):
            if mention and mention not in node["mentions"]:
                node["mentions"].append(mention)

        if name not in node["mentions"]:
            node["mentions"].append(name)

        for chunk_id in entity.get("chunk_ids", []):
            chunk_id = str(chunk_id)

            if chunk_id not in node["chunk_ids"]:
                node["chunk_ids"].append(chunk_id)

        # Keep the first meaningful class if duplicates disagree.
        if node["ontology_class"] == "Unknown" and ontology_class != "Unknown":
            node["ontology_class"] = ontology_class

    return nodes_by_cui, diagnostics


# ============================================================
# CHUNK LOOKUP
# ============================================================

def build_chunk_lookup(chunk_data):
    chunk_lookup = {}

    for chunk in get_chunks(chunk_data):
        if not isinstance(chunk, dict):
            continue

        chunk_id = get_chunk_id(chunk)

        if chunk_id is not None:
            chunk_lookup[chunk_id] = chunk

    return chunk_lookup


# ============================================================
# RECOVER MISSING RELATION ENDPOINTS
# ============================================================

def add_relation_endpoint(
    nodes_by_cui,
    cui,
    name,
    relation_item,
    endpoint,
    diagnostics,
):
    cui = str(cui or "").strip()
    name = str(name or "").strip()

    if not cui or not name:
        diagnostics["missing_endpoint_details"] += 1
        return False

    if cui in nodes_by_cui:
        return True

    ontology_class = classify_entity(name, "Unknown")
    mention = str(
        relation_item.get(f"{endpoint}_mention") or name
    ).strip()

    chunk_id = relation_item.get("chunk_id")
    chunk_ids = [] if chunk_id is None else [str(chunk_id)]

    nodes_by_cui[cui] = {
        "id": cui,
        "cui": cui,
        "name": name,
        "canonical_name": name,
        "ontology_class": ontology_class,
        "semantic_types": [],
        "best_score": relation_item.get(
            f"{endpoint}_linking_score"
        ),
        "mentions": list(dict.fromkeys([mention, name])),
        "chunk_ids": chunk_ids,
        "node_origin": "medical_relations",
        "requires_entity_validation": True,
    }

    diagnostics[f"{endpoint}_nodes_added"] += 1

    if ontology_class == "Unknown":
        diagnostics["recovered_nodes_unknown_class"] += 1

    return True


# ============================================================
# BUILD RELATIONSHIP EDGES
# ============================================================

def build_edges(relations_data, nodes_by_cui, chunk_lookup):
    relations = get_list(relations_data, "relations")

    edges_by_key = {}
    diagnostics = Counter()
    relationship_counts = Counter()
    rejected_relation_counts = Counter()

    for relation_item in relations:
        if not isinstance(relation_item, dict):
            diagnostics["invalid_relation_records"] += 1
            continue

        diagnostics["candidates_loaded"] += 1

        source_id = str(relation_item.get("source_id") or "").strip()
        target_id = str(relation_item.get("target_id") or "").strip()
        relation_type = str(
            relation_item.get("relation") or ""
        ).strip().upper()

        if not source_id or not target_id or not relation_type:
            diagnostics["missing_fields"] += 1
            continue

        if relation_type not in ALLOWED_RELATION_TYPES:
            diagnostics["unsupported_relation_type"] += 1
            rejected_relation_counts[relation_type] += 1
            continue

        if source_id == target_id:
            diagnostics["self_relationships_skipped"] += 1
            continue

        source_name = (
            relation_item.get("source_name")
            or relation_item.get("source_mention")
            or source_id
        )
        target_name = (
            relation_item.get("target_name")
            or relation_item.get("target_mention")
            or target_id
        )

        # Avoid obvious document artifacts as relationship endpoints.
        # Do not remove concepts merely because they are unfamiliar.
        if is_artifact_name(source_name) or is_artifact_name(target_name):
            diagnostics["artifact_relationships_filtered"] += 1
            continue

        if (
            relation_type in {"HAS_SYMPTOM", "HAS_SIGN", "HAS_WARNING_SIGN"}
            and is_invalid_symptom(target_name)
        ):
            diagnostics["invalid_symptom_candidates_filtered"] += 1
            continue

        chunk_id = relation_item.get("chunk_id")
        chunk_id = (
            None if chunk_id is None else str(chunk_id).strip()
        )

        chunk = (
            chunk_lookup.get(chunk_id)
            if chunk_id is not None
            else None
        )

        relation_path = str(
            relation_item.get("source_path") or ""
        )

        if is_excluded_path(relation_path):
            diagnostics["excluded_source_candidates"] += 1
            continue

        if chunk is not None and is_excluded_chunk(chunk):
            diagnostics["excluded_source_candidates"] += 1
            continue

        if source_id not in nodes_by_cui:
            add_relation_endpoint(
                nodes_by_cui,
                source_id,
                source_name,
                relation_item,
                "source",
                diagnostics,
            )

        if target_id not in nodes_by_cui:
            add_relation_endpoint(
                nodes_by_cui,
                target_id,
                target_name,
                relation_item,
                "target",
                diagnostics,
            )

        if source_id not in nodes_by_cui or target_id not in nodes_by_cui:
            diagnostics["candidates_unresolved"] += 1
            continue

        evidence = {
            "chunk_id": chunk_id,
            "sentence": str(
                relation_item.get("evidence_sentence") or ""
            ).strip(),
            "phrase": str(
                relation_item.get("evidence_phrase") or ""
            ).strip(),
            "source_path": relation_path,
            "source_linking_score": relation_item.get(
                "source_linking_score"
            ),
            "target_linking_score": relation_item.get(
                "target_linking_score"
            ),
        }

        key = (source_id, relation_type, target_id)

        if key not in edges_by_key:
            edges_by_key[key] = {
                "id": f"{source_id}|{relation_type}|{target_id}",
                "source": source_id,
                "target": target_id,
                "relation": relation_type,
                "evidence": [],
                "confidence": None,
            }

        edge = edges_by_key[key]

        if evidence not in edge["evidence"]:
            edge["evidence"].append(evidence)

        scores = [
            score
            for score in (
                relation_item.get("source_linking_score"),
                relation_item.get("target_linking_score"),
            )
            if isinstance(score, (int, float))
        ]

        if scores:
            candidate_score = min(scores)
            current_score = edge["confidence"]

            edge["confidence"] = (
                candidate_score
                if current_score is None
                else min(current_score, candidate_score)
            )

        diagnostics["candidates_linked"] += 1

    edges = list(edges_by_key.values())

    for edge in edges:
        relationship_counts[edge["relation"]] += 1

    diagnostics["unique_edges"] = len(edges)

    return (
        edges,
        diagnostics,
        relationship_counts,
        rejected_relation_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("MEDICAL KNOWLEDGE GRAPH BUILDER")
    print("=" * 65)

    entity_data = load_json(ENTITIES_FILE)
    relations_data = load_json(RELATIONS_FILE)
    chunk_data = load_json(CHUNKS_FILE)

    nodes_by_cui, node_diagnostics = build_nodes(entity_data)
    original_node_count = len(nodes_by_cui)

    chunk_lookup = build_chunk_lookup(chunk_data)

    (
        edges,
        edge_diagnostics,
        relationship_counts,
        rejected_relation_counts,
    ) = build_edges(
        relations_data,
        nodes_by_cui,
        chunk_lookup,
    )

    diagnostics = Counter(node_diagnostics)
    diagnostics.update(edge_diagnostics)

    nodes = list(nodes_by_cui.values())

    node_class_counts = Counter(
        node["ontology_class"] for node in nodes
    )

    added_nodes = sum(
        1 for node in nodes
        if node.get("node_origin") == "medical_relations"
    )

    graph = {
        "metadata": {
            "project": "Medical Graph RAG",
            "entity_source": ENTITIES_FILE.name,
            "relationship_source": RELATIONS_FILE.name,
            "chunk_source": CHUNKS_FILE.name,
            "description": (
                "Medical knowledge graph with ontology-linked entities "
                "and evidence-backed relationship candidates."
            ),
            "relationship_note": (
                "Pattern-extracted relationships are candidates and "
                "must be medically validated before being treated as "
                "confirmed facts."
            ),
            "classification_note": (
                "Recovered nodes marked requires_entity_validation "
                "should be reviewed before production use. Unknown "
                "classes indicate that no reliable mapping was found."
            ),
        },
        "statistics": {
            "total_nodes": len(nodes),
            "original_entity_nodes": original_node_count,
            "nodes_added_from_relations": added_nodes,
            "total_edges": len(edges),
            "node_class_counts": dict(node_class_counts),
            "relationship_counts": dict(relationship_counts),
            "edge_build_diagnostics": dict(diagnostics),
            "rejected_relation_types": dict(rejected_relation_counts),
        },
        "nodes": nodes,
        "edges": edges,
    }

    save_json(OUTPUT_FILE, graph)

    print(f"\nOriginal entity nodes: {original_node_count}")
    print(f"Total graph nodes: {len(nodes)}")
    print(f"Nodes added from relations: {added_nodes}")
    print(
        "Relationship candidates: "
        f"{len(get_list(relations_data, 'relations'))}"
    )
    print(f"Unique relationship edges: {len(edges)}")

    print("\nDiagnostics:")
    for key, value in sorted(diagnostics.items()):
        print(f"  {key}: {value}")

    print("\nRelationship types:")
    if relationship_counts:
        for relation_type, count in sorted(relationship_counts.items()):
            print(f"  {relation_type}: {count}")
    else:
        print("  No edges were created.")

    print("\nNode classes:")
    for class_name, count in sorted(node_class_counts.items()):
        print(f"  {class_name}: {count}")

    if rejected_relation_counts:
        print("\nRejected relation types:")
        for relation_type, count in sorted(
            rejected_relation_counts.items()
        ):
            print(f"  {relation_type}: {count}")

    print(f"\nSaved graph to:\n{OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
