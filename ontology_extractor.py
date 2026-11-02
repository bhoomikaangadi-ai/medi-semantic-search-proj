# ============================================================
# MEDICAL ONTOLOGY GRAPH EXTRACTOR
# ============================================================
# Pipeline:
# Medical chunks -> SciSpaCy -> MeSH linking
# -> DocumentChunk nodes + MedicalConcept nodes
# -> MENTIONS relationships -> medical_graph.json
#
# General-purpose medical extraction, not dengue-specific.
# Excludes only kaggle/dengue.csv.
# ============================================================

import json
from pathlib import Path

import spacy
from scispacy.linking import EntityLinker  # Registers linker factory


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

CHUNKS_FILE = ROOT / "medical_chunks_source.json"
OUTPUT_FILE = ROOT / "medical_graph.json"

MODEL_NAME = "en_core_sci_sm"
LINKER_NAME = "mesh"

# Set to 0.0 to keep all returned MeSH candidates.
# Increase to filter lower-confidence matches.
MIN_LINKING_SCORE = 0.0

PROGRESS_EVERY = 50

EXCLUDED_DOCUMENT_PATHS = {
    "kaggle/dengue.csv",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_path(value):
    """Normalize a document path for consistent comparisons."""
    return str(value or "").replace("\\", "/").strip().lower().lstrip("./")


def get_document_path(chunk):
    """Get a stable path identifying the source document."""
    path = normalize_path(chunk.get("document_path"))

    if path:
        return path

    source = normalize_path(chunk.get("source", "unknown"))
    document = normalize_path(chunk.get("document", "unknown"))

    return f"{source}/{document}"


def is_excluded_document(chunk):
    """Exclude the specified synthetic dengue dataset only."""
    path = get_document_path(chunk)

    return any(
        path == excluded or path.endswith("/" + excluded)
        for excluded in EXCLUDED_DOCUMENT_PATHS
    )


def make_chunk_node_id(chunk):
    """Create a unique node ID for every document chunk."""
    source = str(chunk.get("source", "unknown"))
    document_path = get_document_path(chunk)
    chunk_id = str(chunk.get("chunk_id", "unknown"))

    return f"chunk::{source}::{document_path}::{chunk_id}"


def make_concept_node_id(linker_name, concept_id):
    """Namespace concept IDs to avoid collisions with chunk IDs."""
    return f"{linker_name}::{concept_id}"


def get_semantic_types(concept):
    """Safely convert semantic types into JSON-serializable values."""
    return list(getattr(concept, "types", []) or [])


# ============================================================
# LOAD CHUNKS
# ============================================================

print("\n" + "=" * 65)
print("LOADING MEDICAL CHUNKS")
print("=" * 65)

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_FILE}"
    )

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    all_chunks = json.load(file)

if not isinstance(all_chunks, list):
    raise ValueError(
        "medical_chunks_source.json must contain a JSON list."
    )

chunks = []
excluded_count = 0
empty_count = 0

for chunk in all_chunks:
    if not isinstance(chunk, dict):
        continue

    if is_excluded_document(chunk):
        excluded_count += 1
        continue

    text = chunk.get("text")

    if not isinstance(text, str) or not text.strip():
        empty_count += 1
        continue

    chunks.append(chunk)

print("Source chunks:", len(all_chunks))
print("Excluded dengue.csv chunks:", excluded_count)
print("Empty chunks skipped:", empty_count)
print("Chunks selected:", len(chunks))

if not chunks:
    raise RuntimeError("No valid medical chunks were found.")


# ============================================================
# LOAD SCISPACY AND MESH LINKER
# ============================================================

print("\n" + "=" * 65)
print("LOADING SCISPACY")
print("=" * 65)

print("Model:", MODEL_NAME)

nlp = spacy.load(MODEL_NAME)

print("Loading linker:", LINKER_NAME)

try:
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": LINKER_NAME,
        },
    )
except Exception as exc:
    raise RuntimeError(
        f"Could not initialize the {LINKER_NAME} linker. "
        "Check your SciSpaCy installation and linker resources."
    ) from exc

linker = nlp.get_pipe("scispacy_linker")

print("SciSpaCy model and linker loaded successfully.")


# ============================================================
# GRAPH STORAGE
# ============================================================

chunk_nodes = {}
concept_nodes = {}
relationships = []

chunks_with_entities = 0
total_mentions = 0
low_score_mentions = 0
unresolved_mentions = 0
processing_failures = 0


# ============================================================
# PROCESS CHUNKS
# ============================================================

print("\n" + "=" * 65)
print("EXTRACTING MEDICAL CONCEPTS")
print("=" * 65)

for index, chunk in enumerate(chunks, start=1):
    text = chunk["text"].strip()

    source = str(chunk.get("source", "unknown"))
    document = str(chunk.get("document", "unknown"))
    document_path = get_document_path(chunk)
    original_chunk_id = chunk.get("chunk_id", "unknown")

    chunk_node_id = make_chunk_node_id(chunk)

    # Create one node per source chunk.
    chunk_nodes[chunk_node_id] = {
        "id": chunk_node_id,
        "name": f"{document} - chunk {original_chunk_id}",
        "node_type": "DocumentChunk",
        "source": source,
        "document": document,
        "document_path": document_path,
        "chunk_id": original_chunk_id,
        "text": text,
    }

    try:
        doc = nlp(text)
    except Exception as exc:
        processing_failures += 1
        print(
            f"WARNING: Failed to process chunk "
            f"{original_chunk_id}: {exc}"
        )
        continue

    chunk_has_entity = False

    for ent in doc.ents:
        mention_text = ent.text.strip()

        if not mention_text or not ent._.kb_ents:
            unresolved_mentions += 1
            continue

        # Examine the linker candidates in ranking order.
        selected = None

        for candidate_id, raw_score in ent._.kb_ents:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if score < MIN_LINKING_SCORE:
                continue

            concept = linker.kb.cui_to_entity.get(candidate_id)

            if concept is None:
                continue

            selected = (candidate_id, score, concept)
            break

        if selected is None:
            low_score_mentions += 1
            continue

        concept_id, score, concept = selected
        concept_node_id = make_concept_node_id(
            LINKER_NAME,
            concept_id,
        )

        # Create each concept only once.
        if concept_node_id not in concept_nodes:
            concept_nodes[concept_node_id] = {
                "id": concept_node_id,
                "name": concept.canonical_name,
                "concept_id": concept_id,
                "linker": LINKER_NAME,
                "semantic_type": get_semantic_types(concept),
                "node_type": "MedicalConcept",
            }

        # Preserve the source mention and its character offsets.
        relationships.append({
            "source": chunk_node_id,
            "target": concept_node_id,
            "type": "MENTIONS",
            "score": score,
            "mention_text": mention_text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "source_name": source,
            "document": document,
            "document_path": document_path,
            "chunk_id": original_chunk_id,
        })

        total_mentions += 1
        chunk_has_entity = True

    if chunk_has_entity:
        chunks_with_entities += 1

    if (
        index % PROGRESS_EVERY == 0
        or index == len(chunks)
    ):
        print(f"Processed {index}/{len(chunks)} chunks")


# ============================================================
# BUILD GRAPH
# ============================================================

all_nodes = list(chunk_nodes.values()) + list(concept_nodes.values())

# Validate node IDs and relationship endpoints.
node_ids = {node["id"] for node in all_nodes}

if len(node_ids) != len(all_nodes):
    raise RuntimeError("Duplicate node IDs detected.")

invalid_relationships = [
    rel for rel in relationships
    if (
        rel["source"] not in node_ids
        or rel["target"] not in node_ids
    )
]

if invalid_relationships:
    raise RuntimeError(
        f"{len(invalid_relationships)} relationships have "
        "missing source or target nodes."
    )

statistics = {
    "source_chunks": len(all_chunks),
    "selected_chunks": len(chunks),
    "excluded_dengue_csv_chunks": excluded_count,
    "empty_chunks_skipped": empty_count,
    "processing_failures": processing_failures,
    "unresolved_mentions": unresolved_mentions,
    "low_score_mentions": low_score_mentions,
    "chunk_nodes": len(chunk_nodes),
    "unique_medical_concepts": len(concept_nodes),
    "chunks_with_entities": chunks_with_entities,
    "mention_relationships": len(relationships),
    "total_nodes": len(all_nodes),
    "total_relationships": len(relationships),
    "linker": LINKER_NAME,
    "minimum_linking_score": MIN_LINKING_SCORE,
}

graph = {
    "metadata": {
        "name": "Medical Knowledge Graph",
        "method": "SciSpaCy entity linking with MeSH",
        "relationship_semantics": (
            "MENTIONS edges link document chunks to extracted "
            "medical concepts; they are not clinical causal relations."
        ),
    },
    "nodes": all_nodes,
    "relationships": relationships,
    "statistics": statistics,
}


# ============================================================
# SAVE GRAPH
# ============================================================

print("\nSaving graph...")

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        graph,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MEDICAL ONTOLOGY GRAPH CREATED")
print("=" * 65)

for key, value in statistics.items():
    print(f"{key}: {value}")

print("Output:", OUTPUT_FILE)
print("=" * 65)