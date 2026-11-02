
# ============================================================
# ONTOLOGY RELATION EXTRACTOR
# ============================================================
#
# Input:
#   ontology_entities.json
#   medical_chunks_source.json
#   ontology_schema.json
#
# Output:
#   medical_knowledge_graph.json
#
# Features:
#   - Sentence-level evidence
#   - Entity mention matching
#   - Ontology relationship validation
#   - Duplicate removal
#   - Source and chunk provenance
#   - Graph endpoint validation
#
# This is a general-purpose, pattern-based candidate extractor.
# It does not guarantee that extracted relationships are
# clinically correct.
# ============================================================

import json
import re
import time
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENTITIES_FILE = BASE_DIR / "ontology_entities.json"
CHUNKS_FILE = BASE_DIR / "medical_chunks_source.json"
ONTOLOGY_FILE = BASE_DIR / "ontology_schema.json"

OUTPUT_FILE = BASE_DIR / "medical_knowledge_graph.json"

# Keep the excluded source consistent with your project.
EXCLUDED_PATHS = {
    "kaggle/dengue.csv",
}

# Minimum UMLS linking score carried by both endpoints.
# Set to 0.0 to retain all scores produced by the entity extractor.
MIN_ENTITY_SCORE = 0.0

# Avoid connecting every entity in a long sentence.
MAX_ENTITY_DISTANCE_CHARS = 500

# Only use patterns that occur between the two entity mentions.
# The first entity is the source; the second is the target.
#
# Relationship names are matched against ontology_schema.json.
# If an equivalent allowed relationship cannot be found,
# the pattern is skipped rather than inventing a schema relation.
RELATION_PATTERNS = [
    {
        "aliases": [
            "HAS_SYMPTOM",
            "HAS_SIGN",
            "HAS_CLINICAL_MANIFESTATION",
        ],
        "patterns": [
            r"\bpresents with\b",
            r"\bsymptoms include\b",
            r"\bsigns include\b",
            r"\bcharacterized by\b",
            r"\bmanifest(?:s|ed|ing)? with\b",
        ],
    },
    {
        "aliases": [
            "CAUSES",
            "CAUSE",
            "LEADS_TO",
            "RESULTS_IN",
        ],
        "patterns": [
            r"\bcauses\b",
            r"\bleads to\b",
            r"\bresults in\b",
            r"\bcan cause\b",
            r"\bmay cause\b",
        ],
    },
    {
        "aliases": [
            "TREATED_WITH",
            "TREATMENT_FOR",
            "HAS_TREATMENT",
        ],
        "patterns": [
            r"\bis treated with\b",
            r"\bare treated with\b",
            r"\btreated using\b",
            r"\bmanagement includes\b",
            r"\btreatment includes\b",
            r"\btherapy includes\b",
        ],
    },
    {
        "aliases": [
            "DIAGNOSED_BY",
            "DETECTED_BY",
            "CONFIRMED_BY",
        ],
        "patterns": [
            r"\bdiagnosed by\b",
            r"\bdiagnosed using\b",
            r"\bdetected by\b",
            r"\bconfirmed by\b",
            r"\bidentified by\b",
        ],
    },
    {
        "aliases": [
            "TRANSMITTED_BY",
            "SPREAD_BY",
            "TRANSMITTED_THROUGH",
        ],
        "patterns": [
            r"\btransmitted by\b",
            r"\bspread by\b",
            r"\btransmitted through\b",
            r"\bspread through\b",
        ],
    },
    {
        "aliases": [
            "RISK_FACTOR",
            "HAS_RISK_FACTOR",
            "INCREASES_RISK_OF",
        ],
        "patterns": [
            r"\brisk factor for\b",
            r"\brisk factors include\b",
            r"\bincreases the risk of\b",
            r"\bincreased risk of\b",
        ],
    },
    {
        "aliases": [
            "PREVENTED_BY",
            "PREVENTION_METHOD",
            "PREVENTS",
        ],
        "patterns": [
            r"\bprevented by\b",
            r"\bprevention includes\b",
            r"\bprevented through\b",
        ],
    },
]

PROGRESS_EVERY = 50


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_path(value):
    return str(value or "unknown").replace("\\", "/").strip()


def is_excluded_path(path):
    normalized = normalize_path(path).lower()

    return any(
        normalized == excluded.lower()
        or normalized.endswith("/" + excluded.lower())
        for excluded in EXCLUDED_PATHS
    )


def get_chunk_id(chunk, index):
    value = chunk.get("chunk_id", chunk.get("id"))

    if value is None:
        return f"chunk::{index}"

    return str(value)


def get_chunk_text(chunk):
    for field in ("text", "chunk_text", "content", "page_content"):
        value = chunk.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_source_path(chunk):
    for field in (
        "document_path",
        "source_path",
        "file_path",
        "source",
        "file",
    ):
        value = chunk.get(field)

        if value:
            return normalize_path(value)

    return "unknown"


def normalize_relation_name(value):
    """
    Convert relationship labels to a comparable form.

    Examples:
        has symptom -> HAS_SYMPTOM
        Has-Symptom -> HAS_SYMPTOM
    """
    value = str(value or "").strip().upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


def normalize_entity_name(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def safe_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# LOAD INPUT DATA
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("LOADING ONTOLOGY AND ENTITY DATA")
    print("=" * 70)

    entity_data = load_json(ENTITIES_FILE)
    chunk_data = load_json(CHUNKS_FILE)
    ontology = load_json(ONTOLOGY_FILE)

    if not isinstance(entity_data, dict):
        raise ValueError("ontology_entities.json must contain a JSON object.")

    if not isinstance(chunk_data, list):
        raise ValueError("medical_chunks_source.json must contain a JSON list.")

    if not isinstance(ontology, dict):
        raise ValueError("ontology_schema.json must contain a JSON object.")

    allowed_relations_raw = ontology.get("relationship_types", [])

    if not isinstance(allowed_relations_raw, list):
        raise ValueError(
            "ontology_schema.json: relationship_types must be a list."
        )

    allowed_relations = {
        normalize_relation_name(value): value
        for value in allowed_relations_raw
        if isinstance(value, str) and value.strip()
    }

    print(f"Source chunks: {len(chunk_data)}")
    print(f"Ontology relationship types: {len(allowed_relations)}")

    entity_mentions = entity_data.get("entity_mentions", [])

    if not isinstance(entity_mentions, list):
        raise ValueError(
            "ontology_entities.json must contain an entity_mentions list."
        )

    unique_entities = entity_data.get("entities", [])

    if not isinstance(unique_entities, list):
        unique_entities = []

    print(f"Entity mentions: {len(entity_mentions)}")
    print(f"Unique entities: {len(unique_entities)}")

    # --------------------------------------------------------
    # Index the source chunks by their chunk IDs.
    # --------------------------------------------------------

    chunks_by_id = {}
    chunk_sources = {}

    stats = Counter()

    for index, chunk in enumerate(chunk_data):
        if not isinstance(chunk, dict):
            stats["invalid_chunks"] += 1
            continue

        chunk_id = get_chunk_id(chunk, index)
        source_path = get_source_path(chunk)

        if is_excluded_path(source_path):
            stats["excluded_chunks"] += 1
            continue

        text = get_chunk_text(chunk)

        if not text:
            stats["empty_chunks"] += 1
            continue

        if chunk_id in chunks_by_id:
            stats["duplicate_chunk_ids"] += 1
            continue

        chunks_by_id[chunk_id] = text
        chunk_sources[chunk_id] = source_path

    print(f"Usable chunks: {len(chunks_by_id)}")
    print(f"Excluded chunks: {stats['excluded_chunks']}")

    # --------------------------------------------------------
    # Create a canonical entity index.
    #
    # CUI is the primary identity where available.
    # --------------------------------------------------------

    entity_index = {}

    for entity in unique_entities:
        if not isinstance(entity, dict):
            continue

        cui = entity.get("cui")

        if not cui:
            continue

        cui = str(cui)

        entity_index[cui] = {
            "id": cui,
            "cui": cui,
            "name": entity.get("canonical_name") or cui,
            "ontology_class": entity.get("ontology_class", "Other"),
            "semantic_types": entity.get("semantic_types", []),
            "best_score": safe_score(entity.get("best_score", 0.0)),
        }

    # Add any entity that appears in mentions but is missing
    # from the unique-entity list.
    for mention in entity_mentions:
        if not isinstance(mention, dict):
            continue

        cui = mention.get("cui")

        if not cui:
            continue

        cui = str(cui)

        if cui not in entity_index:
            entity_index[cui] = {
                "id": cui,
                "cui": cui,
                "name": mention.get("canonical_name") or cui,
                "ontology_class": mention.get("ontology_class", "Other"),
                "semantic_types": mention.get("semantic_types", []),
                "best_score": safe_score(mention.get("score", 0.0)),
            }

    # --------------------------------------------------------
    # Build a lookup of mentions by chunk.
    # --------------------------------------------------------

    mentions_by_chunk = defaultdict(list)

    for mention in entity_mentions:
        if not isinstance(mention, dict):
            stats["invalid_entity_mentions"] += 1
            continue

        cui = mention.get("cui")
        chunk_id = mention.get("chunk_id")
        text = mention.get("text")

        if cui is None or chunk_id is None or not text:
            stats["incomplete_entity_mentions"] += 1
            continue

        cui = str(cui)
        chunk_id = str(chunk_id)

        if cui not in entity_index:
            stats["missing_entity_records"] += 1
            continue

        score = safe_score(mention.get("score", 0.0))

        if score < MIN_ENTITY_SCORE:
            stats["low_score_mentions"] += 1
            continue

        if chunk_id not in chunks_by_id:
            stats["mentions_without_source_chunk"] += 1
            continue

        mentions_by_chunk[chunk_id].append({
            "cui": cui,
            "text": str(text).strip(),
            "normalized_text": normalize_entity_name(text),
            "score": score,
            "ontology_class": mention.get(
                "ontology_class",
                entity_index[cui]["ontology_class"],
            ),
        })

    print(f"Entities indexed: {len(entity_index)}")
    print(f"Chunks with entity mentions: {len(mentions_by_chunk)}")

    # --------------------------------------------------------
    # Resolve configured relation names to the ontology schema.
    # --------------------------------------------------------

    resolved_patterns = []

    for definition in RELATION_PATTERNS:
        matched_relation = None

        for alias in definition["aliases"]:
            normalized_alias = normalize_relation_name(alias)

            if normalized_alias in allowed_relations:
                matched_relation = allowed_relations[normalized_alias]
                break

        if matched_relation is None:
            continue

        compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in definition["patterns"]
        ]

        resolved_patterns.append({
            "relation": matched_relation,
            "patterns": compiled_patterns,
        })

    if not resolved_patterns:
        print()
        print("WARNING: No configured relationship patterns matched")
        print("the relationship names in ontology_schema.json.")
        print("No relationships will be created until names are aligned.")
        print("Allowed relationships:")

        for relation in sorted(allowed_relations.values()):
            print(f"  - {relation}")

    else:
        print(f"Relationship pattern groups enabled: {len(resolved_patterns)}")

    # --------------------------------------------------------
    # Sentence splitting.
    #
    # This avoids creating a relationship across unrelated
    # sentences in the same chunk.
    # --------------------------------------------------------

    try:
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    except Exception as exc:
        raise RuntimeError(
            "Could not initialize sentence splitting."
        ) from exc

    # --------------------------------------------------------
    # Match entity mentions in each sentence.
    # --------------------------------------------------------

    def find_mentions_in_sentence(sentence_text, chunk_mentions):
        found = []

        for mention in chunk_mentions:
            mention_text = mention["text"]

            if not mention_text:
                continue

            # Escape the entity text to treat it literally.
            # Word boundaries are used for ordinary alphanumeric
            # names, while punctuation-containing names use a
            # flexible whitespace pattern.
            escaped = re.escape(mention_text)
            escaped = escaped.replace(r"\ ", r"\s+")

            pattern = re.compile(
                rf"(?<!\w){escaped}(?!\w)",
                flags=re.IGNORECASE,
            )

            for match in pattern.finditer(sentence_text):
                found.append({
                    **mention,
                    "start": match.start(),
                    "end": match.end(),
                    "surface": match.group(0),
                })

        # Prefer longer spans where two names overlap.
        found.sort(
            key=lambda item: (
                item["start"],
                -(item["end"] - item["start"]),
            )
        )

        non_overlapping = []

        for item in found:
            overlaps = any(
                item["start"] < existing["end"]
                and item["end"] > existing["start"]
                for existing in non_overlapping
            )

            if overlaps:
                continue

            non_overlapping.append(item)

        return sorted(non_overlapping, key=lambda item: item["start"])

    # --------------------------------------------------------
    # Search for a relationship phrase between ordered mentions.
    # --------------------------------------------------------

    def find_relation_between(sentence_text, first, second):
        if first["end"] > second["start"]:
            return None

        distance = second["start"] - first["end"]

        if distance > MAX_ENTITY_DISTANCE_CHARS:
            return None

        between = sentence_text[first["end"]:second["start"]]

        # Skip pairs where a sentence boundary or strong clause
        # boundary appears between the two entities.
        if re.search(r"[.!?;\n]", between):
            return None

        for definition in resolved_patterns:
            for pattern in definition["patterns"]:
                match = pattern.search(between)

                if match:
                    return {
                        "relation": definition["relation"],
                        "evidence_phrase": match.group(0),
                    }

        return None

    # --------------------------------------------------------
    # Create graph nodes.
    # --------------------------------------------------------

    nodes = []

    for cui in sorted(entity_index):
        entity = entity_index[cui]

        nodes.append({
            "id": entity["id"],
            "cui": entity["cui"],
            "label": entity["name"],
            "name": entity["name"],
            "type": "MedicalEntity",
            "ontology_class": entity["ontology_class"],
            "semantic_types": entity["semantic_types"],
            "best_score": entity["best_score"],
        })

    # --------------------------------------------------------
    # Extract candidate relationships.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTING RELATIONSHIPS")
    print("=" * 70)

    relationships = []
    seen_relationships = set()

    chunk_ids = list(chunks_by_id.keys())

    for index, chunk_id in enumerate(chunk_ids, start=1):
        if index == 1 or index % PROGRESS_EVERY == 0 or index == len(chunk_ids):
            print(f"Processed {index}/{len(chunk_ids)} chunks")

        text = chunks_by_id[chunk_id]
        source_path = chunk_sources[chunk_id]
        chunk_mentions = mentions_by_chunk.get(chunk_id, [])

        if not chunk_mentions:
            continue

        doc = nlp(text)

        for sentence_index, sentence in enumerate(doc.sents):
            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            stats["sentences_checked"] += 1

            sentence_mentions = find_mentions_in_sentence(
                sentence_text,
                chunk_mentions,
            )

            stats["matched_sentence_mentions"] += len(sentence_mentions)

            if len(sentence_mentions) < 2:
                continue

            # Consider pairs in their textual order.
            for first_index, first in enumerate(sentence_mentions):
                for second in sentence_mentions[first_index + 1:]:
                    if first["cui"] == second["cui"]:
                        continue

                    relation_match = find_relation_between(
                        sentence_text,
                        first,
                        second,
                    )

                    if relation_match is None:
                        continue

                    relation = relation_match["relation"]

                    key = (
                        chunk_id,
                        sentence_index,
                        first["cui"],
                        relation,
                        second["cui"],
                        sentence_text,
                    )

                    if key in seen_relationships:
                        stats["duplicate_relationships"] += 1
                        continue

                    seen_relationships.add(key)

                    relationships.append({
                        "id": f"rel::{len(relationships) + 1}",
                        "source": first["cui"],
                        "source_name": entity_index[first["cui"]]["name"],
                        "source_mention": first["surface"],
                        "relation": relation,
                        "target": second["cui"],
                        "target_name": entity_index[second["cui"]]["name"],
                        "target_mention": second["surface"],
                        "evidence_phrase": relation_match["evidence_phrase"],
                        "evidence_sentence": sentence_text,
                        "source_path": source_path,
                        "chunk_id": chunk_id,
                        "sentence_index": sentence_index,
                        "source_score": first["score"],
                        "target_score": second["score"],
                        "extraction_method": "sentence_pattern",
                        "review_status": "candidate",
                    })

                    stats["candidate_relationships"] += 1

    # --------------------------------------------------------
    # Validate graph endpoints and allowed relationship types.
    # --------------------------------------------------------

    node_ids = {node["id"] for node in nodes}
    valid_relationships = []

    for relationship in relationships:
        if relationship["source"] not in node_ids:
            stats["invalid_source_endpoints"] += 1
            continue

        if relationship["target"] not in node_ids:
            stats["invalid_target_endpoints"] += 1
            continue

        normalized_relation = normalize_relation_name(
            relationship["relation"]
        )

        if normalized_relation not in allowed_relations:
            stats["disallowed_relationship_types"] += 1
            continue

        valid_relationships.append(relationship)

    # --------------------------------------------------------
    # Count relation types.
    # --------------------------------------------------------

    relation_counts = Counter(
        relationship["relation"]
        for relationship in valid_relationships
    )

    entity_class_counts = Counter(
        node["ontology_class"]
        for node in nodes
    )

    stats["total_nodes"] = len(nodes)
    stats["total_relationships"] = len(valid_relationships)
    stats["unique_entities"] = len(nodes)
    stats["chunks_processed"] = len(chunks_by_id)

    # --------------------------------------------------------
    # Build output.
    # --------------------------------------------------------

    result = {
        "metadata": {
            "project": "Medical Ontology Knowledge Graph",
            "input_entities_file": ENTITIES_FILE.name,
            "input_chunks_file": CHUNKS_FILE.name,
            "ontology_file": ONTOLOGY_FILE.name,
            "excluded_paths": sorted(EXCLUDED_PATHS),
            "minimum_entity_score": MIN_ENTITY_SCORE,
            "maximum_entity_distance_chars": MAX_ENTITY_DISTANCE_CHARS,
            "extraction_method": "sentence-level pattern matching",
            "warning": (
                "Relationships are automatically extracted candidates. "
                "They are not clinically verified facts and should be "
                "reviewed before clinical use."
            ),
        },
        "statistics": dict(stats),
        "ontology_class_counts": dict(entity_class_counts),
        "relationship_counts": dict(relation_counts),
        "allowed_relationships": sorted(allowed_relations.values()),
        "nodes": nodes,
        "relationships": valid_relationships,
    }

    # --------------------------------------------------------
    # Save JSON.
    # --------------------------------------------------------

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("ONTOLOGY RELATION EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Unique entity nodes: {len(nodes)}")
    print(f"Candidate relationships: {len(valid_relationships)}")
    print(f"Sentences checked: {stats['sentences_checked']}")
    print(f"Relationship types found: {len(relation_counts)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Output: {OUTPUT_FILE}")

    if not valid_relationships:
        print()
        print("No relationships were extracted.")
        print("Check the relationship names in ontology_schema.json.")
        print("They must match the aliases configured in RELATION_PATTERNS.")

    print("=" * 70)


if __name__ == "__main__":
    main()
