
import json
import re
from pathlib import Path
from collections import Counter

import spacy
from scispacy.file_cache import DATASET_CACHE


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "medical_chunks_source.json"
OUTPUT_FILE = BASE_DIR / "medical_relations.json"

MODEL_NAME = "en_core_sci_sm"

EXCLUDED_PATHS = {
    "kaggle/dengue.csv",
    "medical_documents/kaggle/dengue.csv",
}

MAX_BETWEEN_CHARS = 120
MAX_RELATION_TO_ENDPOINT_CHARS = 65
MIN_LINKING_SCORE = 0.70
MAX_LINKED_MENTIONS_PER_SENTENCE = 18
MAX_NUMERIC_TOKEN_RATIO = 0.45


# ============================================================
# RELATION PATTERNS
# ============================================================

RELATION_PATTERNS = {
    "HAS_WARNING_SIGN": [
        r"\bwarning signs?\s+(?:include|are|such as)\b",
        r"\bwarning signs?\s+(?:may include|can include)\b",
    ],
    "HAS_COMPLICATION": [
        r"\bcomplications?\s+include\b",
        r"\bcomplicated by\b",
        r"\bmay be complicated by\b",
    ],
    "HAS_RISK_FACTOR": [
        r"\brisk factors?\s+(?:include|are)\b",
        r"\bincreases?\s+the risk of\b",
        r"\bat increased risk of\b",
        r"\bpredisposes?\s+to\b",
        r"\brisk factor for\b",
    ],
    "AFFECTS_BODY_SYSTEM": [
        r"\baffects?\b",
        r"\binvolves?\b",
        r"\baffecting\b",
        r"\binvolvement of\b",
    ],
    "HAS_LAB_FINDING": [
        r"\blaboratory findings?\s+include\b",
        r"\blab findings?\s+include\b",
        r"\bassociated laboratory findings?\s+include\b",
        r"\bshows? evidence of\b",
    ],
    "DETECTED_BY": [
        r"\bdetected by\b",
        r"\bdetected using\b",
        r"\bidentified by\b",
        r"\bidentified using\b",
        r"\bmeasured by\b",
        r"\bmeasured using\b",
    ],
    "DIAGNOSED_BY": [
        r"\bdiagnosed by\b",
        r"\bdiagnosed using\b",
        r"\bdiagnosis is confirmed by\b",
        r"\bconfirmed by\b",
        r"\bdiagnostic tests?\s+include\b",
        r"\bdiagnostic test is\b",
    ],
    "TRANSMITTED_BY": [
        r"\btransmitted by\b",
        r"\bspread by\b",
        r"\bcarried by\b",
    ],
    "HAS_TRANSMISSION_ROUTE": [
        r"\btransmitted through\b",
        r"\bspread through\b",
        r"\btransmitted via\b",
        r"\bspread via\b",
    ],
    "PREVENTED_BY": [
        r"\bprevented by\b",
        r"\bprevention includes\b",
        r"\bprevented through\b",
        r"\bcan be prevented by\b",
    ],
    "MANAGED_WITH": [
        r"\bmanaged with\b",
        r"\bmanagement includes\b",
        r"\bmanaged using\b",
        r"\bclinical management includes\b",
    ],
    "TREATED_WITH": [
        r"\bis treated with\b",
        r"\bare treated with\b",
        r"\btreated using\b",
        r"\btreatment includes\b",
        r"\btherapy includes\b",
        r"\btreated by\b",
    ],
    "REQUIRES_MONITORING": [
        r"\bmonitoring includes\b",
        r"\brequires? monitoring of\b",
        r"\bshould be monitored for\b",
    ],
    "CONTRAINDICATED_IN": [
        r"\bcontraindicated in\b",
        r"\bshould not be used in\b",
        r"\bmust be avoided in\b",
        r"\bavoid in patients with\b",
    ],
    "RECOMMENDED_FOR": [
        r"\brecommended for\b",
        r"\bindicated for\b",
        r"\bused for the treatment of\b",
    ],
    "INDICATES": [
        r"\bis indicative of\b",
        r"\bmay indicate\b",
        r"\bcan indicate\b",
        r"\bindicates?\b",
        r"\bsuggests?\b",
    ],
    "CAUSES": [
        r"\bcauses?\b",
        r"\bleads to\b",
        r"\bresults in\b",
        r"\bcontributes to\b",
    ],
    "CAUSED_BY": [
        r"\bcaused by\b",
        r"\bresults from\b",
        r"\bdue to\b",
        r"\bresulting from\b",
    ],
    "HAS_SYMPTOM": [
        r"\bsymptoms?\s+include\b",
        r"\bpresents with\b",
        r"\bpresenting with\b",
        r"\bmanifest(?:s|ed|ing)? with\b",
        r"\bcomplains of\b",
    ],
    "HAS_SIGN": [
        r"\bclinical signs?\s+include\b",
        r"\bsigns?\s+include\b",
        r"\bexamination reveals\b",
        r"\bphysical examination shows\b",
        r"\bcharacterized by\b",
    ],
    "ASSOCIATED_WITH": [
        r"\bassociated with\b",
        r"\bcorrelated with\b",
        r"\bcommonly seen with\b",
        r"\boccurs with\b",
        r"\blinked to\b",
    ],
    "AFFECTS_POPULATION": [
        r"\bcommonly affects\b",
        r"\bmainly affects\b",
        r"\bprimarily affects\b",
        r"\bmore common in\b",
        r"\bmost common among\b",
    ],
    "OCCURS_IN_REGION": [
        r"\bendemic in\b",
        r"\bprevalent in\b",
        r"\bcommonly occurs in\b",
        r"\breported in\b",
        r"\boccurs in\b",
    ],
    "HAS_SEVERITY": [
        r"\bseverity is classified as\b",
        r"\bseverity is classified into\b",
        r"\bclassified into severity grades?\b",
        r"\bclassified as\b",
        r"\bgraded as\b",
    ],
    "FOLLOWED_BY": [
        r"\bfollowed by\b",
        r"\bprogresses to\b",
        r"\bprogression to\b",
    ],
}


# ============================================================
# FILTER LISTS
# ============================================================

NEGATION_PATTERNS = [
    r"\bno evidence of\b",
    r"\bnot associated with\b",
    r"\bno association with\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bunlikely to\b",
    r"\babsence of\b",
    r"\bneither\b",
]

GENERIC_ENDPOINT_NAMES = {
    "case",
    "case (situation)",
    "data",
    "pair",
    "site",
    "location",
    "records",
    "publications",
    "dimensions",
    "levels",
    "achievement",
    "readiness",
    "completeness",
    "international",
    "locally",
    "items located",
    "system output",
    "collections (publication)",
    "collection (action)",
    "observation in research",
    "control act - action",
    "country",
    "decreasing",
    "progressive",
    "reduced",
    "aspects of signs",
    "increasing",
    "severe",
    "severity modifier",
    "fever qualifier",
    "unknown",
    "not specified",
}

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
    "response process",
    "specimen type",
}

INVALID_SYMPTOM_TARGETS = {
    "mmhg",
    "mm hg",
    "diastole",
    "systole",
    "japanese language",
    "achievement",
    "attention",
    "proportion",
    "policy",
    "manager",
    "persistent",
    "compensated",
    "assessed",
    "evaluation",
    "condition",
    "specimen type - serum",
    "bolus of ingested food",
}

GEOGRAPHIC_SEMANTIC_TYPES = {"T082", "T083"}
DISEASE_SEMANTIC_TYPES = {"T047", "T191"}


# ============================================================
# LOCAL-FIRST SCISPACY CACHE SUPPORT
# ============================================================

def enable_cached_scispacy_resources():
    import scispacy.candidate_generation as candidate_generation
    import scispacy.linking as scispacy_linking

    original_cached_path = candidate_generation.cached_path
    cache_dir = Path(DATASET_CACHE)

    metadata_suffixes = (
        ".npz.json",
        ".bin.json",
        ".joblib.json",
        ".json.json",
        ".jsonl.json",
        ".tsv.json",
    )

    def cached_path_with_local_first(url_or_filename, cache_dir_arg=None):
        if isinstance(url_or_filename, Path):
            url_or_filename = str(url_or_filename)

        if (
            isinstance(url_or_filename, str)
            and url_or_filename.startswith(("http://", "https://"))
        ):
            directory = (
                Path(cache_dir_arg)
                if cache_dir_arg is not None
                else cache_dir
            )

            if directory.exists():
                for metadata_path in directory.glob("*.json"):
                    if not metadata_path.name.endswith(metadata_suffixes):
                        continue

                    try:
                        with metadata_path.open(
                            "r", encoding="utf-8"
                        ) as file:
                            metadata = json.load(file)
                    except (OSError, ValueError):
                        continue

                    if not isinstance(metadata, dict):
                        continue

                    if metadata.get("url") != url_or_filename:
                        continue

                    resource_path = Path(str(metadata_path)[:-5])

                    if resource_path.is_file():
                        print(
                            "Using cached SciSpaCy resource:",
                            resource_path.name,
                        )
                        return str(resource_path)

        return original_cached_path(url_or_filename, cache_dir_arg)

    candidate_generation.cached_path = cached_path_with_local_first

    if hasattr(scispacy_linking, "cached_path"):
        scispacy_linking.cached_path = cached_path_with_local_first


# ============================================================
# INPUT HELPERS
# ============================================================

def load_chunks(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("chunks", "documents", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Could not find a list of chunks in medical_chunks_source.json"
    )


def get_chunk_text(chunk):
    for key in ("text", "chunk_text", "content", "page_content"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_source_path(chunk):
    for key in (
        "document_path",
        "source_path",
        "file_path",
        "source",
        "file",
    ):
        value = chunk.get(key)
        if value:
            return str(value).replace("\\", "/")
    return "unknown"


def is_excluded(source_path):
    normalized = str(source_path).replace("\\", "/").lower().strip()

    for excluded in EXCLUDED_PATHS:
        excluded = excluded.replace("\\", "/").lower().strip()
        if normalized == excluded or normalized.endswith("/" + excluded):
            return True

    return "dengue.csv" in normalized and "kaggle" in normalized


def stable_chunk_id(chunk, index, source_path):
    existing_id = chunk.get("chunk_id", chunk.get("id"))
    if existing_id is not None:
        return str(existing_id)
    return f"{source_path}::chunk::{index}"


# ============================================================
# ENTITY LINKING
# ============================================================

def get_linked_concept(ent, linker):
    candidates = ent._.kb_ents

    if not candidates:
        return None

    concept_id, score = candidates[0]

    try:
        kb_entity = linker.kb.cui_to_entity.get(concept_id)
    except Exception:
        kb_entity = None

    canonical_name = (
        getattr(kb_entity, "canonical_name", None)
        if kb_entity is not None
        else None
    )

    semantic_types = (
        list(getattr(kb_entity, "types", []) or [])
        if kb_entity is not None
        else []
    )

    return {
        "concept_id": str(concept_id),
        "canonical_name": canonical_name or ent.text.strip(),
        "score": float(score),
        "semantic_types": semantic_types,
    }


# ============================================================
# NORMALIZATION AND FILTERING
# ============================================================

def normalize_name(name):
    return " ".join(str(name or "").lower().strip().split())


def semantic_types_of(entity):
    return set(entity.get("semantic_types") or [])


def is_artifact_name(name):
    normalized = normalize_name(name)

    if not normalized or normalized in ARTIFACT_NAMES:
        return True

    if re.fullmatch(
        r"(page|figure|table|section|chapter)\s*[-:#]?\s*\d*",
        normalized,
    ):
        return True

    if normalized in {
        "severe",
        "progressive",
        "decreasing",
        "reduced",
        "increase",
        "decrease",
        "low",
        "high",
        "rapid",
    }:
        return True

    return False


def looks_like_table_or_list(sentence_text):
    text = sentence_text.strip()
    lower = text.lower()

    if not text:
        return True

    if re.search(r"\btable\s+\d+\b", lower):
        return True

    if re.search(r"\bfigure\s+\d+\b", lower):
        return True

    if re.search(r"^\s*(?:\d+[\.\)]\s+|[-•▪]\s+)", text):
        return True

    if re.search(r"\[page\s+\d+\]", lower):
        return True

    tokens = re.findall(r"\b[\w./%-]+\b", text)

    if tokens:
        numeric_tokens = sum(
            bool(re.search(r"\d", token)) for token in tokens
        )

        if numeric_tokens / len(tokens) > MAX_NUMERIC_TOKEN_RATIO:
            return True

    # Filter country-ranking fragments containing percentages.
    if (
        re.search(r"\bdeaths?\s*\(\s*\d+\s*%\s*\)", lower)
        or (
            re.search(r"\bfollowed by\b", lower)
            and re.search(r"\b\d+\s*%", lower)
        )
    ):
        return True

    return False


def has_negation_near_relation(sentence_text, between_text, phrase):
    """
    Conservative heuristic: only reject when a negation cue is near
    the matched phrase. This is not full linguistic scope detection.
    """
    between_lower = between_text.lower()
    phrase_lower = phrase.lower()
    position = between_lower.find(phrase_lower)

    if position >= 0:
        prefix = between_lower[max(0, position - 45):position]

        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, prefix, flags=re.IGNORECASE):
                return True

    # Check a short local context before the relation in the sentence.
    sentence_lower = sentence_text.lower()
    sentence_position = sentence_lower.find(phrase_lower)

    if sentence_position >= 0:
        prefix = sentence_lower[max(0, sentence_position - 45):sentence_position]

        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, prefix, flags=re.IGNORECASE):
                return True

    return False


def should_filter_entity_pair(relation_type, source, target):
    source_name = normalize_name(source["canonical_name"])
    target_name = normalize_name(target["canonical_name"])

    source_types = semantic_types_of(source)
    target_types = semantic_types_of(target)

    if not source_name or not target_name:
        return True

    if source["concept_id"] == target["concept_id"]:
        return True

    if (
        source_name in GENERIC_ENDPOINT_NAMES
        or target_name in GENERIC_ENDPOINT_NAMES
    ):
        return True

    if is_artifact_name(source_name) or is_artifact_name(target_name):
        return True

    clinical_relations = {
        "HAS_SYMPTOM",
        "HAS_SIGN",
        "HAS_WARNING_SIGN",
    }

    if relation_type in clinical_relations:
        if target_name in INVALID_SYMPTOM_TARGETS:
            return True

        if target_types.intersection(GEOGRAPHIC_SEMANTIC_TYPES):
            return True

    if relation_type == "OCCURS_IN_REGION":
        if source_types.intersection(GEOGRAPHIC_SEMANTIC_TYPES):
            return True

        # If types are available, require a geographic target.
        # If the linker provides no types, keep the candidate for review.
        if (
            target_types
            and not target_types.intersection(GEOGRAPHIC_SEMANTIC_TYPES)
        ):
            return True

    if relation_type == "HAS_SYMPTOM":
        if source_name in {
            "country",
            "honduras",
            "nicaragua",
            "mexico",
            "cambodia",
        }:
            return True

    if relation_type == "INDICATES":
        if source_name in {
            "male",
            "males",
            "female",
            "females",
            "decreasing",
            "decrease",
            "reduced",
            "progressive",
        }:
            return True

    if (
        relation_type == "HAS_COMPLICATION"
        and target_name in {
            "severe",
            "severe (severity modifier)",
            "hospital",
            "clinic",
            "healthcare setting",
            "accident and emergency department",
        }
    ):
        return True

    if relation_type == "HAS_SEVERITY":
        if (
            target_types
            and not target_types.intersection(DISEASE_SEMANTIC_TYPES)
        ):
            return True

    return False


# ============================================================
# RELATION MATCHING AND DIRECTION
# ============================================================

def find_relation(sentence_text, first_end, second_start):
    if first_end > second_start:
        return None

    between = sentence_text[first_end:second_start]

    if not between.strip():
        return None

    if len(between) > MAX_BETWEEN_CHARS:
        return None

    if re.search(r"[;\n]", between):
        return None

    if len(between.split()) > 20:
        return None

    matches = []

    for relation_type, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(
                pattern,
                between,
                flags=re.IGNORECASE,
            ):
                matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "length": len(match.group(0)),
                    "type": relation_type,
                    "phrase": match.group(0),
                })

    if not matches:
        return None

    # Earliest phrase wins; longer phrase wins when starts are equal.
    matches.sort(key=lambda item: (item["start"], -item["length"]))
    best = matches[0]

    left_distance = best["start"]
    right_distance = len(between) - best["end"]

    if (
        left_distance > MAX_RELATION_TO_ENDPOINT_CHARS
        or right_distance > MAX_RELATION_TO_ENDPOINT_CHARS
    ):
        return None

    return {
        "type": best["type"],
        "evidence_phrase": best["phrase"],
        "between_text": between,
        "sentence_text": sentence_text,
    }


def correct_relation_direction(relation_type, phrase):
    """
    Return (normalized_relation_type, reverse_endpoints).

    CAUSES and CAUSED_BY are normalized to CAUSES. The endpoints
    are reversed for passive/reverse causal constructions.
    """
    normalized_phrase = " ".join(phrase.lower().split())

    if relation_type == "CAUSED_BY":
        return "CAUSES", True

    if relation_type == "CAUSES" and normalized_phrase in {
        "caused by",
        "results from",
        "due to",
        "resulting from",
    }:
        return "CAUSES", True

    return relation_type, False


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("Loading source chunks...")
    chunks = load_chunks(INPUT_FILE)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading SciSpaCy model: {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

    if "scispacy_linker" not in nlp.pipe_names:
        print("Enabling local-first SciSpaCy cache lookup...")
        enable_cached_scispacy_resources()

        print("Loading UMLS entity linker...")
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "resolve_abbreviations": True,
            },
        )

    linker = nlp.get_pipe("scispacy_linker")

    relations = []
    seen_relations = set()
    relation_counts = Counter()

    stats = {
        "input_chunks": len(chunks),
        "excluded_chunks": 0,
        "empty_chunks": 0,
        "processed_chunks": 0,
        "sentences_checked": 0,
        "table_or_list_sentences_filtered": 0,
        "linked_entity_mentions": 0,
        "low_score_mentions_filtered": 0,
        "candidate_pairs_checked": 0,
        "too_distant_pairs_filtered": 0,
        "relation_phrase_not_found": 0,
        "negated_candidates_filtered": 0,
        "artifact_candidates_filtered": 0,
        "invalid_entity_pairs_filtered": 0,
        "duplicate_candidates_filtered": 0,
        "relations_found": 0,
    }

    print("Extracting sentence-level relationship candidates...")

    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        source_path = get_source_path(chunk)

        if is_excluded(source_path):
            stats["excluded_chunks"] += 1
            continue

        text = get_chunk_text(chunk)

        if not text:
            stats["empty_chunks"] += 1
            continue

        stats["processed_chunks"] += 1
        chunk_id = stable_chunk_id(chunk, chunk_index, source_path)

        document = nlp(text)

        for sentence_index, sentence in enumerate(document.sents):
            stats["sentences_checked"] += 1
            sentence_text = sentence.text.strip()

            if not sentence_text:
                continue

            if looks_like_table_or_list(sentence_text):
                stats["table_or_list_sentences_filtered"] += 1
                continue

            mentions = []

            for ent in sentence.ents:
                linked = get_linked_concept(ent, linker)

                if linked is None:
                    continue

                stats["linked_entity_mentions"] += 1

                if linked["score"] < MIN_LINKING_SCORE:
                    stats["low_score_mentions_filtered"] += 1
                    continue

                mentions.append({
                    **linked,
                    "mention": ent.text.strip(),
                    "start_char": ent.start_char - sentence.start_char,
                    "end_char": ent.end_char - sentence.start_char,
                })

            mentions.sort(key=lambda item: item["start_char"])

            if len(mentions) > MAX_LINKED_MENTIONS_PER_SENTENCE:
                stats["table_or_list_sentences_filtered"] += 1
                continue

            # Only adjacent linked mentions are considered.
            for index in range(len(mentions) - 1):
                first = mentions[index]
                second = mentions[index + 1]

                stats["candidate_pairs_checked"] += 1

                between_length = second["start_char"] - first["end_char"]

                if (
                    between_length < 0
                    or between_length > MAX_BETWEEN_CHARS
                ):
                    stats["too_distant_pairs_filtered"] += 1
                    continue

                relation_match = find_relation(
                    sentence_text,
                    first["end_char"],
                    second["start_char"],
                )

                if relation_match is None:
                    stats["relation_phrase_not_found"] += 1
                    continue

                relation_type = relation_match["type"]
                phrase = relation_match["evidence_phrase"]

                if has_negation_near_relation(
                    sentence_text,
                    relation_match["between_text"],
                    phrase,
                ):
                    stats["negated_candidates_filtered"] += 1
                    continue

                source = first
                target = second

                relation_type, reverse_direction = correct_relation_direction(
                    relation_type,
                    phrase,
                )

                if reverse_direction:
                    source, target = target, source

                if should_filter_entity_pair(
                    relation_type,
                    source,
                    target,
                ):
                    stats["invalid_entity_pairs_filtered"] += 1
                    continue

                relation_key = (
                    chunk_id,
                    sentence_index,
                    source["concept_id"],
                    relation_type,
                    target["concept_id"],
                    normalize_name(phrase),
                )

                if relation_key in seen_relations:
                    stats["duplicate_candidates_filtered"] += 1
                    continue

                seen_relations.add(relation_key)

                relations.append({
                    "source_id": source["concept_id"],
                    "source_name": source["canonical_name"],
                    "source_mention": source["mention"],
                    "relation": relation_type,
                    "target_id": target["concept_id"],
                    "target_name": target["canonical_name"],
                    "target_mention": target["mention"],
                    "evidence_phrase": phrase,
                    "evidence_sentence": sentence_text,
                    "source_path": source_path,
                    "chunk_id": chunk_id,
                    "sentence_index": sentence_index,
                    "source_linking_score": source["score"],
                    "target_linking_score": target["score"],
                    "candidate_status": "requires_medical_validation",
                    "relation_confidence": None,
                })

                relation_counts[relation_type] += 1

    stats["relations_found"] = len(relations)

    output = {
        "metadata": {
            "input_file": INPUT_FILE.name,
            "excluded_paths": sorted(EXCLUDED_PATHS),
            "model": MODEL_NAME,
            "min_linking_score": MIN_LINKING_SCORE,
            "max_between_chars": MAX_BETWEEN_CHARS,
            "max_relation_to_endpoint_chars": (
                MAX_RELATION_TO_ENDPOINT_CHARS
            ),
            "relation_patterns": RELATION_PATTERNS,
            "note": (
                "Rule-based relation candidates with sentence-level "
                "evidence. Linking scores measure entity-linking strength, "
                "not relationship correctness. Filters are heuristic. "
                "Validate candidates before using them as medical facts."
            ),
        },
        "statistics": stats,
        "relationship_type_counts": dict(relation_counts),
        "relations": relations,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\nExtraction completed.")
    print(f"Relations found: {stats['relations_found']}")
    print(f"Excluded chunks: {stats['excluded_chunks']}")

    print("\nRelationship counts:")
    for relation_type, count in sorted(relation_counts.items()):
        print(f"  {relation_type}: {count}")

    print("\nDiagnostics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
