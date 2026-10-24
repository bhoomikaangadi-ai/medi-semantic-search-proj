# ============================================================
# ONTOLOGY ENTITY EXTRACTOR
# ============================================================
#
# GENERAL-PURPOSE MEDICAL ONTOLOGY PIPELINE
#
# Pipeline:
#
# Medical chunks
#       ↓
# SciSpaCy
#       ↓
# UMLS Entity Linking
#       ↓
# Multiple Candidate Validation
#       ↓
# Entity Noise Filtering
#       ↓
# Ontology Classification
#       ↓
# Duplicate Removal
#       ↓
# Chunk Mapping
#       ↓
# JSON Output
#
# This is NOT dengue-specific.
#
# It can process:
#
#   Dengue
#   Diabetes
#   Malaria
#   Pneumonia
#   Cancer
#   Hypertension
#   Tuberculosis
#   etc.
#
# ============================================================


# ============================================================
# ENVIRONMENT
# ============================================================

import os

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "TRUE"

os.environ[
    "OMP_NUM_THREADS"
] = "1"


# ============================================================
# IMPORTS
# ============================================================

import json
import time

from collections import (
    Counter,
    defaultdict
)

import spacy

# IMPORTANT:
# This import registers the SciSpaCy linker factory.
from scispacy.linking import EntityLinker

from entity_validator import (
    validate_entity,
    is_generic_term,
    is_document_noise
)

from entity_filter import (
    clean_concept
)


# ============================================================
# CONFIGURATION
# ============================================================

ONTOLOGY_FILE = (
    "ontology_schema.json"
)

CHUNKS_FILE = (
    "medical_chunks_source.json"
)

OUTPUT_FILE = (
    "ontology_entities.json"
)


# ============================================================
# TEST MODE
# ============================================================
#
# None:
#     Process ALL chunks.
#
# Example:
#
# TEST_CHUNK_ID = 88
#
# processes only chunk 88.
#
# ============================================================

TEST_CHUNK_ID = None


# ============================================================
# SCISPACY CONFIGURATION
# ============================================================

MODEL_NAME = (
    "en_core_sci_sm"
)

LINKER_NAME = (
    "umls"
)


# ============================================================
# UMLS VALIDATION
# ============================================================

MIN_LINKING_SCORE = 0.80

MAX_CANDIDATES = 5


# ============================================================
# PROGRESS
# ============================================================

PROGRESS_EVERY = 10


# ============================================================
# STATISTICS
# ============================================================

statistics = {

    "chunks_total": 0,

    "chunks_processed": 0,

    "chunks_empty": 0,

    "chunks_failed": 0,

    "scispacy_mentions": 0,

    "generic_noise": 0,

    "document_noise": 0,

    "no_umls_candidate": 0,

    "weak_linking_score": 0,

    "mention_canonical_mismatch": 0,

    "outside_ontology": 0,

    "accepted_mentions": 0,

    "unique_entities": 0
}


# ============================================================
# LOAD ONTOLOGY
# ============================================================

print()
print("=" * 70)
print("LOADING ONTOLOGY")
print("=" * 70)


with open(
    ONTOLOGY_FILE,
    "r",
    encoding="utf-8"
) as f:

    ontology = json.load(f)


# ============================================================
# ENTITY CLASSES
# ============================================================

ENTITY_CLASSES = ontology.get(
    "entity_classes",
    {}
)


# ============================================================
# RELATIONSHIPS
# ============================================================

ALLOWED_RELATIONS = set(
    ontology.get(
        "relationship_types",
        []
    )
)


print(
    f"Ontology entity classes: "
    f"{len(ENTITY_CLASSES)}"
)

print(
    f"Ontology relationships: "
    f"{len(ALLOWED_RELATIONS)}"
)


# ============================================================
# DISPLAY ONTOLOGY
# ============================================================

print()
print("Ontology classes:")

for (
    class_name,
    semantic_types
) in ENTITY_CLASSES.items():

    print(
        f"  {class_name} "
        f"-> {semantic_types}"
    )


# ============================================================
# LOAD SCISPACY
# ============================================================

print()
print("=" * 70)
print("LOADING SCISPACY")
print("=" * 70)

print(
    f"Loading model: "
    f"{MODEL_NAME}"
)


nlp = spacy.load(
    MODEL_NAME
)


print(
    "SciSpaCy model loaded."
)


# ============================================================
# CHECK SCISPACY LINKER FACTORY
# ============================================================

print()
print(
    "Checking scispacy_linker factory..."
)


if (
    "scispacy_linker"
    not in nlp.factories
):

    raise RuntimeError(
        "\n"
        "scispacy_linker factory is not registered.\n\n"
        "Make sure this import exists:\n\n"
        "from scispacy.linking import EntityLinker\n"
    )


print(
    "scispacy_linker factory found."
)


# ============================================================
# LOAD UMLS LINKER
# ============================================================

print()
print(
    f"Loading linker: "
    f"{LINKER_NAME}"
)


try:

    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": LINKER_NAME
        }
    )

except Exception as e:

    print()
    print("=" * 70)
    print("ERROR WHILE LOADING UMLS LINKER")
    print("=" * 70)

    print(
        f"Error type: "
        f"{type(e).__name__}"
    )

    print(
        f"Error: "
        f"{e}"
    )

    print()
    print(
        "The SciSpaCy UMLS resource could not be loaded."
    )

    print(
        "Your network/resource cache must be available "
        "the first time the UMLS linker is initialized."
    )

    raise


# ============================================================
# GET LINKER
# ============================================================

linker = nlp.get_pipe(
    "scispacy_linker"
)


print()
print(
    "UMLS linker loaded successfully."
)


# ============================================================
# LOAD MEDICAL CHUNKS
# ============================================================

print()
print("=" * 70)
print("LOADING MEDICAL CHUNKS")
print("=" * 70)


with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


if not isinstance(
    chunks,
    list
):

    raise ValueError(
        "medical_chunks_source.json "
        "must contain a list."
    )


statistics[
    "chunks_total"
] = len(chunks)


print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# SELECT CHUNKS
# ============================================================

if TEST_CHUNK_ID is not None:

    selected_chunks = [

        chunk

        for chunk in chunks

        if chunk.get(
            "chunk_id"
        ) == TEST_CHUNK_ID
    ]

    if not selected_chunks:

        raise ValueError(
            f"Chunk {TEST_CHUNK_ID} "
            f"not found."
        )

    print()
    print(
        "TEST MODE"
    )

    print(
        f"Processing chunk "
        f"{TEST_CHUNK_ID}"
    )

else:

    selected_chunks = chunks

    print()
    print(
        "FULL DATASET MODE"
    )

    print(
        f"Processing all "
        f"{len(selected_chunks)} chunks."
    )


# ============================================================
# ONTOLOGY CLASSIFICATION
# ============================================================
CUSTOM_ENTITY_CLASSES = {
    "dengue shock syndrome": "Complication",
    "severe dengue": "Complication",
    "dengue haemorrhagic fever": "Complication",

    "platelet": "LaboratoryTest",
    "hematocrit": "LaboratoryTest",
    "full blood count": "LaboratoryTest",
    "ns1": "DiagnosticTest",
    "elisa": "DiagnosticTest",
    "pcr": "DiagnosticTest",

    "mosquito net": "PreventionMethod",
    "repellent": "PreventionMethod",
    "vector control": "PreventionMethod",

    "aedes aegypti": "MosquitoSpecies",
    "aedes albopictus": "MosquitoSpecies",

    "tropical": "GeographicRegion",
    "subtropical": "GeographicRegion"
}
def classify_entity(entity):
    """
    Map UMLS semantic types to the ontology.

    IMPORTANT:
    WarningSign and SevereManifestation overlap with
    ordinary Sign/Symptom/Disease semantic types.

    Therefore this function intentionally uses the
    general/base ontology class.

    Contextual classification such as:

        dengue
          |
          HAS_WARNING_SIGN
          |
          bleeding

    should happen during relation extraction.

    Returns:
        ontology class
        or "Other"
    """
    
    name = entity["canonical_name"].lower()

# Match WHO-specific terms using substring matching
    for term, ontology_class in CUSTOM_ENTITY_CLASSES.items():
        if term in name:
             return ontology_class
    semantic_types = set(
        entity.get(
            "semantic_types",
            []
        )
    )


    # --------------------------------------------------------
    # IMPORTANT CLASSIFICATION ORDER
    # --------------------------------------------------------
    #
    # More general/base classes are selected first.
    #
    # This prevents:
    #
    # T047 -> SevereManifestation
    #
    # from incorrectly turning every disease into a
    # severe manifestation.
    #
    # --------------------------------------------------------

    class_priority = [
    "Disease",
    "Complication",
    "Symptom",
    "Sign",
    "LaboratoryTest",
    "DiagnosticTest",
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
]

    for class_name in class_priority:

        type_codes = set(
            ENTITY_CLASSES.get(
                class_name,
                []
            )
        )

        if semantic_types.intersection(
            type_codes
        ):

            return class_name


    return "Other"


# ============================================================
# SELECT BEST UMLS CANDIDATE
# ============================================================

def select_best_umls_candidate(
    mention,
    candidates
):
    """
    Examine multiple UMLS candidates.

    Returns:

        (cui, score, concept)

    or:

        None
    """

    if not candidates:

        return None


    # --------------------------------------------------------
    # Examine candidates in ranking order
    # --------------------------------------------------------

    for (
        cui,
        raw_score
    ) in candidates:

        try:

            score = float(
                raw_score
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        if (
            score
            < MIN_LINKING_SCORE
        ):

            continue


        # ----------------------------------------------------
        # CUI existence
        # ----------------------------------------------------

        if (
            cui
            not in linker.kb.cui_to_entity
        ):

            continue


        concept = (
            linker.kb.cui_to_entity[
                cui
            ]
        )


        canonical_name = (
            concept.canonical_name
        )


        # ----------------------------------------------------
        # Validate candidate
        # ----------------------------------------------------

        try:

            valid = validate_entity(

                mention=mention,

                canonical_name=(
                    canonical_name
                ),

                score=score,

                min_score=(
                    MIN_LINKING_SCORE
                )
            )

        except Exception:

            valid = False


        if not valid:

            continue


        # ----------------------------------------------------
        # Candidate accepted
        # ----------------------------------------------------

        return (
            cui,
            score,
            concept
        )


    return None


# ============================================================
# GLOBAL STORAGE
# ============================================================

all_entity_mentions = []

all_unique_entities = {}

entity_to_chunks = defaultdict(
    set
)

entity_to_mentions = defaultdict(
    set
)


# ============================================================
# ONTOLOGY COUNTS
# ============================================================

ontology_class_counts = Counter()


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# PROCESS CHUNKS
# ============================================================

print()
print("=" * 70)
print("PROCESSING MEDICAL CHUNKS")
print("=" * 70)


for (
    index,
    chunk
) in enumerate(
    selected_chunks,
    start=1
):


    # ========================================================
    # CHUNK INFORMATION
    # ========================================================

    chunk_id = chunk.get(
        "chunk_id"
    )

    text = chunk.get(
        "text",
        ""
    )


    # ========================================================
    # EMPTY CHUNK
    # ========================================================

    if not isinstance(
        text,
        str
    ):

        statistics[
            "chunks_empty"
        ] += 1

        continue


    text = text.strip()


    if not text:

        statistics[
            "chunks_empty"
        ] += 1

        continue


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index == 1
        or index % PROGRESS_EVERY == 0
        or index == len(selected_chunks)
    ):

        print(
            f"[{index}/"
            f"{len(selected_chunks)}] "
            f"Processing chunk "
            f"{chunk_id}..."
        )


    # ========================================================
    # RUN SCISPACY
    # ========================================================

    try:

        doc = nlp(
            text
        )

    except Exception as e:

        statistics[
            "chunks_failed"
        ] += 1

        print()

        print(
            f"WARNING: Failed "
            f"chunk {chunk_id}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        continue


    statistics[
        "chunks_processed"
    ] += 1


    # ========================================================
    # ENTITY EXTRACTION
    # ========================================================

    for ent in doc.ents:


        statistics[
            "scispacy_mentions"
        ] += 1


        # ----------------------------------------------------
        # Mention
        # ----------------------------------------------------

        mention = (
            ent.text.strip()
        )


        if not mention:

            continue


        # ----------------------------------------------------
        # Generic term
        # ----------------------------------------------------

        try:

            if is_generic_term(
                mention
            ):

                statistics[
                    "generic_noise"
                ] += 1

                continue

        except Exception:

            pass


        # ----------------------------------------------------
        # Document noise
        # ----------------------------------------------------

        try:

            if is_document_noise(
                mention
            ):

                statistics[
                    "document_noise"
                ] += 1

                continue

        except Exception:

            pass


        # ----------------------------------------------------
        # UMLS candidates
        # ----------------------------------------------------

        if not ent._.kb_ents:

            statistics[
                "no_umls_candidate"
            ] += 1

            continue


        # ----------------------------------------------------
        # Candidate list
        # ----------------------------------------------------

        candidates = (
            ent._.kb_ents[
                :MAX_CANDIDATES
            ]
        )


        # ----------------------------------------------------
        # Select valid candidate
        # ----------------------------------------------------

        selected_candidate = (
            select_best_umls_candidate(
                mention,
                candidates
            )
        )


        # ====================================================
        # NO VALID CANDIDATE
        # ====================================================

        if selected_candidate is None:


            best_cui, best_score = (
                candidates[0]
            )


            try:

                best_score = float(
                    best_score
                )

            except (
                TypeError,
                ValueError
            ):

                best_score = 0.0


            if (
                best_score
                < MIN_LINKING_SCORE
            ):

                statistics[
                    "weak_linking_score"
                ] += 1

            else:

                statistics[
                    "mention_canonical_mismatch"
                ] += 1


            continue


        # ====================================================
        # VALID CANDIDATE
        # ====================================================

        (
            cui,
            score,
            concept
        ) = selected_candidate


        # ====================================================
        # BASIC FILTER
        # ====================================================

        concept_for_filter = {

            "name": mention
        }


        try:

            clean_result = (
                clean_concept(
                    concept_for_filter
                )
            )

        except Exception:

            clean_result = True


        if not clean_result:

            statistics[
                "generic_noise"
            ] += 1

            continue


        # ====================================================
        # CREATE ENTITY
        # ====================================================

        entity = {

            "text":
                mention,

            "cui":
                cui,

            "canonical_name":
                concept.canonical_name,

            "semantic_types":
                list(concept.types),

            "score":
                float(score)
        }


        # ====================================================
        # ONTOLOGY CLASSIFICATION
        # ====================================================

        ontology_class = (
            classify_entity(
                entity
            )
        )


        entity[
            "ontology_class"
        ] = ontology_class


        # ====================================================
        # OUTSIDE ONTOLOGY
        # ====================================================

        if (
            ontology_class
            == "Other"
        ):

            statistics[
                "outside_ontology"
            ] += 1

            continue


        # ====================================================
        # CHUNK ID
        # ====================================================

        entity[
            "chunk_id"
        ] = chunk_id


        # ====================================================
        # ACCEPT ENTITY
        # ========================================================

        all_entity_mentions.append(
            entity
        )


        statistics[
            "accepted_mentions"
        ] += 1


        # ====================================================
        # ONTOLOGY COUNT
        # ====================================================

        ontology_class_counts[
            ontology_class
        ] += 1


        # ====================================================
        # UNIQUE ENTITY KEY
        # ====================================================

        entity_key = (

            cui,

            concept.canonical_name,

            ontology_class
        )


        # ====================================================
        # CREATE UNIQUE ENTITY
        # ====================================================

        if (
            entity_key
            not in all_unique_entities
        ):

            all_unique_entities[
                entity_key
            ] = {

                "cui":
                    cui,

                "canonical_name":
                    concept.canonical_name,

                "ontology_class":
                    ontology_class,

                "semantic_types":
                    list(
                        concept.types
                    ),

                "best_score":
                    float(score),

                "mentions":
                    [],

                "chunk_ids":
                    []
            }


        # ====================================================
        # TRACK MENTION
        # ====================================================

        entity_to_mentions[
            entity_key
        ].add(
            mention
        )


        # ====================================================
        # TRACK CHUNK
        # ====================================================

        entity_to_chunks[
            entity_key
        ].add(
            chunk_id
        )


        # ====================================================
        # UPDATE BEST SCORE
        # ====================================================

        if (
            score
            >
            all_unique_entities[
                entity_key
            ][
                "best_score"
            ]
        ):

            all_unique_entities[
                entity_key
            ][
                "best_score"
            ] = float(
                score
            )


# ============================================================
# FINALIZE UNIQUE ENTITIES
# ============================================================

for (
    entity_key,
    entity
) in all_unique_entities.items():

    entity[
        "mentions"
    ] = sorted(
        entity_to_mentions[
            entity_key
        ]
    )

    entity[
        "chunk_ids"
    ] = sorted(
        entity_to_chunks[
            entity_key
        ]
    )


statistics[
    "unique_entities"
] = len(
    all_unique_entities
)


# ============================================================
# PROCESSING TIME
# ============================================================

elapsed_time = (
    time.time()
    - start_time
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(
    f"Chunks processed: "
    f"{statistics['chunks_processed']}"
)

print(
    f"SciSpaCy mentions: "
    f"{statistics['scispacy_mentions']}"
)

print(
    f"Accepted mentions: "
    f"{statistics['accepted_mentions']}"
)

print(
    f"Unique medical entities: "
    f"{statistics['unique_entities']}"
)

print(
    f"Processing time: "
    f"{elapsed_time:.2f} seconds"
)
if statistics["scispacy_mentions"] > 0:
    acceptance_rate = (
        statistics["accepted_mentions"]
        / statistics["scispacy_mentions"]
    ) * 100
else:
    acceptance_rate = 0.0

print(
    f"Entity acceptance rate: "
    f"{acceptance_rate:.2f}%"
)

# ============================================================
# ONTOLOGY COUNTS
# ============================================================

print()
print("=" * 70)
print("ONTOLOGY CLASS COUNTS")
print("=" * 70)


if ontology_class_counts:

    for (
        class_name,
        count
    ) in sorted(
        ontology_class_counts.items()
    ):

        print(
            f"{class_name}: "
            f"{count}"
        )

else:

    print(
        "No ontology entities found."
    )


# ============================================================
# VALIDATION STATISTICS
# ============================================================

print()
print("=" * 70)
print("VALIDATION STATISTICS")
print("=" * 70)


for (
    key,
    value
) in statistics.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# UNIQUE ENTITY LIST
# ============================================================

unique_entities_list = list(
    all_unique_entities.values()
)


unique_entities_list.sort(
    key=lambda x: (
        x[
            "ontology_class"
        ],
        x[
            "canonical_name"
        ].lower()
    )
)


# ============================================================
# CHUNK → ENTITIES
# ============================================================

chunk_entities = defaultdict(
    list
)


for entity in all_entity_mentions:

    chunk_id = entity[
        "chunk_id"
    ]

    chunk_entities[
        chunk_id
    ].append(
        entity
    )


chunk_entities = {

    str(chunk_id): entities

    for (
        chunk_id,
        entities
    )
    in chunk_entities.items()
}


# ============================================================
# FINAL RESULT
# ============================================================

result = {

    "metadata": {

        "ontology_file":
            ONTOLOGY_FILE,

        "chunks_file":
            CHUNKS_FILE,

        "model":
            MODEL_NAME,

        "linker":
            LINKER_NAME,

        "min_linking_score":
            MIN_LINKING_SCORE,

        "max_candidates":
            MAX_CANDIDATES,

        "test_chunk_id":
            TEST_CHUNK_ID,

        "total_chunks":
            len(selected_chunks),

        "processed_chunks":
            statistics[
                "chunks_processed"
            ],

               "unique_entities":
            statistics[
                "unique_entities"
            ],

        "entity_acceptance_rate":
            round(
                acceptance_rate,
                2
            )
        
    },
        


    # --------------------------------------------------------
    # UNIQUE MEDICAL CONCEPTS
    # --------------------------------------------------------

    "entities":
        unique_entities_list,


    # --------------------------------------------------------
    # MENTION LEVEL
    # --------------------------------------------------------

    "entity_mentions":
        all_entity_mentions,


    # --------------------------------------------------------
    # CHUNK → ENTITY
    # --------------------------------------------------------

    "chunk_entities":
        chunk_entities,


    # --------------------------------------------------------
    # ONTOLOGY COUNTS
    # --------------------------------------------------------

    "ontology_class_counts":
        dict(
            ontology_class_counts
        ),


    # --------------------------------------------------------
    # ALLOWED RELATIONSHIPS
    # --------------------------------------------------------

    "allowed_relationships":
        sorted(
            ALLOWED_RELATIONS
        ),


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    "statistics":
        statistics
}


# ============================================================
# SAVE JSON
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ONTOLOGY ENTITY EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Total chunks: "
    f"{len(selected_chunks)}"
)

print(
    f"Processed chunks: "
    f"{statistics['chunks_processed']}"
)

print(
    f"SciSpaCy mentions: "
    f"{statistics['scispacy_mentions']}"
)

print(
    f"Accepted entity mentions: "
    f"{statistics['accepted_mentions']}"
)

print(
    f"Unique medical entities: "
    f"{statistics['unique_entities']}"
)

print(
    f"Output file: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)