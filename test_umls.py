# ============================================================
# SCISPACY ENTITY LINKING TEST
# ============================================================

import os

# Avoid duplicate OpenMP problems on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

# Hide the sklearn version warning temporarily
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn"
)

import spacy
import scispacy

from scispacy.linking import EntityLinker


print("=" * 70)
print("SCISPACY ENTITY LINKING TEST")
print("=" * 70)


# ============================================================
# VERSION CHECK
# ============================================================

print("\nChecking versions...")

print("spaCy     :", spacy.__version__)
print("SciSpaCy  :", scispacy.__version__)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model: en_core_sci_sm")

nlp = spacy.load("en_core_sci_sm")

print("SciSpaCy model loaded.")


# ============================================================
# CHECK FACTORY
# ============================================================

print("\nChecking scispacy_linker factory...")

if "scispacy_linker" not in spacy.registry.factories:
    print("\nERROR:")
    print("scispacy_linker factory is NOT registered.")
    print("\nRegistered factories:")
    print(
        sorted(
            spacy.registry.factories.get_all().keys()
        )
    )

    raise RuntimeError(
        "scispacy_linker factory is not registered. "
        "Make sure 'import scispacy' is executed before "
        "adding the linker."
    )

print("scispacy_linker factory found.")


# ============================================================
# ADD UMLS LINKER
# ============================================================

print("\nLoading linker: umls")

try:

    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": False,
            "linker_name": "umls"
        }
    )

except Exception as e:

    print("\n" + "=" * 70)
    print("ERROR WHILE LOADING UMLS LINKER")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))

    print("\nPossible causes:")
    print("1. UMLS linker data is not cached.")
    print("2. Internet connection interrupted.")
    print("3. SciSpaCy linker resource download failed.")
    print("4. Cached TF-IDF files are incompatible/corrupted.")

    raise


print("UMLS linker loaded successfully.")


# ============================================================
# GET LINKER
# ============================================================

linker = nlp.get_pipe("scispacy_linker")


# ============================================================
# TEST TEXT
# ============================================================

text = (
    "Patients with dengue develop fever, headache, "
    "myalgia and nausea."
)

print("\n" + "=" * 70)
print("TEST TEXT")
print("=" * 70)

print(text)


# ============================================================
# RUN NLP
# ============================================================

print("\nRunning SciSpaCy...")

doc = nlp(text)


# ============================================================
# DISPLAY ENTITIES
# ============================================================

print("\n" + "=" * 70)
print("DETECTED ENTITIES")
print("=" * 70)

count = 0

for ent in doc.ents:

    print(f"\nEntity: {ent.text}")

    if not ent._.kb_ents:

        print("  No UMLS candidate found.")
        continue

    count += 1

    # Display top 3 candidates
    for index, (cui, score) in enumerate(
        ent._.kb_ents[:3],
        start=1
    ):

        concept = linker.kb.cui_to_entity[cui]

        print(f"  Candidate {index}")
        print(f"    CUI   : {cui}")
        print(f"    Name  : {concept.canonical_name}")
        print(f"    Score : {float(score):.3f}")
        print(f"    Types : {concept.types}")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print(
    f"Entities with UMLS candidates: {count}"
)

print("=" * 70)