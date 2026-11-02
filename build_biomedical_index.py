
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

CHUNKS_FILE = ROOT / "medical_chunks_source.json"

# Separate output files to protect existing indexes.
EMBEDDINGS_FILE = ROOT / "medical_clinical_embeddings.npy"
INDEX_FILE = ROOT / "medical_clinical_faiss.index"
CLINICAL_CHUNKS_FILE = ROOT / "medical_clinical_chunks.json"

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Exclude only this exact dataset from clinical retrieval.
EXCLUDED_DOCUMENT_PATHS = {
    "kaggle/dengue.csv"
}

MIN_CHUNK_LENGTH = 100


# ============================================================
# LOAD SOURCE CHUNKS
# ============================================================

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Source chunks file not found: {CHUNKS_FILE}"
    )

with open(CHUNKS_FILE, encoding="utf-8") as f:
    all_chunks = json.load(f)

if not isinstance(all_chunks, list):
    raise ValueError(
        "medical_chunks_source.json must contain a JSON list."
    )

print("Total source chunks:", len(all_chunks))


# ============================================================
# FILTER CLINICAL CHUNKS
# ============================================================

chunks = []

excluded_count = 0
short_count = 0

for chunk in all_chunks:
    document_path = str(
        chunk.get("document_path", "")
    ).replace("\\", "/").lower()

    # Use document_path when available.
    # Fall back to source/document for older records.
    if not document_path:
        source = str(chunk.get("source", "")).lower()
        document = Path(
            str(chunk.get("document", ""))
        ).name.lower()

        document_path = f"{source}/{document}"

    if document_path in EXCLUDED_DOCUMENT_PATHS:
        excluded_count += 1
        continue

    text = str(chunk.get("text", "")).strip()

    if len(text) < MIN_CHUNK_LENGTH:
        short_count += 1
        continue

    chunks.append(chunk)


# ============================================================
# VALIDATE FILTERING
# ============================================================

if not chunks:
    raise RuntimeError(
        "No clinical chunks found. Check the source JSON."
    )

remaining_excluded = [
    chunk for chunk in chunks
    if str(
        chunk.get("document_path", "")
    ).replace("\\", "/").lower()
    in EXCLUDED_DOCUMENT_PATHS
]

if remaining_excluded:
    raise RuntimeError(
        "Excluded dengue.csv records were found in clinical chunks."
    )

texts = [
    str(chunk["text"]).strip()
    for chunk in chunks
]

print("Excluded dengue.csv chunks:", excluded_count)
print("Short chunks skipped:", short_count)
print("Chunks selected for clinical search:", len(chunks))


# ============================================================
# LOAD BIOMEDICAL MODEL
# ============================================================

print("\nLoading biomedical retrieval model...")
print("Model:", MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating passage embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype("float32")


# ============================================================
# BUILD FAISS INDEX
# ============================================================

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# ============================================================
# SAVE OUTPUTS
# ============================================================

# Save the matching embeddings and index.
np.save(EMBEDDINGS_FILE, embeddings)

faiss.write_index(
    index,
    str(INDEX_FILE)
)

# Save the exact chunk order used to build the index.
with open(
    CLINICAL_CHUNKS_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("CLINICAL INDEX CREATED SUCCESSFULLY")
print("========================================")
print("Source chunks:", len(all_chunks))
print("Clinical chunks indexed:", len(chunks))
print("Excluded dengue.csv chunks:", excluded_count)
print("Short chunks skipped:", short_count)
print("Embedding shape:", embeddings.shape)
print("FAISS vectors:", index.ntotal)
print("Saved index:", INDEX_FILE.name)
print("Saved embeddings:", EMBEDDINGS_FILE.name)
print("Saved chunks:", CLINICAL_CHUNKS_FILE.name)
print("========================================")
