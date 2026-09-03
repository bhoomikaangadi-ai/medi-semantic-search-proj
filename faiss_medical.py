import json
import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "medical_embeddings.npy"
DATA_FILE = "medical_chunks_source.json"
INDEX_FILE = "medical_faiss.index"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading BioBERT embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
).astype("float32")

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# LOAD MEDICAL CHUNKS
# ============================================================

print("Loading medical chunks...")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    "Number of chunks:",
    len(chunks)
)


# ============================================================
# VERIFY DATA
# ============================================================

if len(embeddings) != len(chunks):

    raise ValueError(
        "Number of embeddings does not match "
        "number of medical chunks."
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

print(
    "Creating FAISS index..."
)

index = faiss.IndexFlatL2(
    dimension
)


# ============================================================
# ADD EMBEDDINGS
# ============================================================

index.add(
    embeddings
)

print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# ============================================================
# SAVE INDEX
# ============================================================

faiss.write_index(
    index,
    INDEX_FILE
)

print()
print("========================================")
print("FAISS INDEX CREATED")
print("========================================")

print(
    "Index dimension:",
    index.d
)

print(
    "Number of vectors:",
    index.ntotal
)

print(
    "Saved to:",
    INDEX_FILE
)

print("========================================")