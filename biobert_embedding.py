import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"

INPUT_FILE = "medical_chunks_source.json"

OUTPUT_FILE = "medical_embeddings.npy"


# ============================================================
# LOAD BIOBERT
# ============================================================

print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    BIOBERT_MODEL
)

model = AutoModel.from_pretrained(
    BIOBERT_MODEL
)

model.eval()

print("BioBERT loaded successfully.")


# ============================================================
# GENERATE BIOBERT EMBEDDING
# ============================================================

def get_biobert_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    masked_embeddings = (
        token_embeddings * mask
    )

    summed = torch.sum(
        masked_embeddings,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    embedding = summed / counts

    return embedding[0].cpu().numpy()


# ============================================================
# LOAD MEDICAL CHUNKS
# ============================================================

print()
print("Loading medical chunks...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

embeddings = []

for i, chunk in enumerate(chunks):

    text = chunk["text"]

    embedding = get_biobert_embedding(
        text
    )

    embeddings.append(
        embedding
    )

    if (i + 1) % 10 == 0:

        print(
            f"Processed {i + 1}/{len(chunks)} chunks"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

embeddings = np.array(
    embeddings,
    dtype="float32"
)


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_FILE,
    embeddings
)


# ============================================================
# VERIFY
# ============================================================

print()
print("========================================")
print("BIOBERT EMBEDDING COMPLETE")
print("========================================")

print(
    "Number of embeddings:",
    embeddings.shape[0]
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)

print(
    "Saved to:",
    OUTPUT_FILE
)

print("========================================")