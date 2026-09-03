import os
import json
import re
from collections import Counter

from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_FOLDER = "medical_documents"
OUTPUT_FILE = "medical_chunks_source.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 200


# ============================================================
# EXTRACT PDF PAGES
# ============================================================

def extract_pdf_pages(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return pages


# ============================================================
# CLEAN INDIVIDUAL PAGE
# ============================================================

def clean_page_text(text):

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# REMOVE REPEATED LINES
# ============================================================

def remove_repeated_lines(pages):

    cleaned_pages = []

    # Extract individual lines before normalization
    all_lines = []

    for page in pages:

        lines = page.splitlines()

        for line in lines:

            line = re.sub(r"\s+", " ", line).strip()

            if line:
                all_lines.append(line)

    # Count how frequently each line appears
    line_counts = Counter(all_lines)

    # A line appearing on many pages is likely a
    # repeated header/footer.
    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= 5
    }

    # Rebuild pages without repeated lines
    for page in pages:

        new_lines = []

        for line in page.splitlines():

            line = re.sub(r"\s+", " ", line).strip()

            if not line:
                continue

            if line in repeated_lines:
                continue

            new_lines.append(line)

        cleaned_pages.append(
            " ".join(new_lines)
        )

    return cleaned_pages


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(text):

    text = clean_page_text(text)

    chunks = []

    start = 0

    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:

            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents():

    all_chunks = []

    for root, directories, files in os.walk(
        DOCUMENTS_FOLDER
    ):

        for filename in files:

            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(
                root,
                filename
            )

            print()
            print(f"Processing: {pdf_path}")

            try:

                # --------------------------------------------
                # Extract pages
                # --------------------------------------------

                pages = extract_pdf_pages(
                    pdf_path
                )

                print(
                    f"Extracted {len(pages)} pages"
                )

                # --------------------------------------------
                # Remove repeated headers/footers
                # --------------------------------------------

                pages = remove_repeated_lines(
                    pages
                )

                # --------------------------------------------
                # Combine cleaned pages
                # --------------------------------------------

                text = "\n".join(pages)

                # --------------------------------------------
                # Create chunks
                # --------------------------------------------

                chunks = create_chunks(text)

                # --------------------------------------------
                # Determine source
                # --------------------------------------------

                relative_path = os.path.relpath(
                    pdf_path,
                    DOCUMENTS_FOLDER
                )

                source = relative_path.split(
                    os.sep
                )[0]

                # --------------------------------------------
                # Store chunks
                # --------------------------------------------

                for chunk_id, chunk in enumerate(
                    chunks
                ):

                    all_chunks.append(
                        {
                            "text": chunk,
                            "source": source,
                            "document": filename,
                            "chunk_id": chunk_id
                        }
                    )

                print(
                    f"Created {len(chunks)} chunks"
                )

            except Exception as e:

                print(
                    f"ERROR processing {pdf_path}: {e}"
                )

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("========================================")
    print("DOCUMENT PROCESSING COMPLETE")
    print("========================================")
    print(
        f"Total chunks: {len(all_chunks)}"
    )
    print(
        f"Output file: {OUTPUT_FILE}"
    )
    print("========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_documents()