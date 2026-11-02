
import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_FOLDER = PROJECT_ROOT / "medical_documents"
OUTPUT_FILE = PROJECT_ROOT / "medical_chunks_source.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 200

SUPPORTED_EXTENSIONS = {
    ".pdf", ".html", ".htm", ".txt",
    ".csv", ".docx", ".json"
}

# Exclude only this specific document from clinical extraction.
# Other Kaggle documents, including dengue_cases_in_india.csv,
# remain eligible for processing.
EXCLUDED_DOCUMENTS = {
    "kaggle/dengue.csv"
}


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """Normalize whitespace without joining words together."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path):
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                f"[Page {page_number}]\n{text.strip()}"
            )

    return "\n\n".join(pages)


# ============================================================
# HTML EXTRACTION
# ============================================================

def extract_html(html_path):
    """Extract visible text and preserve HTML table structure."""
    with open(
        html_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:
        soup = BeautifulSoup(file.read(), "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else html_path.stem
    )

    # Remove non-content elements.
    for element in soup.find_all(
        ["script", "style", "noscript", "nav", "footer"]
    ):
        element.decompose()

    parts = [f"Document title: {title}"]

    # Preserve tables as labelled rows.
    for table_number, table in enumerate(
        soup.find_all("table"), start=1
    ):
        parts.append(f"\nTable {table_number}:")

        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])

            values = [
                cell.get_text(" ", strip=True)
                for cell in cells
            ]

            if values:
                parts.append(" | ".join(values))

    page_text = soup.get_text(" ", strip=True)

    if page_text:
        parts.append("\nPage text:")
        parts.append(page_text)

    return "\n".join(parts), title


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(txt_path):
    """Read a plain-text document."""
    with open(
        txt_path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as file:
        return file.read()


# ============================================================
# CSV EXTRACTION
# ============================================================

def extract_csv(csv_path):
    """Convert CSV rows into readable, labelled text."""
    parts = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline=""
    ) as file:

        sample = file.read(8192)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)

        if reader.fieldnames:
            headers = [
                str(header).strip()
                if header is not None
                else ""
                for header in reader.fieldnames
            ]

            parts.append("CSV columns: " + " | ".join(headers))

            for row_number, row in enumerate(reader, start=1):
                values = []

                for header in reader.fieldnames:
                    value = row.get(header)

                    if value is not None and str(value).strip():
                        values.append(
                            f"{header}: {str(value).strip()}"
                        )

                if values:
                    parts.append(
                        f"Row {row_number}: " + "; ".join(values)
                    )

    return "\n".join(parts)


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(docx_path):
    """Extract paragraphs and tables from a Word document."""
    document = Document(str(docx_path))
    parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    for table_number, table in enumerate(
        document.tables, start=1
    ):
        parts.append(f"\nTable {table_number}:")

        for row in table.rows:
            values = [
                cell.text.strip().replace("\n", " ")
                for cell in row.cells
            ]

            if any(values):
                parts.append(" | ".join(values))

    return "\n".join(parts)


# ============================================================
# JSON EXTRACTION
# ============================================================

def flatten_json(value, prefix=""):
    """Convert nested JSON into readable key/value lines."""
    lines = []

    if isinstance(value, dict):
        for key, item in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(item, new_prefix))

    elif isinstance(value, list):
        for index, item in enumerate(value):
            new_prefix = f"{prefix}[{index}]"
            lines.extend(flatten_json(item, new_prefix))

    else:
        if value is not None:
            lines.append(f"{prefix}: {value}")

    return lines


def extract_json(json_path):
    """Read JSON and turn its structured data into text."""
    with open(
        json_path,
        "r",
        encoding="utf-8-sig"
    ) as file:
        data = json.load(file)

    return "\n".join(flatten_json(data))


# ============================================================
# DISPATCH TO THE CORRECT EXTRACTOR
# ============================================================

def extract_document(file_path):
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path), file_path.stem

    if extension in {".html", ".htm"}:
        return extract_html(file_path)

    if extension == ".txt":
        return extract_txt(file_path), file_path.stem

    if extension == ".csv":
        return extract_csv(file_path), file_path.stem

    if extension == ".docx":
        return extract_docx(file_path), file_path.stem

    if extension == ".json":
        return extract_json(file_path), file_path.stem

    raise ValueError(f"Unsupported file format: {extension}")


# ============================================================
# CREATE OVERLAPPING CHUNKS
# ============================================================

def create_chunks(text):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# PROCESS ALL DOCUMENTS
# ============================================================

def process_documents():
    if not DOCUMENTS_FOLDER.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_FOLDER}"
        )

    all_chunks = []
    processed_files = 0
    failed_files = 0
    skipped_files = 0

    files = sorted(
        path
        for path in DOCUMENTS_FOLDER.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for file_path in files:
        relative_path = file_path.relative_to(DOCUMENTS_FOLDER)
        relative_path_text = relative_path.as_posix()

        # Exclude the specified document before reading it.
        if relative_path_text.lower() in EXCLUDED_DOCUMENTS:
            skipped_files += 1
            print(
                f"Skipping excluded document: {relative_path_text}"
            )
            continue

        # Preserve the first folder as the source category.
        source = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "unknown"
        )

        print(f"\nProcessing: {relative_path_text}")

        try:
            text, title = extract_document(file_path)
            text = clean_text(text)

            if not text:
                print("WARNING: No text extracted.")
                continue

            chunks = create_chunks(text)

            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": source,
                    "document": file_path.name,
                    "chunk_id": chunk_id,
                    "document_path": relative_path_text,
                    "title": title,
                    "file_type": (
                        file_path.suffix.lower().lstrip(".")
                    )
                })

            processed_files += 1

            print(f"Extracted characters: {len(text)}")
            print(f"Created chunks: {len(chunks)}")

        except Exception as error:
            failed_files += 1
            print(
                f"ERROR processing {relative_path_text}: {error}"
            )

    # Write to a temporary file first, then replace the output.
    # This avoids leaving a partially written JSON file.
    temporary_file = OUTPUT_FILE.with_suffix(".json.tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_file.replace(OUTPUT_FILE)

    print("\n========================================")
    print("DOCUMENT PROCESSING COMPLETE")
    print("========================================")
    print(f"Files discovered: {len(files)}")
    print(f"Files processed: {processed_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Files failed: {failed_files}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_documents()
