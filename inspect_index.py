
import json
from pathlib import Path
from collections import Counter

root = Path(r"D:\Projects\Medical-Graph-RAG-PROJ")

chunks_path = root / "medical_clinical_chunks.json"
graph_path = root / "medical_graph.json"

with chunks_path.open("r", encoding="utf-8") as f:
    chunks = json.load(f)

with graph_path.open("r", encoding="utf-8") as f:
    graph = json.load(f)

print("TOTAL CLINICAL CHUNKS:", len(chunks))

print("\nCHUNK FILE FORMAT:")
print(type(chunks).__name__)

if isinstance(chunks, dict):
    print("Top-level keys:", list(chunks.keys()))
    records = chunks.get("chunks", chunks.get("documents", []))
elif isinstance(chunks, list):
    records = chunks
else:
    records = []

print("\nCHUNKS BY SOURCE:")
source_counts = Counter()

for item in records:
    if isinstance(item, dict):
        source = (
            item.get("source")
            or item.get("document")
            or item.get("file_name")
            or item.get("metadata", {}).get("source", "UNKNOWN")
        )
        source_counts[str(source)] += 1

for source, count in source_counts.most_common():
    print(f"{count:5} chunks | {source}")

print("\nGRAPH SUMMARY:")
print("Nodes:", len(graph.get("nodes", [])))
print("Relationships:", len(graph.get("relationships", [])))

print("\nFIRST CHUNK RECORD:")
if records:
    print(json.dumps(records[0], indent=2, ensure_ascii=False)[:2000])
