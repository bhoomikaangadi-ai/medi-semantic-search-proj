
import json
import os
from pathlib import Path
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
GRAPH_FILE = ROOT / "medical_graph.json"

URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 500


def batches(items, size=BATCH_SIZE):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main():
    if not PASSWORD:
        raise SystemExit(
            "Set NEO4J_PASSWORD in PowerShell before running this script."
        )

    with GRAPH_FILE.open(encoding="utf-8") as file:
        graph = json.load(file)

    concepts = graph.get("nodes", [])
    relationships = graph.get("relationships", graph.get("edges", []))

    if not concepts:
        raise SystemExit("No concept nodes found in medical_graph.json.")

    # Validate references before connecting to Neo4j.
    concept_ids = {str(node["id"]) for node in concepts if node.get("id")}
    missing_targets = sorted({
        str(rel.get("target"))
        for rel in relationships
        if rel.get("target") is not None
        and str(rel["target"]) not in concept_ids
    })

    if missing_targets:
        raise SystemExit(
            f"Found {len(missing_targets)} relationship targets "
            "missing from the concept nodes. Import cancelled."
        )

    # Create chunk records from relationship metadata.
    chunk_documents = {}
    for rel in relationships:
        source = rel.get("source")
        if source is not None and str(source).startswith("chunk_"):
            chunk_id = str(source)
            chunk_documents.setdefault(
                chunk_id, set()
            ).add(str(rel.get("document", "unknown")))

    chunk_rows = [
        {
            "id": chunk_id,
            "documents": sorted(documents),
        }
        for chunk_id, documents in chunk_documents.items()
    ]

    # Use one dedicated label so existing unrelated nodes are preserved.
    concept_rows = [
        {
            "id": str(node["id"]),
            "name": str(node.get("name", "")),
            "semantic_type": node.get("semantic_type", []),
            "source": str(node.get("source", "unknown")),
        }
        for node in concepts
        if node.get("id")
    ]

    rel_rows = []
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        if source is None or target is None:
            continue

        rel_rows.append({
            "source": str(source),
            "target": str(target),
            "score": rel.get("score"),
            "document": str(rel.get("document", "unknown")),
        })

    print("JSON concepts:", len(concept_rows))
    print("Unique chunk IDs:", len(chunk_rows))
    print("JSON relationships:", len(rel_rows))

    driver = GraphDatabase.driver(
        URI, auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            # Create constraints for stable, repeatable imports.
            session.run("""
                CREATE CONSTRAINT medical_concept_id IF NOT EXISTS
                FOR (n:MedicalOntologyConcept)
                REQUIRE n.id IS UNIQUE
            """).consume()

            session.run("""
                CREATE CONSTRAINT medical_chunk_id IF NOT EXISTS
                FOR (n:MedicalDocumentChunk)
                REQUIRE n.id IS UNIQUE
            """).consume()

            # Import concepts without deleting existing data.
            for batch in batches(concept_rows):
                session.run("""
                    UNWIND $rows AS row
                    MERGE (n:MedicalOntologyConcept {id: row.id})
                    SET n.name = row.name,
                        n.semantic_type = row.semantic_type,
                        n.source = row.source
                """, rows=batch).consume()

            # Create minimal chunk nodes from available metadata.
            for batch in batches(chunk_rows):
                session.run("""
                    UNWIND $rows AS row
                    MERGE (c:MedicalDocumentChunk {id: row.id})
                    SET c.documents = row.documents
                """, rows=batch).consume()

            # Connect chunks to concepts.
            for batch in batches(rel_rows):
                session.run("""
                    UNWIND $rows AS row
                    MATCH (c:MedicalDocumentChunk {id: row.source})
                    MATCH (n:MedicalOntologyConcept {id: row.target})
                    MERGE (c)-[r:MENTIONS]->(n)
                    SET r.score = row.score,
                        r.document = row.document
                """, rows=batch).consume()

            result = session.run("""
                RETURN
                  (SELECT count(*) FROM []) AS unused
            """) if False else None

            counts = session.run("""
                MATCH (n:MedicalOntologyConcept)
                WITH count(n) AS concepts
                MATCH (c:MedicalDocumentChunk)
                WITH concepts, count(c) AS chunks
                OPTIONAL MATCH (:MedicalDocumentChunk)-[r:MENTIONS]->(:MedicalOntologyConcept)
                RETURN concepts, chunks, count(r) AS mentions
            """).single()

            print("\nImport verification:")
            print("Concept nodes:", counts["concepts"])
            print("Chunk nodes:", counts["chunks"])
            print("MENTIONS relationships:", counts["mentions"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
