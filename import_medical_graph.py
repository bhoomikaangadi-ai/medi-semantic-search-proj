
import json
import os
from pathlib import Path

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
GRAPH_FILE = ROOT / "medical_graph.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

BATCH_SIZE = 500


def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}")

    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    relationships = graph.get("relationships", [])

    node_ids = [node.get("id") for node in nodes]

    if any(node_id is None for node_id in node_ids):
        raise ValueError("A graph node is missing its ID.")

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node IDs found; import cancelled.")

    valid_ids = set(node_ids)

    for rel in relationships:
        if rel.get("source") not in valid_ids:
            raise ValueError(f"Missing source node: {rel.get('source')}")
        if rel.get("target") not in valid_ids:
            raise ValueError(f"Missing target node: {rel.get('target')}")
        if rel.get("type") != "MENTIONS":
            raise ValueError(
                f"Unexpected relationship type: {rel.get('type')}"
            )

    return nodes, relationships


def batches(items, size=BATCH_SIZE):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def import_nodes(tx, nodes):
    query = """
    UNWIND $rows AS row
    MERGE (n:MedicalGraphEntity {id: row.id})
    SET n.name = row.name,
        n.node_type = row.node_type,
        n.source = row.source,
        n.document = row.document,
        n.document_path = row.document_path,
        n.chunk_id = toString(row.chunk_id),
        n.text = row.text,
        n.semantic_type = row.semantic_type
    RETURN count(n) AS count
    """
    result = tx.run(query, rows=nodes)
    return result.single()["count"]


def import_mentions(tx, relationships):
    query = """
    UNWIND $rows AS row
    MATCH (source:MedicalGraphEntity {id: row.source})
    MATCH (target:MedicalGraphEntity {id: row.target})
    MERGE (source)-[r:MENTIONS {
        document_path: row.document_path,
        chunk_id: toString(row.chunk_id),
        start_char: row.start_char,
        end_char: row.end_char,
        mention_text: row.mention_text
    }]->(target)
    SET r.score = row.score,
        r.source_name = row.source_name,
        r.document = row.document
    RETURN count(r) AS count
    """
    result = tx.run(query, rows=relationships)
    return result.single()["count"]


def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Set it in PowerShell before importing."
        )

    nodes, relationships = load_graph()

    print("Validated graph file:", GRAPH_FILE.name)
    print("Nodes to import:", len(nodes))
    print("MENTIONS relationships to import:", len(relationships))
    print("Connecting to Neo4j:", NEO4J_URI)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            session.run("""
                CREATE CONSTRAINT medical_graph_entity_id IF NOT EXISTS
                FOR (n:MedicalGraphEntity)
                REQUIRE n.id IS UNIQUE
            """).consume()

            for batch in batches(nodes):
                session.execute_write(import_nodes, batch)

            for batch in batches(relationships):
                session.execute_write(import_mentions, batch)

            node_result = session.run("""
                MATCH (n:MedicalGraphEntity)
                RETURN count(n) AS count
            """).single()

            relation_result = session.run("""
                MATCH (:MedicalGraphEntity)-[r:MENTIONS]->
                      (:MedicalGraphEntity)
                RETURN count(r) AS count
            """).single()

            print("\nImport finished.")
            print("MedicalGraphEntity nodes in database:", node_result["count"])
            print("MENTIONS relationships in database:", relation_result["count"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
