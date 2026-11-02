
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict


# --------------------------------------------------
# Locate the existing graph file
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

GRAPH_PATHS = [
    BASE_DIR / "medical_graph.json",
    PROJECT_DIR / "medical_graph.json",
]


def find_graph_file():
    for path in GRAPH_PATHS:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "medical_graph.json not found in backend or project root."
    )


# --------------------------------------------------
# Read graph data
# --------------------------------------------------

def load_graph(path):
    with open(path, "r", encoding="utf-8") as file:
        graph = json.load(file)

    if not isinstance(graph, dict):
        raise ValueError("Graph JSON must contain an object.")

    nodes = graph.get("nodes", [])

    # Support common relationship-array names
    edges = (
        graph.get("edges")
        or graph.get("relationships")
        or graph.get("links")
        or []
    )

    if not isinstance(nodes, list):
        raise ValueError("'nodes' must be a JSON array.")

    if not isinstance(edges, list):
        raise ValueError("Relationship data must be a JSON array.")

    return nodes, edges


# --------------------------------------------------
# Generate candidate inferences
# --------------------------------------------------

def generate_candidate_inferences(
    graph_path=None,
    min_support=2,
    min_link_score=0.0
):
    if graph_path is None:
        graph_path = find_graph_file()

    nodes, edges = load_graph(graph_path)

    # Build node lookup
    node_lookup = {}

    for node in nodes:
        node_id = node.get("id")
        if node_id:
            node_lookup[node_id] = node

    # Identify medical concepts
    concepts = {}

    for node in nodes:
        if node.get("node_type") == "MedicalConcept":
            node_id = node.get("id")
            name = node.get("name")

            if node_id and name:
                concepts[node_id] = name

    # Identify document chunks
    chunks = set()

    for node in nodes:
        if node.get("node_type") == "DocumentChunk":
            node_id = node.get("id")
            if node_id:
                chunks.add(node_id)

    print("Total nodes:", len(nodes))
    print("Total edges:", len(edges))
    print("Total medical concepts:", len(concepts))
    print("Total document chunks:", len(chunks))

    # Group concepts by chunk
    chunk_concepts = defaultdict(set)
    chunk_metadata = {}

    mentions_count = 0
    matched_edges = 0

    for edge in edges:

        relationship = str(
            edge.get("type", edge.get("label", ""))
        ).strip().upper()

        if relationship != "MENTIONS":
            continue

        mentions_count += 1

        source_id = edge.get("source")
        target_id = edge.get("target")

        if not source_id or not target_id:
            continue

        # Determine which endpoint is the chunk
        # and which endpoint is the medical concept.
        source_node = node_lookup.get(source_id, {})
        target_node = node_lookup.get(target_id, {})

        if source_id in chunks or source_node.get("node_type") == "DocumentChunk":
            chunk_id = source_id
            concept_id = target_id

        elif target_id in chunks or target_node.get("node_type") == "DocumentChunk":
            chunk_id = target_id
            concept_id = source_id

        else:
            # Fallback for chunk IDs not present in the nodes array
            if str(source_id).startswith("chunk::"):
                chunk_id = source_id
                concept_id = target_id
            elif str(target_id).startswith("chunk::"):
                chunk_id = target_id
                concept_id = source_id
            else:
                continue

        if concept_id not in concepts:
            continue

        try:
            score = float(edge.get("score", 0.0))
        except (ValueError, TypeError):
            score = 0.0

        if score < min_link_score:
            continue

        chunk_concepts[chunk_id].add(concept_id)
        matched_edges += 1

        # Preserve supporting document information
        chunk_metadata[chunk_id] = {
            "document": edge.get("document", ""),
            "source": edge.get("source_name", ""),
            "document_path": edge.get("document_path", ""),
            "chunk_id": edge.get("chunk_id")
        }

    print("Total MENTIONS edges:", mentions_count)
    print("Matched chunk-concept edges:", matched_edges)
    print("Chunks containing concepts:", len(chunk_concepts))

    # Generate concept pairs from each chunk
    pair_evidence = defaultdict(dict)

    for chunk_id, concept_ids in chunk_concepts.items():

        if len(concept_ids) < 2:
            continue

        for concept_a, concept_b in combinations(
            sorted(concept_ids), 2
        ):
            metadata = chunk_metadata.get(chunk_id, {})

            pair_evidence[(concept_a, concept_b)][chunk_id] = {
                "document": metadata.get("document", ""),
                "source": metadata.get("source", ""),
                "document_path": metadata.get("document_path", ""),
                "chunk_id": metadata.get("chunk_id")
            }

    # Create candidate associations
    inferences = []

    for (concept_a, concept_b), evidence_by_chunk in pair_evidence.items():

        evidence = list(evidence_by_chunk.values())

        if len(evidence) < min_support:
            continue

        unique_documents = {
            item["document_path"] or item["document"]
            for item in evidence
            if item["document_path"] or item["document"]
        }

        inferences.append({
            "concept_1": concepts[concept_a],
            "concept_1_id": concept_a,
            "concept_2": concepts[concept_b],
            "concept_2_id": concept_b,
            "relationship": "CO_OCCURS_WITH",
            "inference_type": "candidate_association",
            "supporting_chunks": len(evidence),
            "supporting_documents": len(unique_documents),
            "evidence": evidence,
            "clinical_causation_established": False
        })

    inferences.sort(
        key=lambda item: item["supporting_chunks"],
        reverse=True
    )

    return inferences


# --------------------------------------------------
# Main program
# --------------------------------------------------

if __name__ == "__main__":

    try:
        graph_path = find_graph_file()

        print("\nLoading graph:", graph_path)

        results = generate_candidate_inferences(
            graph_path=graph_path,
            min_support=2,
            min_link_score=0.0
        )

        print("\n========== KNOWLEDGE INFERENCE ==========")
        print("Total candidate inferences:", len(results))

        for index, item in enumerate(results[:20], start=1):

            print(
                f"\n{index}. {item['concept_1']} "
                f"<-> {item['concept_2']}"
            )

            print("Relationship:", item["relationship"])
            print("Supporting chunks:", item["supporting_chunks"])
            print("Supporting documents:", item["supporting_documents"])

            for evidence in item["evidence"]:
                print(
                    "Evidence:",
                    evidence["document"],
                    "| Chunk:",
                    evidence["chunk_id"]
                )

        output_path = BASE_DIR / "candidate_inferences.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\nResults saved to:", output_path)

    except Exception as error:
        print("\nERROR:", error)
