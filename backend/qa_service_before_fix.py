
import json
import re
from pathlib import Path
from functools import lru_cache
from collections import deque

import faiss
import ollama
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
OLLAMA_MODEL = "qwen2.5:7b"

INDEX_FILE = ROOT / "medical_clinical_faiss.index"
CHUNKS_FILE = ROOT / "medical_clinical_chunks.json"
GRAPH_FILE = ROOT / "medical_graph.json"

FINAL_SOURCE_COUNT = 4
CANDIDATE_COUNT = 50
GRAPH_HOPS = 1
MAX_GRAPH_CONTEXT = 20

INDIA_TERMS = {
    "india", "indian", "andhra pradesh", "state-wise",
    "state wise", "national total", "across states",
    "in the country", "indian government", "ncvbdc",
}

STATISTICS_TERMS = {
    "statistics", "case", "cases", "death", "deaths",
    "number", "numbers", "how many", "year-wise",
    "year wise", "annual", "total", "latest", "current",
    "trend", "data", "figures", "reported",
}


def normalise(value):
    """Normalize text for robust comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


@lru_cache(maxsize=1)
def load_resources():
    """Load model, FAISS index, chunks, and knowledge graph once."""

    for path in (INDEX_FILE, CHUNKS_FILE, GRAPH_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading biomedical embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading clinical chunks...")
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Loading knowledge graph...")
    with GRAPH_FILE.open("r", encoding="utf-8") as file:
        graph = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Clinical chunks must be a non-empty JSON list.")

    if not isinstance(graph, dict):
        raise ValueError("Knowledge graph must be a JSON object.")

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS has {index.ntotal} vectors but "
            f"{len(chunks)} chunks were loaded. "
            "Rebuild the index or use the matching chunk file."
        )

    if index.d != model.get_sentence_embedding_dimension():
        raise ValueError(
            "Embedding model dimension does not match the FAISS index."
        )

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            "Expected a FAISS inner-product index. "
            "Check how the index was built."
        )

    print(f"Loaded {len(chunks)} chunks and {index.ntotal} vectors.")
    print_graph_summary(graph)

    return model, index, chunks, graph


def print_graph_summary(graph):
    """Print graph structure for debugging."""
    nodes = graph.get("nodes", [])
    edges = graph.get("relationships", graph.get("edges", []))

    print(
        f"Knowledge graph: {len(nodes) if isinstance(nodes, list) else 'unknown'} "
        "nodes, "
        f"{len(edges) if isinstance(edges, list) else 'unknown'} edges"
    )


def graph_node_name(node):
    if not isinstance(node, dict):
        return None

    return (
        node.get("canonical_name")
        or node.get("name")
        or node.get("label")
        or node.get("id")
    )


def graph_node_id(node):
    if not isinstance(node, dict):
        return None

    value = node.get("id") or node.get("node_id")
    return str(value) if value is not None else None


def graph_edge_endpoints(edge):
    """Support common source/target and subject/object edge formats."""
    if not isinstance(edge, dict):
        return None, None

    source = (
        edge.get("source")
        or edge.get("from")
        or edge.get("subject")
        or edge.get("source_id")
    )
    target = (
        edge.get("target")
        or edge.get("to")
        or edge.get("object")
        or edge.get("target_id")
    )

    if source is None or target is None:
        return None, None

    # Some graph formats store complete node dictionaries in edges.
    if isinstance(source, dict):
        source = source.get("id") or source.get("name")
    if isinstance(target, dict):
        target = target.get("id") or target.get("name")

    if source is None or target is None:
        return None, None

    return str(source), str(target)


def graph_edge_label(edge):
    if not isinstance(edge, dict):
        return "RELATED_TO"

    return str(
        edge.get("relation")
        or edge.get("type")
        or edge.get("label")
        or edge.get("predicate")
        or "RELATED_TO"
    )


def build_graph_lookup(graph):
    """Build node and adjacency lookups without modifying the graph."""
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get("relationships", graph.get("edges", []))

    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    nodes_by_id = {}
    name_to_ids = {}

    for node in raw_nodes:
        node_id = graph_node_id(node)
        name = graph_node_name(node)

        if node_id is not None:
            nodes_by_id[node_id] = node

        if name:
            name_to_ids.setdefault(normalise(name), set()).add(
                node_id if node_id is not None else str(name)
            )

    adjacency = {}

    for edge in raw_edges:
        source, target = graph_edge_endpoints(edge)

        if source is None or target is None:
            continue

        relation = graph_edge_label(edge)

        adjacency.setdefault(source, []).append(
            (target, relation, "outgoing")
        )
        adjacency.setdefault(target, []).append(
            (source, relation, "incoming")
        )

    return nodes_by_id, name_to_ids, adjacency, raw_edges


def is_indian_statistics_question(question):
    """Detect questions about Indian dengue statistics."""
    q = normalise(question)

    india_match = any(normalise(term) in q for term in INDIA_TERMS)
    stats_match = any(normalise(term) in q for term in STATISTICS_TERMS)

    return india_match and stats_match


def is_indian_government_source(chunk):
    """Identify the Indian government dengue table."""
    document = normalise(chunk.get("document", ""))
    source = normalise(chunk.get("source", ""))
    title = normalise(chunk.get("title", ""))
    text = normalise(chunk.get("text", ""))

    official_source = any(term in value for value in (
        source, document, title
    ) for term in (
        "india govt",
        "ncvbdc",
        "dengue situation in india",
        "national center for vector borne diseases control",
        "national centre for vector borne diseases control",
    ))

    table_content = (
        "affected states" in text
        or "cases and deaths in the country" in text
    )

    return official_source or (
        table_content and ("2025" in text or "2026" in text)
    )


def is_state_statistics_chunk(chunk, requested_state=None):
    """Identify chunks containing a state-wise annual statistics table."""
    text = normalise(chunk.get("text", ""))

    if requested_state:
        state = normalise(requested_state)
        if state not in text:
            return False

    has_table_heading = (
        "affected states" in text
        or "cases and deaths in the country" in text
    )

    has_years = sum(
        year in text for year in ("2021", "2022", "2023", "2024", "2025", "2026")
    ) >= 2

    return has_table_heading or has_years


def extract_requested_state(question):
    """Return a known state name if the user mentions one."""
    q = normalise(question)

    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
        "Chhattisgarh", "Goa", "Gujarat", "Haryana",
        "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
        "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
        "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
    ]

    for state in states:
        if normalise(state) in q:
            return state

    return None



def related_graph_context(chunk, graph, hops=GRAPH_HOPS):
    """
    Return only actual MENTIONS relationships connected directly
    to the retrieved chunk. Avoid matching unrelated entity names
    found elsewhere in the passage.
    """
    chunk_id = chunk.get("chunk_id")

    if chunk_id is None:
        return []

    concepts = get_chunk_graph_concepts(
        chunk_id=chunk_id,
        graph=graph,
        limit=MAX_GRAPH_CONTEXT,
    )

    relationships = []

    for concept in concepts:
        name = concept.get("name", "").strip()
        relation = concept.get("relationship", "MENTIONS")

        if not name:
            continue

        relationships.append(
            f"chunk_{chunk_id} --[{relation}]--> {name}"
        )

    return relationships



def retrieve_candidates(question, model, index, chunks):
    """Run vector search and return candidate chunks with similarity scores."""
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    if query_vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match FAISS index.")

    count = min(CANDIDATE_COUNT, index.ntotal)
    similarities, indices = index.search(query_vector, count)

    results = []
    seen = set()

    for similarity, idx in zip(similarities[0], indices[0]):
        idx = int(idx)

        if idx < 0 or idx >= len(chunks) or idx in seen:
            continue

        seen.add(idx)
        chunk = chunks[idx]

        if not isinstance(chunk, dict) or not chunk.get("text"):
            continue

        results.append({
            "index": idx,
            "similarity": float(similarity),
            "priority": 0.0,
        })

    return results


def select_evidence(question, candidates, chunks):
    """
    Rank semantic candidates and inject matching government statistics
    chunks from the full chunk collection when necessary.
    """
    stats_question = is_indian_statistics_question(question)
    requested_state = extract_requested_state(question)

    by_index = {item["index"]: item for item in candidates}

    if stats_question:
        # Do not depend only on the top FAISS results for a known table.
        # Inspect all chunks for the official state-wise table.
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or not chunk.get("text"):
                continue

            official = is_indian_government_source(chunk)
            table_match = is_state_statistics_chunk(
                chunk,
                requested_state=requested_state,
            )

            if official and table_match:
                by_index.setdefault(idx, {
                    "index": idx,
                    # This is not a measured vector score if it was not
                    # returned by FAISS; mark it as zero for transparency.
                    "similarity": 0.0,
                    "priority": 0.0,
                    "injected_by_rule": True,
                })

    ranked = list(by_index.values())

    for item in ranked:
        chunk = chunks[item["index"]]

        if stats_question:
            if is_indian_government_source(chunk):
                item["priority"] = 2.0
            elif is_state_statistics_chunk(chunk, requested_state):
                item["priority"] = 1.0

        item["rank_score"] = item["similarity"] + item.get("priority", 0.0)

    ranked.sort(
        key=lambda item: item["rank_score"],
        reverse=True,
    )

    # Keep the official table among the final evidence passages for
    # Indian statistics questions whenever one was found.
    selected = ranked[:FINAL_SOURCE_COUNT]

    if stats_question:
        official_items = [
            item for item in ranked
            if is_indian_government_source(chunks[item["index"]])
            and is_state_statistics_chunk(
                chunks[item["index"]],
                requested_state=requested_state,
            )
        ]

        if official_items and not any(
            is_indian_government_source(chunks[item["index"]])
            for item in selected
        ):
            selected[-1] = official_items[0]

        # Remove duplicates after any replacement.
        deduplicated = []
        seen_indices = set()

        for item in selected:
            if item["index"] not in seen_indices:
                deduplicated.append(item)
                seen_indices.add(item["index"])

        selected = deduplicated

    return selected


def get_chunk_graph_concepts(chunk_id, graph, limit=10):
    """
    Return medical concepts linked to a retrieved chunk.

    Graph edges have the form:
    chunk_93 --[MENTIONS]--> CUI
    """
    chunk_node_id = f"chunk_{chunk_id}"

    nodes_by_id = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if node.get("id") is not None
    }

    results = []
    seen = set()

    for edge in graph.get("relationships", []):
        if str(edge.get("source")) != chunk_node_id:
            continue

        if edge.get("type") != "MENTIONS":
            continue

        target_id = str(edge.get("target", ""))
        node = nodes_by_id.get(target_id)

        if not node:
            continue

        name = node.get("name", "").strip()
        if not name:
            continue

        key = (target_id, name.casefold())
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "id": target_id,
            "name": name,
            "semantic_type": node.get("semantic_type", []),
            "source": node.get("source", ""),
            "score": float(edge.get("score", 0.0)),
            "relationship": edge.get("type", "MENTIONS"),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]

def answer_question(question: str) -> dict:
    """Answer using semantic evidence and connected knowledge-graph facts."""
    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    model, index, chunks, graph = load_resources()

    candidates = retrieve_candidates(question, model, index, chunks)
    ranked = select_evidence(question, candidates, chunks)

    if not ranked:
        raise ValueError("No valid medical passages were retrieved.")

    evidence = []
    sources = []
    all_graph_relationships = []
    all_concepts = set()

    for item in ranked:
        chunk = chunks[item["index"]]
        document = str(chunk.get("document", "Unknown document"))
        source = str(chunk.get("source", document))
        chunk_id = chunk.get("chunk_id", item["index"])
        chunk_text = str(chunk["text"])

        relationships = related_graph_context(chunk, graph)

        all_graph_relationships.extend(relationships)

        evidence.append(
            f"[Document: {document}; Source: {source}; "
            f"Chunk: {chunk_id}]\n{chunk_text}"
        )

        for relationship in relationships:
            all_concepts.add(relationship)

        # Return score separately from rule-based ranking priority.
        sources.append({
            "title": document,
            "document": document,
            "source": source,
            "chunk_id": chunk_id,
            "score": round(item["similarity"], 4),
            "selected_by_rule": bool(item.get("injected_by_rule", False)),
            "text": chunk_text,
            "graph_relationships": relationships,
        })

    stats_question = is_indian_statistics_question(question)

    statistics_instructions = ""
    if stats_question:
        statistics_instructions = """
- This is a question about Indian dengue statistics.
- Use the Indian government state-wise table if it appears in the evidence.
- Extract the requested state's values directly from the table.
- Keep cases and deaths separate and preserve the year labels.
- Do not confuse the cases column with the deaths column.
- Do not substitute figures from the Americas or other countries.
- Do not invent missing numbers or calculate totals from incomplete data.
- If 2026 is marked with an asterisk, state that the reporting period
  or footnote must be verified before calling it a complete annual total.
- If the required government table is still absent, say that clearly.
"""

    graph_instructions = """
- Use graph relationships only when they are relevant to the question.
- Treat graph edges as supporting context, not as proof of a clinical claim.
- Preserve the direction and relation label of each graph edge.
- Do not mention unrelated graph entities.
- Do not claim a relationship exists unless it appears in the graph context.
"""

    prompt = f"""
You are a medical information assistant using retrieved documents.

Answer the user's question using the retrieved evidence below.
Do not invent medical facts, statistics, dates, or sources.

General instructions:
- Answer the specific question directly in simple English.
- Prefer evidence that directly answers the question.
- If evidence is insufficient, explain what is missing.
- Similarity scores are ranking signals, not proof of correctness.
- Do not claim that symptoms alone confirm a diagnosis.
- For urgent medical warning signs, advise prompt professional care.

{statistics_instructions}

Knowledge graph instructions:
{graph_instructions}

Question:
{question}

Retrieved document passages:
{chr(10).join(evidence)}

Knowledge graph relationships:
{chr(10).join(dict.fromkeys(all_graph_relationships))
    if all_graph_relationships else "No relevant graph relationships were matched."}

Give a concise, structured answer grounded in the evidence.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "concepts": sorted(all_concepts)[:MAX_GRAPH_CONTEXT],
        "graph_relationships": list(
            dict.fromkeys(all_graph_relationships)
        )[:MAX_GRAPH_CONTEXT],
        "model": OLLAMA_MODEL,
        "retrieved_count": len(sources),
        "warning": (
            "For educational information only. This response is not a "
            "medical diagnosis or a substitute for professional care."
        ),
    }
