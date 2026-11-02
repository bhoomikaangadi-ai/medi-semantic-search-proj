import os
import uuid
from dotenv import load_dotenv
import base64
import requests
import json
import re
from pathlib import Path
from functools import lru_cache

import faiss
import ollama
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GENERATED_IMAGES_DIR = ROOT / "static" / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


BACKEND_PUBLIC_URL = os.getenv(
    "BACKEND_PUBLIC_URL",
    "http://localhost:8000"
).rstrip("/")
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
OLLAMA_MODEL = "qwen2.5:7b"

INDEX_FILE = ROOT / "medical_clinical_faiss.index"
CHUNKS_FILE = ROOT / "medical_clinical_chunks.json"
GRAPH_FILE = ROOT / "medical_graph.json"

FINAL_SOURCE_COUNT = 4
CANDIDATE_COUNT = 50
GRAPH_HOPS = 1
MAX_GRAPH_CONTEXT = 20

# Ignore graph edges below this score to reduce weak links.
# A high score does not prove clinical relevance.
MIN_GRAPH_SCORE = 0.80

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

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]


def normalise(value):
    """Normalize text for comparisons."""
    return re.sub(
        r"[^a-z0-9]+", " ", str(value).lower()
    ).strip()


@lru_cache(maxsize=1)
def load_resources():
    """Load and validate the embedding model, index, chunks and graph."""

    for path in (INDEX_FILE, CHUNKS_FILE, GRAPH_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

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
        raise ValueError(
            "Clinical chunks must be a non-empty JSON list."
        )

    if not isinstance(graph, dict):
        raise ValueError(
            "Knowledge graph must be a JSON object."
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS has {index.ntotal} vectors but "
            f"{len(chunks)} chunks were loaded. "
            "Use matching index and chunk files."
        )

    dimension = model.get_embedding_dimension()

    if index.d != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"index={index.d}, model={dimension}."
        )

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            "Expected a FAISS inner-product index. "
            "Check how the index was created."
        )

    print(
        f"Loaded {len(chunks)} chunks and "
        f"{index.ntotal} vectors."
    )
    print_graph_summary(graph)

    return model, index, chunks, graph


def print_graph_summary(graph):
    nodes = graph.get("nodes", [])
    edges = graph.get(
        "relationships", graph.get("edges", [])
    )

    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0

    print(
        f"Knowledge graph: {node_count} nodes, "
        f"{edge_count} edges"
    )




def generate_medical_images(question, answer, evidence):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        print("Cloudflare credentials are missing.")
        return []

    question_text = str(question)[:500]
    answer_text = str(answer)[:2000]

    evidence_text = "\n".join(
        str(item) for item in evidence[:3]
    )[:1000]

    # Three different visual purposes for every answer.
    image_types = [
        {
            "type": "Medical overview",
            "focus": "A clear overview of the main medical topic and its important structures."
        },
        {
            "type": "Medical explanation",
            "focus": "The mechanism, process, causes or progression explained in the answer."
        },
        {
            "type": "Clinical features",
            "focus": "The symptoms, signs, diagnosis, prevention or treatment aspects mentioned in the answer."
        }
    ]

    generated_images = []

    for item in image_types:
        prompt = f"""
Create a medically accurate educational illustration.

Topic: {question_text}

Answer context: {answer_text[:700]}

Image purpose: {item["focus"]}

Requirements:
- Show only information relevant to the topic and answer.
- Explain this aspect clearly through a visual illustration.
- Use correct medical anatomy and terminology.
- Use a clean white background and clear details.
- Use simple Indian English for any labels.
- Include only facts and numbers explicitly supported by the answer.
- Never invent, alter or repeat medical facts or statistics.
- Avoid unnecessary text, decorative elements and misleading visuals.
"""

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/"
            f"@cf/black-forest-labs/flux-1-schnell"
        )

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json={"prompt": prompt},
                timeout=120
            )

            response.raise_for_status()
            result = response.json()

            image_data = result.get("result", {}).get("image")

            if not image_data:
                print(f"Image generation failed: {item['type']}")
                continue

            image_bytes = base64.b64decode(image_data)

            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = GENERATED_IMAGES_DIR / filename

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            generated_images.append({
                "image_url": (
                    f"{BACKEND_PUBLIC_URL}/generated-images/{filename}"
                ),
                "type": item["type"],
                "description": item["type"]
            })

            print(f"Generated image: {item['type']}")

        except Exception as error:
            print(f"Image generation error ({item['type']}): {error}")

    return generated_images



def graph_node_name(node):
    if not isinstance(node, dict):
        return None

    value = (
        node.get("canonical_name")
        or node.get("name")
        or node.get("label")
        or node.get("id")
    )

    return str(value).strip() if value is not None else None


def graph_node_id(node):
    if not isinstance(node, dict):
        return None

    value = node.get("id") or node.get("node_id")

    return str(value) if value is not None else None


def graph_edge_endpoints(edge):
    """Read common graph edge formats."""
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
    """Build reusable lookup structures for graph inspection."""
    raw_nodes = graph.get("nodes", [])
    raw_edges = graph.get(
        "relationships", graph.get("edges", [])
    )

    if not isinstance(raw_nodes, list):
        raw_nodes = []

    if not isinstance(raw_edges, list):
        raw_edges = []

    nodes_by_id = {}
    name_to_ids = {}
    adjacency = {}

    for node in raw_nodes:
        node_id = graph_node_id(node)
        name = graph_node_name(node)

        if node_id is not None:
            nodes_by_id[node_id] = node

        if name:
            lookup_id = node_id if node_id is not None else name
            name_to_ids.setdefault(
                normalise(name), set()
            ).add(lookup_id)

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
    q = normalise(question)

    india_match = any(
        normalise(term) in q for term in INDIA_TERMS
    )
    stats_match = any(
        normalise(term) in q for term in STATISTICS_TERMS
    )

    return india_match and stats_match


def is_indian_government_source(chunk):
    document = normalise(chunk.get("document", ""))
    source = normalise(chunk.get("source", ""))
    title = normalise(chunk.get("title", ""))
    text = normalise(chunk.get("text", ""))

    official_source = any(
        term in value
        for value in (source, document, title)
        for term in (
            "india govt",
            "ncvbdc",
            "dengue situation in india",
            "national center for vector borne diseases control",
            "national centre for vector borne diseases control",
        )
    )

    table_content = (
        "affected states" in text
        or "cases and deaths in the country" in text
    )

    return official_source or (
        table_content and (
            "2025" in text or "2026" in text
        )
    )


def is_state_statistics_chunk(chunk, requested_state=None):
    text = normalise(chunk.get("text", ""))

    if requested_state:
        if normalise(requested_state) not in text:
            return False

    has_table_heading = (
        "affected states" in text
        or "cases and deaths in the country" in text
    )

    has_years = sum(
        year in text
        for year in (
            "2021", "2022", "2023",
            "2024", "2025", "2026"
        )
    ) >= 2

    return has_table_heading or has_years


def extract_requested_state(question):
    q = normalise(question)

    for state in INDIAN_STATES:
        if normalise(state) in q:
            return state

    return None


def get_chunk_graph_concepts(
    chunk_id,
    graph,
    limit=MAX_GRAPH_CONTEXT,
):
    """
    Return unique, directly linked MENTIONS concepts.

    Duplicate edges are removed. Weak links are filtered.
    These links are graph annotations, not clinical proof.
    """
    if chunk_id is None:
        return []

    nodes = graph.get("nodes", [])
    edges = graph.get(
        "relationships", graph.get("edges", [])
    )

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return []

    node_lookup = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = graph_node_id(node)

        if node_id is not None:
            node_lookup[node_id] = node

    chunk_node_id = f"chunk_{chunk_id}"
    unique = {}

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        source, target = graph_edge_endpoints(edge)

        if source != chunk_node_id:
            continue

        if graph_edge_label(edge).upper() != "MENTIONS":
            continue

        node = node_lookup.get(target)

        if not node:
            continue

        name = graph_node_name(node)

        if not name:
            continue

        try:
            score = float(edge.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        if score < MIN_GRAPH_SCORE:
            continue

        key = (target, normalise(name))

        concept = {
            "id": target,
            "name": name,
            "semantic_type": node.get("semantic_type", []),
            "relationship": "MENTIONS",
            "score": score,
        }

        # Keep the highest-scoring duplicate edge.
        if key not in unique or score > unique[key]["score"]:
            unique[key] = concept

    results = sorted(
        unique.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:max(0, limit)]


def related_graph_context(
    chunk,
    graph,
    hops=GRAPH_HOPS,
):
    """
    Return direct chunk-to-concept links only.
    The hops argument is retained for compatibility.
    """
    chunk_id = chunk.get("chunk_id")

    if chunk_id is None:
        return []

    concepts = get_chunk_graph_concepts(
        chunk_id=chunk_id,
        graph=graph,
        limit=MAX_GRAPH_CONTEXT,
    )

    return [
        {
            "source_chunk": f"chunk_{chunk_id}",
            "target_id": concept["id"],
            "target_name": concept["name"],
            "relationship": concept["relationship"],
            "score": concept["score"],
            "semantic_type": concept["semantic_type"],
        }
        for concept in concepts
    ]


def retrieve_candidates(question, model, index, chunks):
    """Run semantic retrieval using the existing FAISS index."""
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match FAISS index."
        )

    count = min(CANDIDATE_COUNT, index.ntotal)

    similarities, indices = index.search(
        query_vector, count
    )

    results = []
    seen = set()

    for similarity, idx in zip(
        similarities[0], indices[0]
    ):
        idx = int(idx)

        if idx < 0 or idx >= len(chunks) or idx in seen:
            continue

        seen.add(idx)
        chunk = chunks[idx]

        if (
            not isinstance(chunk, dict)
            or not chunk.get("text")
        ):
            continue

        results.append({
            "index": idx,
            "similarity": float(similarity),
            "priority": 0.0,
        })

    return results


def select_evidence(question, candidates, chunks):
    """
    Rank semantic candidates and add matching official
    government table chunks for Indian statistics questions.
    """
    stats_question = is_indian_statistics_question(question)
    requested_state = extract_requested_state(question)

    by_index = {
        item["index"]: dict(item)
        for item in candidates
    }

    if stats_question:
        for idx, chunk in enumerate(chunks):
            if (
                not isinstance(chunk, dict)
                or not chunk.get("text")
            ):
                continue

            official = is_indian_government_source(chunk)

            table_match = is_state_statistics_chunk(
                chunk,
                requested_state=requested_state,
            )

            if official and table_match:
                by_index.setdefault(
                    idx,
                    {
                        "index": idx,
                        "similarity": 0.0,
                        "priority": 0.0,
                        "injected_by_rule": True,
                    },
                )

    ranked = list(by_index.values())

    for item in ranked:
        chunk = chunks[item["index"]]
        item.setdefault("injected_by_rule", False)

        if stats_question:
            if is_indian_government_source(chunk):
                item["priority"] = 2.0
            elif is_state_statistics_chunk(
                chunk, requested_state
            ):
                item["priority"] = 1.0

        item["rank_score"] = (
            item["similarity"]
            + item.get("priority", 0.0)
        )

    ranked.sort(
        key=lambda item: item["rank_score"],
        reverse=True,
    )

    selected = ranked[:FINAL_SOURCE_COUNT]

    if stats_question:
        official_items = [
            item
            for item in ranked
            if is_indian_government_source(
                chunks[item["index"]]
            )
            and is_state_statistics_chunk(
                chunks[item["index"]],
                requested_state=requested_state,
            )
        ]

        if official_items and not any(
            is_indian_government_source(
                chunks[item["index"]]
            )
            for item in selected
        ):
            if selected:
                selected[-1] = official_items[0]
            else:
                selected = [official_items[0]]

    deduplicated = []
    seen_indices = set()

    for item in selected:
        if item["index"] not in seen_indices:
            deduplicated.append(item)
            seen_indices.add(item["index"])

    return deduplicated


def answer_question(question: str) -> dict:
    """Answer a question using retrieved evidence and graph annotations."""
    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    model, index, chunks, graph = load_resources()

    candidates = retrieve_candidates(
        question, model, index, chunks
    )

    ranked = select_evidence(
        question, candidates, chunks
    )

    if not ranked:
        raise ValueError(
            "No valid medical passages were retrieved."
        )

    evidence = []
    sources = []
    all_graph_relationships = []
    all_concepts = {}

    for item in ranked:
        chunk = chunks[item["index"]]

        document = str(
            chunk.get("document", "Unknown document")
        )
        source = str(chunk.get("source", document))
        chunk_id = chunk.get(
            "chunk_id", item["index"]
        )
        chunk_text = str(chunk["text"])

        graph_facts = related_graph_context(
            chunk, graph
        )

        for fact in graph_facts:
            all_graph_relationships.append(fact)

            concept_key = (
                fact["target_id"],
                normalise(fact["target_name"]),
            )

            existing = all_concepts.get(concept_key)

            if (
                existing is None
                or fact["score"] > existing["score"]
            ):
                all_concepts[concept_key] = {
                    "id": fact["target_id"],
                    "name": fact["target_name"],
                    "semantic_type": fact["semantic_type"],
                    "source": source,
                    "score": fact["score"],
                }

        evidence.append(
            f"[Document: {document}; "
            f"Source: {source}; Chunk: {chunk_id}]\n"
            f"{chunk_text}"
        )

        sources.append({
            "title": str(
                chunk.get("title", document)
            ),
            "document": document,
            "source": source,
            "chunk_id": chunk_id,
            "score": round(item["similarity"], 4),
            "selected_by_rule": bool(
                item.get("injected_by_rule", False)
            ),
            "text": chunk_text,
            "graph_relationships": [
                f"{fact['source_chunk']} "
                f"--[{fact['relationship']}]--> "
                f"{fact['target_name']}"
                for fact in graph_facts
            ],
        })

    stats_question = is_indian_statistics_question(
        question
    )

    statistics_instructions = ""

    if stats_question:
        statistics_instructions = """
- This question concerns Indian dengue statistics.
- Prefer the Indian government state-wise table if present.
- Extract requested values directly from the supplied table.
- Keep case counts and death counts separate.
- Preserve the correct year labels.
- Do not use figures from other countries.
- Do not invent missing numbers or calculate totals from
  incomplete data.
- If 2026 has an asterisk, explain that its reporting period
  or footnote needs verification before treating it as a
  complete annual total.
- If the required government table is absent, say so clearly.
"""

    graph_instructions = """
- Graph edges are supporting annotations, not proof of a
  clinical claim.
- Use graph concepts only when relevant to the question.
- A MENTIONS edge does not establish a causal relationship.
- Do not claim a graph relationship that is not supplied.
- Prioritize the retrieved passage text for clinical answers.
"""

    graph_lines = list(dict.fromkeys(
        f"{fact['source_chunk']} "
        f"--[{fact['relationship']}]--> "
        f"{fact['target_name']} "
        f"(score: {fact['score']:.3f})"
        for fact in all_graph_relationships
    ))

    prompt = f"""
You are a medical information assistant using retrieved documents.

Answer the user's question using the evidence below.
Do not invent medical facts, statistics, dates, or sources.

General instructions:
- Answer directly in simple English.
- Prefer passages that directly support the answer.
- If evidence is insufficient, explain what is missing.
- Similarity scores are ranking signals, not proof of correctness.
- Symptoms alone cannot confirm a diagnosis.
- For urgent medical warning signs, advise prompt professional care.
- Distinguish source-supported facts from uncertainty.
- Do not claim a document is current unless its date supports that.
- Do not treat graph annotations as clinical evidence by themselves.

{statistics_instructions}

Knowledge graph instructions:
{graph_instructions}

Question:
{question}

Retrieved document passages:
{chr(10).join(evidence)}

Direct graph relationships:
{chr(10).join(graph_lines) if graph_lines else "No direct graph relationships matched these chunks."}

Give a concise, structured answer grounded in the retrieved evidence.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    answer_text = response["message"]["content"]

    images = generate_medical_images(
    question=question,
    answer=answer_text,
    evidence=evidence
    )

    concepts = sorted(
        all_concepts.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:MAX_GRAPH_CONTEXT]

    formatted_relationships = list(dict.fromkeys(
        f"{fact['source_chunk']} "
        f"--[{fact['relationship']}]--> "
        f"{fact['target_name']}"
        for fact in all_graph_relationships
    ))

    return {
        "answer": answer_text,
        "images": images,
        "sources": sources,
        "concepts": concepts,
        "graph_relationships": formatted_relationships[
            :MAX_GRAPH_CONTEXT
        ],
        "model": OLLAMA_MODEL,
        "retrieved_count": len(sources),
        "warning": (
            "For educational information only. This response "
            "is not a medical diagnosis or a substitute for "
            "professional care."
        ),
    }
