import uuid
import requests

from camel.models import OllamaModel
from camel.agents import ChatAgent
from camel.messages import BaseMessage


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

CHAT_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# OLLAMA CHAT MODEL
# ============================================================

ollama_model = OllamaModel(
    model_type=CHAT_MODEL,
    model_config_dict={
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    url=f"{OLLAMA_BASE_URL}/v1",
)


# ============================================================
# SYSTEM PROMPTS
# ============================================================

sys_prompt_one = """
You are a medical Graph RAG assistant.

Answer the question using ONLY the provided medical
graph information.

Do not invent medical facts.

If the provided information is insufficient,
clearly say that the information is insufficient.

Give a concise and factual answer.
"""


sys_prompt_two = """
You are a medical Graph RAG assistant.

Improve the previous answer using the provided
graph references.

Use citations such as [1], [2], or [1][3]
when the references support the answer.

Do not invent medical information.

If the references do not support a statement,
do not include that statement.
"""


# ============================================================
# EMBEDDING USING OLLAMA
# ============================================================

def get_embedding(text):
    """
    Generate a 768-dimensional embedding using
    Ollama nomic-embed-text.
    """

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise ValueError(
            "Ollama returned no embeddings."
        )

    return embeddings[0]


# ============================================================
# FETCH NODE IDs
# ============================================================

def fetch_texts(n4j):

    query = """
    MATCH (n)
    WHERE n.id IS NOT NULL
    RETURN n.id AS id
    """

    return n4j.query(query)


# ============================================================
# ADD EMBEDDING TO ONE NODE
# ============================================================

def add_embeddings(n4j, node_id, embedding):

    query = """
    MATCH (n)
    WHERE n.id = $node_id
    SET n.embedding = $embedding
    """

    n4j.query(
        query,
        params={
            "node_id": node_id,
            "embedding": embedding,
        },
    )


# ============================================================
# ADD EMBEDDINGS TO EXISTING NODES
# ============================================================

def add_nodes_emb(n4j):

    nodes = fetch_texts(n4j)

    for node in nodes:

        node_id = node["id"]

        if not node_id:
            continue

        print(
            f"Generating embedding for: {node_id}"
        )

        embedding = get_embedding(node_id)

        add_embeddings(
            n4j,
            node_id,
            embedding,
        )

    print(
        "All node embeddings generated."
    )


# ============================================================
# ADD EMBEDDINGS TO NEW GRAPH ELEMENTS
# ============================================================

def add_ge_emb(graph_element):

    for node in graph_element.nodes:

        print(
            f"Generating embedding for node: {node.id}"
        )

        embedding = get_embedding(
            node.id
        )

        node.properties["embedding"] = embedding

    return graph_element


# ============================================================
# ADD GRAPH ID
# ============================================================

def add_gid(graph_element, gid):

    for node in graph_element.nodes:

        node.properties["gid"] = gid

    for rel in graph_element.relationships:

        rel.properties["gid"] = gid

    return graph_element


# ============================================================
# ADD SUMMARY
# ============================================================

def add_sum(n4j, content, gid):

    from summerize import process_chunks

    print("Adding summary...")

    summaries = process_chunks(content)

    create_sum_query = """
    CREATE (s:Summary {
        content: $sum,
        gid: $gid
    })
    RETURN s
    """

    result = n4j.query(
        create_sum_query,
        {
            "sum": summaries,
            "gid": gid,
        },
    )

    link_sum_query = """
    MATCH (s:Summary {gid: $gid})
    MATCH (n)
    WHERE n.gid = $gid
      AND NOT n:Summary

    CREATE (s)-[:SUMMARIZES]->(n)

    RETURN s, n
    """

    n4j.query(
        link_sum_query,
        {
            "gid": gid,
        },
    )

    return result


# ============================================================
# FIND INDEX OF LARGEST NUMBER
# ============================================================

def find_index_of_largest(nums):

    sorted_with_index = sorted(
        (num, index)
        for index, num in enumerate(nums)
    )

    return sorted_with_index[-1][1]


# ============================================================
# LOCAL QWEN LLM CALL
# ============================================================

def call_llm(system_prompt, user_prompt):

    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Medical Graph RAG Assistant",
            content=system_prompt,
        ),
        model=ollama_model,
    )

    message = BaseMessage.make_user_message(
        role_name="User",
        content=user_prompt,
    )

    response = agent.step(message)

    return response.msg.content


# ============================================================
# RETURN GRAPH CONTEXT
# ============================================================

def ret_context(n4j, gid):

    context = []

    query = """
    MATCH (n)
    WHERE n.gid = $gid
      AND NOT n:Summary

    WITH collect(n) AS nodes

    UNWIND nodes AS n
    UNWIND nodes AS m

    MATCH (n)-[r]-(m)

    WHERE n.gid = m.gid
      AND elementId(n) < elementId(m)
      AND NOT n:Summary
      AND NOT m:Summary

    RETURN
        n.id AS NodeId1,
        type(r) AS relType,
        m.id AS NodeId2
    """

    results = n4j.query(
        query,
        {
            "gid": gid,
        },
    )

    for row in results:

        context.append(
            f"{row['NodeId1']} "
            f"-[{row['relType']}]-> "
            f"{row['NodeId2']}"
        )

    return context


# ============================================================
# LINK CONTEXT
# ============================================================

def link_context(n4j, gid):

    context = []

    query = """
    MATCH (n)
    WHERE n.gid = $gid
      AND NOT n:Summary

    MATCH (n)-[:REFERENCE]->(m)

    WHERE NOT m:Summary

    RETURN
        n.id AS NodeId1,
        m.id AS NodeId2
    """

    results = n4j.query(
        query,
        {
            "gid": gid,
        },
    )

    reference_index = 1

    for row in results:

        context.append(
            f"Reference [{reference_index}]: "
            f"{row['NodeId1']} "
            f"REFERENCES "
            f"{row['NodeId2']}"
        )

        reference_index += 1

    return context


# ============================================================
# MERGE SIMILAR NODES
# ============================================================

def merge_similar_nodes(n4j, gid):

    merge_query = """
    WITH 0.85 AS threshold

    MATCH (n), (m)

    WHERE NOT n:Summary
      AND NOT m:Summary
      AND n.gid = $gid
      AND m.gid = $gid
      AND n <> m
      AND n.embedding IS NOT NULL
      AND m.embedding IS NOT NULL

    WITH n, m,
         gds.similarity.cosine(
             n.embedding,
             m.embedding
         ) AS similarity

    WHERE similarity > threshold

    WITH n, m

    CALL apoc.refactor.mergeNodes(
        [n, m],
        {
            properties: 'overwrite',
            mergeRels: true
        }
    )
    YIELD node

    RETURN count(*) AS merged
    """

    return n4j.query(
        merge_query,
        {
            "gid": gid,
        },
    )


# ============================================================
# CREATE REFERENCE LINKS BETWEEN TWO GRAPHS
# ============================================================

def ref_link(n4j, gid1, gid2):

    query = """
    MATCH (a)
    WHERE a.gid = $gid1
      AND NOT a:Summary
      AND a.embedding IS NOT NULL

    WITH collect(a) AS GraphA

    MATCH (b)
    WHERE b.gid = $gid2
      AND NOT b:Summary
      AND b.embedding IS NOT NULL

    WITH GraphA, collect(b) AS GraphB

    UNWIND GraphA AS n
    UNWIND GraphB AS m

    WITH n, m, 0.60 AS threshold

    WHERE n <> m

    WITH n, m,
         gds.similarity.cosine(
             n.embedding,
             m.embedding
         ) AS similarity

    WHERE similarity > threshold

    MERGE (m)-[:REFERENCE]->(n)

    RETURN n.id AS node1,
           m.id AS node2,
           similarity
    """

    return n4j.query(
        query,
        {
            "gid1": gid1,
            "gid2": gid2,
        },
    )


# ============================================================
# UUID
# ============================================================

def str_uuid():

    return str(uuid.uuid4())


# ============================================================
# SEARCH SIMILAR NODES
# ============================================================
def search_similar_nodes(n4j, query, top_k=5):
    """
    Find the most relevant medical graph nodes using
    Ollama embeddings, then retrieve their relationships.
    """

    # --------------------------------------------------
    # 1. Generate embedding for the user's question
    # --------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = get_embedding(query)

    # --------------------------------------------------
    # 2. Get all embedded nodes
    # --------------------------------------------------

    cypher = """
    MATCH (n)
    WHERE n.embedding IS NOT NULL
      AND n.id IS NOT NULL
      AND NOT n:Summary

    RETURN
        elementId(n) AS node_element_id,
        n.id AS id,
        labels(n) AS labels,
        n.gid AS gid,
        n.embedding AS embedding
    """

    results = n4j.query(cypher)

    scored = []

    # --------------------------------------------------
    # 3. Calculate cosine similarity
    # --------------------------------------------------

    for row in results:

        node_embedding = row["embedding"]

        if not node_embedding:
            continue

        dot_product = sum(
            a * b
            for a, b in zip(
                query_embedding,
                node_embedding
            )
        )

        query_norm = sum(
            a * a
            for a in query_embedding
        ) ** 0.5

        node_norm = sum(
            a * a
            for a in node_embedding
        ) ** 0.5

        if query_norm == 0 or node_norm == 0:
            continue

        similarity = (
            dot_product /
            (query_norm * node_norm)
        )

        scored.append({
            "node_element_id": row["node_element_id"],
            "id": row["id"],
            "labels": row["labels"],
            "gid": row["gid"],
            "similarity": similarity
        })

    # --------------------------------------------------
    # 4. Sort by relevance
    # --------------------------------------------------

    scored.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------
    # 5. Select top K nodes
    # --------------------------------------------------

    top_nodes = scored[:top_k]

    if not top_nodes:
        return []

    # --------------------------------------------------
    # 6. Retrieve relationships around top nodes
    # --------------------------------------------------

    relevant_data = []

    for item in top_nodes:

        node_id = item["id"]
        gid = item["gid"]

        relationship_query = """
        MATCH (n)
        WHERE n.id = $node_id
          AND n.gid = $gid
          AND NOT n:Summary

        OPTIONAL MATCH (n)-[r]-(m)

        WHERE NOT m:Summary

        RETURN
            n.id AS node_id,
            labels(n) AS node_labels,
            type(r) AS relationship,
            m.id AS connected_node
        """

        relationships = n4j.query(
            relationship_query,
            {
                "node_id": node_id,
                "gid": gid
            }
        )

        connections = []

        for rel in relationships:

            if rel["connected_node"] is not None:

                connections.append({
                    "relationship": rel["relationship"],
                    "connected_node": rel["connected_node"]
                })

        relevant_data.append({
            "id": node_id,
            "labels": item["labels"],
            "gid": gid,
            "similarity": item["similarity"],
            "connections": connections
        })

    return relevant_data

# ============================================================
# GET RELEVANT GRAPH CONTEXT
# ============================================================

def get_relevant_graph_context(
    n4j,
    query,
    top_k=5,
):

    search_results = search_similar_nodes(
        n4j,
        query,
        top_k=top_k,
    )

    context = []

    for result in search_results:

        node_id = result["id"]

        query = """
        MATCH (n)
        WHERE n.id = $node_id

        OPTIONAL MATCH (n)-[r]-(other)

        RETURN
            n.id AS node_id,
            type(r) AS relationship,
            other.id AS other_id
        """

        rows = n4j.query(
            query,
            {
                "node_id": node_id,
            },
        )

        found_relationship = False

        for row in rows:

            if (
                row["relationship"]
                and row["other_id"]
            ):

                context.append(
                    f"{row['node_id']} "
                    f"-[{row['relationship']}]-> "
                    f"{row['other_id']}"
                )

                found_relationship = True

        if not found_relationship:

            context.append(
                f"Node: {node_id}"
            )

    # Remove duplicates
    context = list(
        dict.fromkeys(context)
    )

    return context


# ============================================================
# ANSWER QUESTION USING GRAPH RAG
# ============================================================

def answer_question(
    n4j,
    question,
    top_k=5,
):

    print(
        "\n========================================"
    )

    print(
        "QUESTION:"
    )

    print(question)

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Retrieve relevant graph
    # --------------------------------------------------------

    print(
        "\nSearching medical graph..."
    )

    context = get_relevant_graph_context(
        n4j,
        question,
        top_k=top_k,
    )

    if not context:

        return (
            "I could not find relevant "
            "information in the medical graph."
        )

    print(
        f"Retrieved {len(context)} graph facts."
    )

    # --------------------------------------------------------
    # Prepare context
    # --------------------------------------------------------

    graph_context = "\n".join(
        context
    )

    # --------------------------------------------------------
    # Ask Qwen
    # --------------------------------------------------------

    user_prompt = f"""
Question:

{question}


Relevant medical graph information:

{graph_context}
"""

    print(
        "\nSending relevant information to Qwen..."
    )

    answer = call_llm(
        sys_prompt_one,
        user_prompt,
    )

    return answer


# ============================================================
# FINAL RESPONSE
# ============================================================

def get_response(n4j, gid, query, top_k=5):

    print("\n========================================")
    print("MEDICAL GRAPH SEARCH")
    print("========================================")

    results = search_similar_nodes(
        n4j,
        query,
        top_k=top_k
    )

    if not results:
        return "No relevant medical information was found."

    context = []

    print("\nRelevant graph information:\n")

    for i, result in enumerate(results, 1):

        print(
            f"{i}. {result['id']} "
            f"(similarity={result['similarity']:.4f})"
        )

        context.append(
            f"Node {i}: {result['id']}"
        )

        for connection in result["connections"]:

            relationship = connection["relationship"]
            connected_node = connection["connected_node"]

            context.append(
                f"  {result['id']} "
                f"--[{relationship}]--> "
                f"{connected_node}"
            )

    context_text = "\n".join(context)

    # --------------------------------------------------
    # Ask Qwen using only retrieved graph information
    # --------------------------------------------------

    user_prompt = (
        "Question:\n"
        + query
        + "\n\n"
        + "Relevant medical graph information:\n"
        + context_text
        + "\n\n"
        + "Answer the question using only this information."
    )

    print("\nGenerating answer using Qwen...")

    response = call_llm(
        sys_prompt_one,
        user_prompt
    )

    return response