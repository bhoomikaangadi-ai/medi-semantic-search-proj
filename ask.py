from camel.storages import Neo4jGraph
from utils import search_similar_nodes, call_llm


# --------------------------------------------------
# Neo4j connection
# --------------------------------------------------

n4j = Neo4jGraph(
    url="bolt://127.0.0.1:7687",
    username="neo4j",
    password="Medical@123",
)


# --------------------------------------------------
# Retrieve relevant graph context
# --------------------------------------------------

def get_relevant_context(query, top_k=5):

    nodes = search_similar_nodes(
        n4j,
        query,
        top_k=top_k
    )

    context = []

    for node in nodes:

        node_id = node["id"]

        context.append(
            f"Node: {node_id}"
        )

        cypher = """
        MATCH (n)-[r]-(m)
        WHERE n.id = $node_id

        RETURN
            n.id AS source,
            type(r) AS relationship,
            m.id AS target
        """

        relationships = n4j.query(
            cypher,
            {
                "node_id": node_id
            }
        )

        for rel in relationships:

            context.append(
                f"{rel['source']} "
                f"--[{rel['relationship']}]--> "
                f"{rel['target']}"
            )

    return "\n".join(context)


# --------------------------------------------------
# Ask Qwen
# --------------------------------------------------

def ask_question(query):

    context = get_relevant_context(
        query,
        top_k=5
    )

    prompt = f"""
Question:
{query}

Relevant medical graph information:
{context}

Answer the question using ONLY the information
provided above.

Do not invent medical information.

If the information is insufficient, say:
"Insufficient information in the medical graph."

Give a concise answer.
"""

    system_prompt = """
You are a medical Graph RAG question-answering assistant.

Use only the retrieved graph information.

Do not hallucinate or add information
that is not present in the retrieved context.
"""

    print("\nGenerating answer using Qwen...\n")

    answer = call_llm(
        system_prompt,
        prompt
    )

    return answer


# --------------------------------------------------
# Interactive real-time question answering
# --------------------------------------------------

if __name__ == "__main__":

    print("========================================")
    print("Medical Graph RAG")
    print("Real-time Question Answering")
    print("Type 'exit' to stop")
    print("========================================")

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\n========================================")
        print("FINAL ANSWER")
        print("========================================")
        print(answer)