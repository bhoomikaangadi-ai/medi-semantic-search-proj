from utils import search_similar_nodes
from camel.storages import Neo4jGraph

# Use the same Neo4j connection settings
n4j = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="Medical@123"
)

query = "What medications were given to the patient?"

results = search_similar_nodes(
    n4j,
    query,
    top_k=5
)

print("\n==============================")
print("SEARCH RESULTS")
print("==============================")

for result in results:

    print("\nNode:", result["id"])
    print("Similarity:", result["similarity"])

    for connection in result["connections"]:

        print(
            " ",
            result["id"],
            "--[",
            connection["relationship"],
            "]-->",
            connection["connected_node"]
        )