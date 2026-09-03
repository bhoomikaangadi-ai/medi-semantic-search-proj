from utils import get_response
from camel.storages import Neo4jGraph


# ============================================================
# NEO4J CONNECTION
# ============================================================

n4j = Neo4jGraph(
    url="bolt://127.0.0.1:7687",
    username="neo4j",
    password="YOUR_NEO4J_PASSWORD"
)


# ============================================================
# IMPORTANT
# ============================================================
# Replace this with the gid of the graph you constructed.
#
# If your run.py prints the gid, use that value here.
# ============================================================

gid = "YOUR_GID"


# ============================================================
# QUESTION
# ============================================================

query = "What treatments were administered to the patient?"


# ============================================================
# GET ANSWER
# ============================================================

answer = get_response(
    n4j,
    gid,
    query,
    top_k=5
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print("FINAL ANSWER")
print("========================================")

print(answer)