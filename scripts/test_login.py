from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://127.0.0.1:7687",
    auth=None
)

driver.verify_connectivity()
print("Connected successfully!")
driver.close()