from camel.storages import Neo4jGraph


n4j = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="Medical@123"
)


print("\n========== NODE COUNT ==========")

result = n4j.query("""
MATCH (n)
RETURN count(n) AS count
""")

print(result)


print("\n========== RELATIONSHIP COUNT ==========")

result = n4j.query("""
MATCH ()-[r]->()
RETURN count(r) AS count
""")

print(result)


print("\n========== NODES ==========")

result = n4j.query("""
MATCH (n)
RETURN n.id AS id,
       labels(n) AS labels,
       n.gid AS gid
LIMIT 30
""")

for row in result:
    print(row)


print("\n========== RELATIONSHIPS ==========")

result = n4j.query("""
MATCH (a)-[r]->(b)
RETURN a.id AS source,
       type(r) AS relationship,
       b.id AS target
LIMIT 30
""")

for row in result:
    print(row)