# ============================================================
# GRAPH RETRIEVER
# ============================================================

import json

GRAPH_FILE = "medical_knowledge_graph.json"

print("Loading Medical Knowledge Graph...")

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

# ------------------------------------------------------------
# Fast lookup dictionaries
# ------------------------------------------------------------

name_to_node = {}
id_to_node = {}

for node in nodes:

    name = node.get("canonical_name", node.get("name", "")).lower()

    if name:
        name_to_node[name] = node

    id_to_node[node["id"]] = node

print(f"Loaded {len(nodes)} nodes.")
print(f"Loaded {len(edges)} edges.")


# ============================================================
# FIND ENTITY
# ============================================================

def find_entity(query):

    query = query.lower().strip()

    if query in name_to_node:
        return name_to_node[query]

    matches = []

    for name, node in name_to_node.items():
        if query in name:
            matches.append(node)

    if matches:
        matches.sort(
            key=lambda n: len(
                n.get("canonical_name", n.get("name", ""))
            )
        )
        return matches[0]

    return None

# ============================================================
# SEARCH MULTIPLE ENTITIES
# ============================================================

def search_entities(query, limit=5):

    query = query.lower().strip()

    if not query:
        return []

    matches = []

    for name, node in name_to_node.items():

        if query == name:
            score = 3

        elif name.startswith(query):
            score = 2

        elif query in name:
            score = 1

        else:
            continue

        matches.append((score, len(name), node))

    matches.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [item[2] for item in matches[:limit]]

# ============================================================
# GET RELATED ENTITIES
# ============================================================

def get_related(node_id):

    results = []
    seen = set()

    for edge in edges:

        relation = edge.get("relation", "RELATED_TO")

        if edge["source"] == node_id:

            target = id_to_node.get(edge["target"])

            if target:

                key = (target["id"], relation)

                if key not in seen:

                    seen.add(key)

                    results.append({
                        "relation": relation,
                        "entity": target.get(
                            "canonical_name",
                            target.get("name")
                        ),
                        "type": target.get(
                            "ontology_class",
                            target.get("type")
                        )
                    })

        elif edge["target"] == node_id:

            source = id_to_node.get(edge["source"])

            if source:

                key = (source["id"], relation)

                if key not in seen:

                    seen.add(key)

                    results.append({
                        "relation": relation,
                        "entity": source.get(
                            "canonical_name",
                            source.get("name")
                        ),
                        "type": source.get(
                            "ontology_class",
                            source.get("type")
                        )
                    })

    results.sort(
        key=lambda x: (
            x["relation"],
            x["entity"]
        )
    )

    return results


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\nMedical Knowledge Graph Search")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nSearch: ").strip()

        if query.lower() == "exit":
            break

        entity = find_entity(query)

        if entity is None:
            print("Entity not found.")
            continue

        print("\n" + "=" * 70)
        print(f"Entity : {entity['canonical_name']}")
        print(f"Type   : {entity['ontology_class']}")
        print("=" * 70)

        related = get_related(entity["id"])

        for item in related:
            print(
                f"{item['relation']:20} → "
                f"{item['entity']} ({item['type']})"
            )

        print(f"\nTotal related entities: {len(related)}")