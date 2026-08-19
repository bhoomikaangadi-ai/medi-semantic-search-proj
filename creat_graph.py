from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.models import OllamaModel
from camel.loaders import UnstructuredIO

from data_chunk import run_chunk
from utils import add_ge_emb, add_gid, merge_similar_nodes


def creat_metagraph(args, content, gid, n4j):

    # ============================================================
    # Ollama Qwen model
    # ============================================================

    uio = UnstructuredIO()

    ollama_model = OllamaModel(
        model_type="qwen2.5:3b",
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        url="http://127.0.0.1:11434/v1",
    )

    kg_agent = KnowledgeGraphAgent(
        model=ollama_model
    )

    # Keep original document
    whole_content = content

    # ============================================================
    # CHUNKING
    # ============================================================

    if args.grained_chunk:

        content = run_chunk(content)

    else:

        words = content.split()

        chunk_size = 300

        content = []

        for i in range(0, len(words), chunk_size):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            if chunk.strip():
                content.append(chunk)

    print("\n========================================")
    print("Total chunks:", len(content))
    print("========================================")

    # ============================================================
    # PROCESS EACH CHUNK
    # ============================================================

    for i, cont in enumerate(content):

        if not cont.strip():
            continue

        print("\n----------------------------------------")
        print(
            f"Processing chunk {i + 1}/{len(content)}"
        )
        print("----------------------------------------")

        print(
            "Chunk length:",
            len(cont)
        )

        # --------------------------------------------------------
        # Convert text to CAMEL element
        # --------------------------------------------------------

        element_example = uio.create_element_from_text(
            text=cont
        )

        print("Sending chunk to Qwen...")

        try:

            # ----------------------------------------------------
            # Extract graph using Qwen
            # ----------------------------------------------------

            graph_elements = kg_agent.run(
                element_example,
                parse_graph_elements=True
            )

            # ----------------------------------------------------
            # Validate response
            # ----------------------------------------------------

            if graph_elements is None:

                print(
                    "WARNING: Qwen returned None."
                )

                continue

            if not hasattr(
                graph_elements,
                "nodes"
            ):

                print(
                    "WARNING: Invalid graph response."
                )

                continue

            if len(
                graph_elements.nodes
            ) == 0:

                print(
                    "WARNING: No nodes extracted "
                    "from this chunk."
                )

                continue

            # ----------------------------------------------------
            # Print extracted graph
            # ----------------------------------------------------

            print(
                "Nodes:",
                len(graph_elements.nodes)
            )

            print(
                "Relationships:",
                len(
                    graph_elements.relationships
                )
            )

            # ----------------------------------------------------
            # Generate local Ollama embeddings
            # ----------------------------------------------------

            print(
                "Generating embeddings..."
            )

            graph_elements = add_ge_emb(
                graph_elements
            )

            # ----------------------------------------------------
            # Add graph ID
            # ----------------------------------------------------

            graph_elements = add_gid(
                graph_elements,
                gid
            )

            # ----------------------------------------------------
            # Store in Neo4j
            # ----------------------------------------------------

            n4j.add_graph_elements(
                graph_elements=[
                    graph_elements
                ]
            )

            print(
                "Graph elements added successfully."
            )

        except Exception as e:

            print(
                "ERROR processing chunk:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            continue

    # ============================================================
    # MERGE SIMILAR NODES
    # ============================================================

    if args.ingraphmerge:

        print(
            "\n========================================"
        )

        print(
            "Merging similar nodes..."
        )

        print(
            "========================================"
        )

        try:

            result = merge_similar_nodes(
                n4j,
                gid
            )

            print(
                "Merge completed."
            )

            print(
                result
            )

        except Exception as e:

            print(
                "Merge failed:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    # DO NOT generate summary here yet.
    #
    # We are keeping summary generation disabled because
    # the old implementation depended on OpenAI.
    #
    # Once the graph construction is working correctly,
    # we can add a completely local Ollama summary.

    print(
        "\n========================================"
    )

    print(
        "GRAPH CONSTRUCTION COMPLETED"
    )

    print(
        "========================================"
    )

    return n4j