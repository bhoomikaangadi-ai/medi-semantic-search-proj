import os
import argparse

from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO

from dataloader import load_high
from data_chunk import run_chunk
from creat_graph import creat_metagraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import *
from nano_graphrag import GraphRAG, QueryParam


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("-simple", action="store_true")
parser.add_argument("-construct_graph", action="store_true")
parser.add_argument("-inference", action="store_true")
parser.add_argument("-grained_chunk", action="store_true")
parser.add_argument("-trinity", action="store_true")
parser.add_argument("-trinity_gid1", type=str)
parser.add_argument("-trinity_gid2", type=str)
parser.add_argument("-ingraphmerge", action="store_true")
parser.add_argument("-crossgraphmerge", action="store_true")

parser.add_argument("-dataset", type=str, default="mimic_ex")
parser.add_argument("-data_path", type=str, default="./dataset_test")
parser.add_argument(
    "-test_data_path",
    type=str,
    default="./dataset_ex/report_0.txt"
)

args = parser.parse_args()


# ============================================================
# Simple NanoGraphRAG Demo
# ============================================================

if args.simple:

    graph_func = GraphRAG(working_dir="./nanotest")

    with open("./dataset_ex/report_0.txt", "r", encoding="utf-8") as f:
        graph_func.insert(f.read())

    print(
        graph_func.query(
            "What is the main symptom of the patient?",
            param=QueryParam(mode="local")
        )
    )

else:

    # ========================================================
    # Neo4j Connection (Corrected)
    # ========================================================
    # Neo4j Connection

    n4j = Neo4jGraph(
    url="bolt://127.0.0.1:7687",   # use bolt instead of neo4j
    username="",
    password="",
    )

   
    print("Connected to Neo4j successfully.")

    # ========================================================
    # Construct Graph
    # ========================================================

    if args.construct_graph:

        if args.dataset == "mimic_ex":

            files = [

                file

                for file in os.listdir(args.data_path)

                if os.path.isfile(os.path.join(args.data_path, file))
            ]

            for file_name in files:

                file_path = os.path.join(args.data_path, file_name)

                content = load_high(file_path)

                gid = str_uuid()

                n4j = creat_metagraph(
                    args,
                    content,
                    gid,
                    n4j
                )

                if args.trinity:
                    link_context(n4j, args.trinity_gid1)

            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

    # ========================================================
    # Inference
    # ========================================================

    if args.inference:

        question = load_high("./prompt.txt")

        summary = process_chunks(question)

        gid = seq_ret(n4j, summary)

        response = get_response(
            n4j,
            gid,
            question
        )

        print("\nAnswer:")
        print(response)