
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import ollama


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

GRAPH_PATH = PROJECT_DIR / "medical_graph.json"
CANDIDATE_PATH = BASE_DIR / "candidate_inferences.json"

OUTPUT_PATH = BASE_DIR / "validated_inferences.json"

MODEL_NAME = "qwen2.5:7b"

# Rule-based validation settings
MIN_SUPPORTING_CHUNKS = 2
MIN_SUPPORTING_DOCUMENTS = 1

# Number of evidence chunks sent to Ollama per candidate
MAX_EVIDENCE_CHUNKS = 5

# Avoid sending too many candidates in one execution.
# Set to None to process all candidates.
MAX_CANDIDATES = 20

# Delay between Ollama requests
REQUEST_DELAY = 0.2

# Generic concepts that often create noisy associations.
GENERIC_CONCEPTS = {
    "patients",
    "patient",
    "diseases",
    "disease",
    "health",
    "medicine",
    "medical",
    "guidelines",
    "control",
    "prevention",
    "treatment",
    "management",
    "diagnosis",
    "people",
    "person",
    "study",
    "studies",
    "results",
    "data",
    "information",
}


# ==================================================
# LOAD JSON FILES
# ==================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# BUILD CHUNK LOOKUP
# ==================================================

def build_chunk_lookup(graph):

    chunk_lookup = {}

    for node in graph.get("nodes", []):

        if node.get("node_type") != "DocumentChunk":
            continue

        chunk_id = node.get("id")

        if not chunk_id:
            continue

        chunk_lookup[chunk_id] = {
            "text": node.get("text", ""),
            "document": node.get("document", ""),
            "source": node.get("source", ""),
            "document_path": node.get("document_path", ""),
            "chunk_id": node.get("chunk_id")
        }

    return chunk_lookup


# ==================================================
# NORMALIZE CONCEPT NAMES
# ==================================================

def normalize_name(name):

    return re.sub(
        r"\s+",
        " ",
        str(name).strip().lower()
    )


# ==================================================
# RULE-BASED VALIDATION
# ==================================================

def rule_based_validation(candidate):

    reasons = []

    concept_1 = candidate.get("concept_1", "")
    concept_2 = candidate.get("concept_2", "")

    concept_1_id = candidate.get("concept_1_id")
    concept_2_id = candidate.get("concept_2_id")

    support = candidate.get("supporting_chunks", 0)
    documents = candidate.get("supporting_documents", 0)

    # Check concept information
    if not concept_1 or not concept_2:
        reasons.append("Missing concept name")

    if not concept_1_id or not concept_2_id:
        reasons.append("Missing concept ID")

    if concept_1_id == concept_2_id:
        reasons.append("Both concepts have the same ID")

    if normalize_name(concept_1) == normalize_name(concept_2):
        reasons.append("Both concepts have the same name")

    # Check support
    if support < MIN_SUPPORTING_CHUNKS:
        reasons.append("Insufficient supporting chunks")

    if documents < MIN_SUPPORTING_DOCUMENTS:
        reasons.append("Insufficient supporting documents")

    # Check generic concepts
    generic_1 = normalize_name(concept_1) in GENERIC_CONCEPTS
    generic_2 = normalize_name(concept_2) in GENERIC_CONCEPTS

    if generic_1 and generic_2:
        reasons.append(
            "Both concepts are generic and may produce a noisy association"
        )

    # Calculate a simple rule score.
    # This is a filtering score, NOT a medical probability.
    score = 0.0

    if support >= 2:
        score += 0.30

    if support >= 5:
        score += 0.20

    if documents >= 2:
        score += 0.20

    if not generic_1 and not generic_2:
        score += 0.30

    score = round(min(score, 1.0), 2)

    # Hard failures
    if any(
        reason in reasons
        for reason in [
            "Missing concept name",
            "Missing concept ID",
            "Both concepts have the same ID",
            "Both concepts have the same name",
            "Insufficient supporting chunks",
            "Insufficient supporting documents"
        ]
    ):
        status = "rejected"

    elif generic_1 or generic_2:
        status = "needs_review"

    else:
        status = "passed"

    return {
        "status": status,
        "rule_score": score,
        "reasons": reasons
    }


# ==================================================
# PREPARE EVIDENCE
# ==================================================

def prepare_evidence(candidate, chunk_lookup):

    evidence = candidate.get("evidence", [])

    prepared = []
    seen = set()

    for item in evidence:

        chunk_id = item.get("chunk_id")

        if chunk_id is None:
            continue

        # Find the original document chunk
        # using its source, document path and chunk ID.
        matching_ids = [
            key for key in chunk_lookup
            if key.endswith(f"::{chunk_id}")
            and (
                not item.get("document_path")
                or item.get("document_path") in key
            )
        ]

        if not matching_ids:
            continue

        actual_chunk_id = matching_ids[0]

        if actual_chunk_id in seen:
            continue

        seen.add(actual_chunk_id)

        chunk = chunk_lookup[actual_chunk_id]

        text = chunk.get("text", "").strip()

        if not text:
            continue

        prepared.append({
            "chunk_node_id": actual_chunk_id,
            "document": chunk.get("document", ""),
            "source": chunk.get("source", ""),
            "document_path": chunk.get("document_path", ""),
            "chunk_id": chunk.get("chunk_id"),
            "text": text[:3500]
        })

        if len(prepared) >= MAX_EVIDENCE_CHUNKS:
            break

    return prepared


# ==================================================
# OLLAMA LLM VALIDATION
# ==================================================

def llm_validation(candidate, evidence):

    system_prompt = """
You are an evidence assessment assistant for a medical
knowledge graph research project.

Your task is to assess whether two medical concepts have
a meaningful relationship based ONLY on the supplied
document evidence.

IMPORTANT RULES:

1. Do not use outside medical knowledge.
2. Do not invent facts.
3. Do not assume causation from co-occurrence.
4. Do not assume treatment, prevention or diagnosis
   relationships unless the evidence explicitly supports them.
5. Distinguish meaningful relationships from concepts
   that merely appear in the same document section.
6. If evidence is insufficient or ambiguous, choose
   needs_review.
7. Do not provide medical advice.
8. Return valid JSON only.
"""

    user_prompt = f"""
Assess this candidate relationship.

Concept 1: {candidate.get("concept_1")}
Concept 2: {candidate.get("concept_2")}

Original relationship:
{candidate.get("relationship")}

Supporting chunks:
{candidate.get("supporting_chunks")}

Supporting documents:
{candidate.get("supporting_documents")}

Evidence:

{json.dumps(evidence, ensure_ascii=False, indent=2)}

Choose exactly one decision:

accepted:
The evidence clearly demonstrates a meaningful association
between the two concepts.

rejected:
The evidence shows that the association is misleading,
unrelated, or based only on incidental co-occurrence.

needs_review:
The evidence is ambiguous, incomplete, or insufficient.

Return JSON with these fields:

{{
    "decision": "accepted | rejected | needs_review",
    "relationship_description": "Short explanation",
    "evidence_summary": "What the supplied text supports",
    "supporting_evidence": ["Evidence observation"],
    "contradicting_evidence": ["Contradicting observation"],
    "confidence": 0.0,
    "reason": "Reason for the decision"
}}

Confidence is an assessment of evidence strength,
not a medical probability.

Never claim that clinical causation is established.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    content = response.message.content

    result = json.loads(content)

    allowed_decisions = {
        "accepted",
        "rejected",
        "needs_review"
    }

    decision = result.get("decision")

    if decision not in allowed_decisions:
        raise ValueError("Invalid LLM decision")

    confidence = float(result.get("confidence", 0))

    if not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence value")

    return result


# ==================================================
# HYBRID VALIDATION
# ==================================================

def validate_candidates(
    candidates,
    chunk_lookup
):

    validated_results = []

    total = len(candidates)

    for index, candidate in enumerate(candidates, start=1):

        print(
            f"\n[{index}/{total}] Validating: "
            f"{candidate.get('concept_1')} <-> "
            f"{candidate.get('concept_2')}"
        )

        rule_result = rule_based_validation(candidate)

        # Preserve original candidate information
        result = dict(candidate)

        result["rule_validation"] = rule_result

        # Hard rule failures do not need an LLM request.
        if rule_result["status"] == "rejected":

            result["validation_status"] = "rejected"
            result["llm_validation"] = None

            print("Rule validation: REJECTED")

            validated_results.append(result)
            continue

        # Recover original chunk text
        evidence = prepare_evidence(
            candidate,
            chunk_lookup
        )

        result["evidence_text"] = evidence

        # Require at least two text-bearing evidence chunks.
        if len(evidence) < 2:

            result["validation_status"] = "needs_review"

            result["llm_validation"] = {
                "decision": "needs_review",
                "reason": (
                    "Not enough supporting chunk text "
                    "could be recovered from medical_graph.json"
                )
            }

            print("Evidence unavailable: NEEDS REVIEW")

            validated_results.append(result)
            continue

        # Ask Ollama to validate evidence
        try:

            llm_result = llm_validation(
                candidate,
                evidence
            )

            result["llm_validation"] = llm_result

            decision = llm_result["decision"]

            # Hybrid final decision
            if rule_result["status"] == "needs_review":

                result["validation_status"] = "needs_review"

            elif decision == "accepted":

                result["validation_status"] = "accepted"

            elif decision == "rejected":

                result["validation_status"] = "rejected"

            else:

                result["validation_status"] = "needs_review"

            print(
                "LLM decision:",
                decision.upper()
            )

            print(
                "Final decision:",
                result["validation_status"].upper()
            )

        except Exception as error:

            result["validation_status"] = "needs_review"

            result["llm_validation"] = {
                "decision": "needs_review",
                "reason": str(error)
            }

            print("Ollama validation error:", error)
            print("Final decision: NEEDS REVIEW")

        validated_results.append(result)

        time.sleep(REQUEST_DELAY)

    return validated_results


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(results):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nResults saved to:", OUTPUT_PATH)


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    print("\n========== HYBRID INFERENCE VALIDATION ==========")

    print("\nLoading medical graph...")

    graph = load_json(GRAPH_PATH)

    print("Loading candidate inferences...")

    candidates = load_json(CANDIDATE_PATH)

    if not isinstance(candidates, list):
        raise ValueError(
            "candidate_inferences.json must contain a JSON array"
        )

    chunk_lookup = build_chunk_lookup(graph)

    print("\nTotal graph chunks:", len(chunk_lookup))
    print("Total candidates:", len(candidates))
    print("Ollama model:", MODEL_NAME)

    # Candidate inference generation sorts by support,
    # so process the strongest candidates first.
    candidates.sort(
        key=lambda item: item.get("supporting_chunks", 0),
        reverse=True
    )

    if MAX_CANDIDATES is not None:
        candidates = candidates[:MAX_CANDIDATES]

    results = validate_candidates(
        candidates,
        chunk_lookup
    )

    # Save all results, including rejected and review items.
    save_results(results)

    accepted = sum(
        item["validation_status"] == "accepted"
        for item in results
    )

    rejected = sum(
        item["validation_status"] == "rejected"
        for item in results
    )

    needs_review = sum(
        item["validation_status"] == "needs_review"
        for item in results
    )

    print("\n========== VALIDATION SUMMARY ==========")

    print("Total processed:", len(results))
    print("Accepted:", accepted)
    print("Rejected:", rejected)
    print("Needs review:", needs_review)

    print("\nValidation completed.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR:", error)
