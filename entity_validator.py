# ============================================================
# ENTITY VALIDATOR
# ============================================================
#
# Purpose:
#
# Validate SciSpaCy/UMLS entity candidates before they
# enter the medical ontology.
#
# This file MUST NOT import itself.
#
# Responsibilities:
#
#   1. Normalize text
#   2. Detect generic terms
#   3. Detect document noise
#   4. Compare extracted mention with UMLS canonical name
#   5. Validate UMLS linking score
#
# ============================================================

import re


# ============================================================
# GENERIC TERMS
# ============================================================

GENERIC_TERMS = {
    "patient",
    "patients",
    "person",
    "people",
    "individual",
    "individuals",
    "case",
    "cases",
    "subject",
    "subjects",
    "participant",
    "participants",
    "disease",
    "diseases",
    "illness",
    "condition",
    "conditions",
    "finding",
    "findings",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "problem",
    "problems",
}


# ============================================================
# DOCUMENT NOISE
# ============================================================

DOCUMENT_NOISE_TERMS = {
    "chapter",
    "chapters",
    "figure",
    "fig",
    "textbox",
    "table",
    "tables",
    "source",
    "sources",
    "course",
    "days",
    "day",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "page",
    "pages",
    "section",
    "sections",
    "phase",
    "phases",
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    """
    Normalize text for safe comparison.

    Example:

        "High-Grade Fever"
            ->
        "high grade fever"
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# GENERIC TERM CHECK
# ============================================================

def is_generic_term(text):
    """
    Return True if the extracted mention is too generic
    to be useful as a medical ontology entity.
    """

    normalized = normalize_text(text)

    if not normalized:
        return True

    if normalized in GENERIC_TERMS:
        return True

    return False


# ============================================================
# DOCUMENT NOISE CHECK
# ============================================================

def is_document_noise(text):
    """
    Detect obvious PDF/document extraction noise.
    """

    normalized = normalize_text(text)

    if not normalized:
        return True

    if normalized in DOCUMENT_NOISE_TERMS:
        return True

    # Pure number
    if normalized.isdigit():
        return True

    # Very short meaningless tokens
    if len(normalized) <= 1:
        return True

    return False


# ============================================================
# MENTION ↔ CANONICAL MATCH
# ============================================================

def mention_matches_canonical(
    mention,
    canonical_name
):
    """
    Determine whether an extracted mention is reasonably
    compatible with the UMLS canonical concept name.

    Examples:

        dengue
        Dengue Fever

        fever
        Fever

        myalgia
        Myalgia

    are considered compatible.

    IMPORTANT:
    This is intentionally conservative enough to reject
    clearly unrelated UMLS candidates.
    """

    mention = normalize_text(mention)

    canonical = normalize_text(canonical_name)

    if not mention or not canonical:
        return False

    # Exact match
    if mention == canonical:
        return True

    # Mention contained in canonical name
    if mention in canonical:
        return True

    # Canonical contained in mention
    if canonical in mention:
        return True

    mention_words = set(
        mention.split()
    )

    canonical_words = set(
        canonical.split()
    )

    # Common meaningful words
    common_words = (
        mention_words.intersection(
            canonical_words
        )
    )

    if common_words:
        return True

    return False


# ============================================================
# VALIDATE ENTITY
# ============================================================

def validate_entity(
    mention,
    canonical_name,
    score,
    min_score=0.80
):
    """
    Validate a UMLS candidate.

    Conditions:

        1. Mention must exist.
        2. Canonical name must exist.
        3. Score must meet minimum threshold.
        4. Mention must be reasonably compatible with
           the canonical UMLS concept.

    Returns:
        True / False
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not mention:
        return False

    if not canonical_name:
        return False

    # --------------------------------------------------------
    # Score validation
    # --------------------------------------------------------

    try:
        score = float(score)
    except (
        TypeError,
        ValueError
    ):
        return False

    if score < float(min_score):
        return False

    # --------------------------------------------------------
    # Generic mention
    # --------------------------------------------------------

    if is_generic_term(mention):
        return False

    # --------------------------------------------------------
    # Document noise
    # --------------------------------------------------------

    if is_document_noise(mention):
        return False

    # --------------------------------------------------------
    # Mention ↔ canonical validation
    # --------------------------------------------------------

    if not mention_matches_canonical(
        mention,
        canonical_name
    ):
        return False

    return True