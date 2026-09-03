# ============================================================
# ENTITY FILTER
# ============================================================
#
# Basic pre-filter for extracted medical concepts.
#
# This does NOT perform UMLS validation.
#
# ============================================================

import re


# ============================================================
# DOCUMENT STOP WORDS
# ============================================================

STOP_WORDS = {
    "patient",
    "patients",
    "case",
    "cases",
    "chapter",
    "chapters",
    "source",
    "sources",
    "course",
    "days",
    "day",
    "figure",
    "fig",
    "textbox",
    "table",
    "tables",
    "page",
    "pages",
    "section",
    "sections",
}


# ============================================================
# CLEAN CONCEPT
# ============================================================

def clean_concept(concept):
    """
    Basic entity filtering.

    Expected input:

        {
            "name": "dengue"
        }

    Returns:

        True  -> keep
        False -> remove
    """

    if not isinstance(
        concept,
        dict
    ):
        return False

    name = concept.get(
        "name",
        ""
    )

    if not isinstance(
        name,
        str
    ):
        return False

    name = name.strip().lower()

    if not name:
        return False

    # --------------------------------------------------------
    # Normalize punctuation
    # --------------------------------------------------------

    name = re.sub(
        r"[^\w\s-]",
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    if not name:
        return False

    # --------------------------------------------------------
    # Document noise
    # --------------------------------------------------------

    if name in STOP_WORDS:
        return False

    # --------------------------------------------------------
    # Pure numbers
    # --------------------------------------------------------

    if name.isdigit():
        return False

    # --------------------------------------------------------
    # Single character
    # --------------------------------------------------------

    if len(name) == 1:
        return False

    return True