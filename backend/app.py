
import json
from pathlib import Path
from typing import Any
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.qa_service import answer_question

ROOT = Path(__file__).resolve().parent.parent
ENTITIES_FILE = ROOT / "ontology_entities.json"
GRAPH_FILE = ROOT / "medical_knowledge_graph.json"

app = FastAPI(title="Medical Graph RAG API")
GENERATED_IMAGES_DIR = ROOT / "static" / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/generated-images/{filename}")
def get_generated_image(filename: str):
    image_path = GENERATED_IMAGES_DIR / filename

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)
@app.get("/debug-generated-images")
def debug_generated_images():
    return {
        "directory": str(GENERATED_IMAGES_DIR),
        "directory_exists": GENERATED_IMAGES_DIR.exists(),
        "files": [file.name for file in GENERATED_IMAGES_DIR.glob("*")]
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def read_json(path: Path) -> Any:
    try:
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )
    except (OSError, json.JSONDecodeError):
        return None


def list_from(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("entities", "nodes", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "entities_file_found": ENTITIES_FILE.exists(),
        "graph_file_found": GRAPH_FILE.exists(),
        "qa_adapter": "configured",
    }


@app.get("/api/stats")
def stats():
    entities_data = list_from(read_json(ENTITIES_FILE))

    return {
        "metrics": [
            {"label": "Source chunks", "value": "471",
             "note": "Reported pipeline run"},
            {"label": "Candidate mentions", "value": "22,292",
             "note": "SciSpaCy mentions"},
            {"label": "Accepted mentions", "value": "4,288",
             "note": "Passed extraction filters"},
            {"label": "Unique entities",
             "value": str(len(entities_data) or 682),
             "note": "Loaded entity list / reported count"},
        ]
    }


@app.get("/api/entities")
def entities(limit: int = 100, q: str = ""):
    items = list_from(read_json(ENTITIES_FILE))

    def label(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(
                item.get("canonical_name")
                or item.get("name")
                or item.get("label")
                or ""
            )
        return str(item)

    if q:
        items = [
            item for item in items
            if q.lower() in label(item).lower()
        ]

    limit = max(1, min(limit, 500))
    return {"count": len(items), "entities": items[:limit]}


@app.post("/api/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a medical question.",
        )

    try:
        return answer_question(question)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        # Details are logged in the backend terminal for troubleshooting.
        print(f"Medical QA request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=(
                "The medical QA service could not complete this request. "
                "Check the backend terminal and confirm Ollama is running."
            ),
        ) from exc
