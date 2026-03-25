import sys
from pathlib import Path

# Make sure imports from this folder work when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import json

from pdf_reading import content_extraction
from graph_extraction import extract_graph, add_embeddings
from graph_convert import to_cytoscape, from_cytoscape
from graph_store import GraphStore
from merge import merge_by_label, merge_by_embedding

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT        = Path(__file__).parent.parent          # project root
GRAPHS_DIR  = ROOT / "data" / "graphs"
LOG_NOTES   = ROOT / "log" / "extracted_notes"
LOG_CLAUDE  = ROOT / "log" / "graphs_from_claude"


def graph_path(name: str) -> Path:
    """Return the file path for a named graph."""
    return GRAPHS_DIR / f"{name}.json"


# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.get("/graphs")
def list_graphs():
    """Return the names of all saved graphs (file stems in data/graphs/)."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    names = [p.stem for p in sorted(GRAPHS_DIR.glob("*.json"))]
    return {"graphs": names}


@app.get("/graph/{name}")
def get_graph(name: str):
    """Return the named graph in Cytoscape.js format."""
    path = graph_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found.")
    with open(path) as f:
        return json.load(f)


@app.post("/graph/{name}")
async def save_graph(name: str, request: Request):
    """
    Save (or create) a named graph from Cytoscape.js format.

    Args:
        name:    The graph name (used as filename stem).
        request: HTTP request whose JSON body is {nodes, edges}.
    """
    body = await request.json()
    path = graph_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(body, f, indent=4)
    return {"status": "saved"}


@app.patch("/graph/{name}")
async def rename_graph(name: str, request: Request):
    """
    Rename a graph by moving its JSON file to a new stem name.

    Args:
        name:    Current graph name.
        request: HTTP request whose JSON body is {"new_name": "..."}.
    """
    body = await request.json()
    new_name = body.get("new_name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name is required.")
    old_path = graph_path(name)
    new_path = graph_path(new_name)
    if not old_path.exists():
        raise HTTPException(status_code=404, detail=f"Graph '{name}' not found.")
    if new_path.exists():
        raise HTTPException(status_code=409, detail=f"Graph '{new_name}' already exists.")
    old_path.rename(new_path)
    return {"status": "renamed", "name": new_name}


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """
    Accept a PDF upload, run the full extraction pipeline, merge into graph.json,
    and return the updated graph.

    Args:
        file: The uploaded PDF file.

    Returns:
        The updated graph in Cytoscape.js format.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the upload to a temp file so pdfplumber can open it by path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        title = Path(file.filename).stem
        content = content_extraction(tmp_path, store_extraction=True, log_path=LOG_NOTES)
        graph = extract_graph(content, title, save_log=True, log_path=LOG_CLAUDE)
        graph = add_embeddings(graph)

        # Return candidates as Cytoscape format for the review modal.
        # Do not merge yet — wait for /confirm.
        return to_cytoscape(graph)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/confirm/{name}")
async def confirm_graph(name: str, request: Request):
    """
    Receive the user-approved subset of extracted nodes and edges,
    merge into the named graph, and return the updated graph.

    The body is a Cytoscape.js dict (the approved subset from the review modal).
    It is converted to KnowledgeGraph, merged at that level, then saved back as Cytoscape.

    Args:
        name:    The graph to merge into.
        request: HTTP request whose JSON body is the approved graph in Cytoscape.js format.

    Returns:
        The updated graph in Cytoscape.js format.
    """
    body = await request.json()
    approved = from_cytoscape(body)

    store = GraphStore()
    store.load(graph_path(name))
    store.merge(approved, merge_by_embedding)
    store.save(graph_path(name))

    return to_cytoscape(store.graph)


# ── STATIC FILES (frontend) ───────────────────────────────────────────────────
# Served last so API routes above take priority.
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
