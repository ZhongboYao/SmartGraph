# SmartGraph

A personal ML knowledge graph tool. Drop in PDF notes, review the extracted concepts, and watch your graph grow. Built for building and exploring a personal knowledge base over time.

## Features

- **PDF upload** — extracts key ML concepts and relationships via Claude
- **Review modal** — inspect and approve extracted nodes before they are added
- **Semantic merging** — uses sentence-transformer embeddings + FAISS to deduplicate nodes across uploads
- **Multiple graphs** — create, switch between, and rename independent graphs
- **Interactive editing** — add/delete nodes and edges, rename, recolor, annotate with notes
- **Auto-save** — every edit is persisted immediately

## Setup

**Prerequisites:** [Anaconda](https://www.anaconda.com) and an [Anthropic API key](https://console.anthropic.com).

```bash
# 1. Create and activate the conda environment
conda create -n smartgraph python=3.11
conda activate smartgraph

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Running

**macOS:** double-click `start.command`.

Or manually:
```bash
conda activate smartgraph
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000).

## Project Structure

```
SmartGraph/
├── backend/
│   ├── main.py              # FastAPI routes
│   ├── pdf_reading.py       # PDF text extraction
│   ├── graph_extraction.py  # Claude extraction + Pydantic models
│   ├── graph_convert.py     # KnowledgeGraph ↔ Cytoscape.js conversion
│   ├── graph_store.py       # Load / merge / save graph
│   ├── merge.py             # merge_by_label, merge_by_embedding
│   └── embedding.py         # Sentence-transformer embeddings + FAISS index
├── frontend/
│   └── index.html           # Single-page Cytoscape.js UI
├── data/
│   ├── graphs/              # Saved graphs (one JSON per graph)
│   └── test_data/           # Benchmark data for merge evaluation
├── test/
│   ├── merge_by_embedding_test.ipynb   # Merge benchmark notebook
│   └── baseline_pipeline_test.ipynb   # End-to-end pipeline test
├── log/                     # Extraction logs (notes, Claude outputs)
├── requirements.txt
└── start.command            # macOS launcher
```

## Architecture

Graph data is stored in **Cytoscape.js format** (`data/graphs/<name>.json`), which preserves visual properties (colors, notes). On load it is converted to a typed `KnowledgeGraph` for merge operations, then converted back and saved.

```
PDF → text → Claude → KnowledgeGraph → review modal
                                            ↓ approved subset
                              existing graph (load) → merge → save → Cytoscape
```

**Merge strategy:** `merge_by_label` for exact deduplication; `merge_by_embedding` for semantic deduplication using cosine similarity via FAISS.

## Future Work

- **Merge accuracy for abbreviations** — the embedding model scores acronym/full-name pairs (e.g. "RAG" vs "Retrieval Augmented Generation") with low similarity because it encodes each string independently. The current fix is to instruct Claude to always output full names. A more robust solution would use a cross-encoder, which sees both labels jointly and handles these cases naturally.

- **Merge accuracy for partial-label variants** — pairs like "Dropout" vs "Dropout Layer" or "Attention" vs "Attention Mechanism" score below the merge threshold (~0.7–0.9) even though they refer to the same concept. Suffix stripping or a cross-encoder reranker would address this.

- **Page chunking for long PDFs** — the current pipeline sends the full PDF text to Claude in one call. For documents longer than a few pages this may hit context limits. A natural extension is to split the PDF into N-page chunks, run extraction on each chunk independently, and merge the resulting candidate graphs before the review modal.

- **Batch PDF upload** — currently one PDF is processed per upload. Supporting multi-file selection would allow uploading an entire lecture series at once, with all extracted candidates combined into a single review modal before merging.
