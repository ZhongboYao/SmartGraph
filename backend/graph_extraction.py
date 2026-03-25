import os
import anthropic
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import json
from embedding import compute_embeddings

load_dotenv()

class Node(BaseModel):
    label: str
    embedding: list[float] = []


class Edge(BaseModel):
    source: str
    target: str
    relationship: str
    strength: str   


class KnowledgeGraph(BaseModel):
    name: str = ""
    nodes: list[Node]
    edges: list[Edge]


SYSTEM_PROMPT = """You are a knowledge graph extractor for machine learning and AI notes.

Given a note, extract:
1. Key concepts as nodes — use human-readable labels (e.g. "Contrastive Learning", "Mean Pooling").
   Prefer concepts, methods, algorithms, architectures, and models — things that exist as named
   ideas in the field, not just within this one note.
   Avoid nodes that describe steps, stages, or procedural details of a specific algorithm unless
   the step is itself a well-known named technique (e.g. "Beam Search" is fine; "Step 3: update weights" is not).
   Skip overly generic terms like "model" or "neural network" unless the note is specifically about them.
   Do not use abbreviations, always use full names.

2. Relationships between concepts as edges, with:
   - relationship: a short verb phrase (e.g. "trained with", "uses", "applied to", "type of")
   - strength:
       "strong" — the target is a subtype, specialisation, or direct evolution of the source
                  (e.g. ML → Clustering, DDPM → Stable Diffusion)
       "weak"   — the target uses the source as a component, tool, or sub-block
                  (e.g. Classification → Cross Entropy, Transformer → Attention)"""


def extract_graph(text: str, title: str, save_log: bool = False, log_path: Path = None) -> KnowledgeGraph:
    """
    Extract a knowledge graph from raw note text using Claude.

    Args:
        text:      Raw text content extracted from a PDF note.
        title:     Name for this extraction, saved as graph.name and used as the log filename.
        save_log:  If True, save the raw KnowledgeGraph to log_path as JSON.
        log_path:  Directory to save the log file. Required if save_log is True.

    Returns:
        A KnowledgeGraph with nodes (concepts) and edges (relationships).
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Extract the knowledge graph from this note:\n\n{text}"}
        ],
        output_format=KnowledgeGraph,
    )

    graph = response.parsed_output
    graph.name = title

    if save_log:
        log_path.mkdir(parents=True, exist_ok=True)
        output_file = log_path / (title + '_graph.json')
        with open(output_file, "w") as f:
            json.dump(graph.model_dump(), f, indent=4)

    return graph

def add_embeddings(graph: KnowledgeGraph, model_name: str = "all-MiniLM-L6-v2") -> KnowledgeGraph:
    """
    Compute and attach embeddings to each node in the graph.

    Args:
        graph:      A KnowledgeGraph whose nodes will be embedded.
        model_name: Sentence-transformers model to use for encoding.
                    Defaults to "all-MiniLM-L6-v2".

    Returns:
        The same KnowledgeGraph with node.embedding populated for every node.
    """
    labels = [node.label for node in graph.nodes]
    embeddings_dict = compute_embeddings(labels, model_name)
    for node in graph.nodes:
        node.embedding = embeddings_dict[node.label]
    return graph


