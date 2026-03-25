from graph_extraction import KnowledgeGraph, Node, Edge
from pathlib import Path
import json

def to_cytoscape(graph: KnowledgeGraph, save_log: bool = False, log_path: Path = None) -> dict:
    """
    Convert a KnowledgeGraph to Cytoscape.js format.

    Args:
        graph:     A KnowledgeGraph with nodes and edges.
        save_log:  If True, save the Cytoscape.js JSON to log_path.
        log_path:  Directory to save the log file. Only used if save_log is True.

    Returns:
        A dict with "nodes" and "edges" in Cytoscape.js format, ready to load into Cytoscape.js.
    """
    nodes = [
        {"data": {"id": node.label, "label": node.label, "embedding": node.embedding}}
        for node in graph.nodes
    ]
    edges = [
        {"data": {
            "source": edge.source,
            "target": edge.target,
            "label": edge.relationship,
            "strength": edge.strength
        }}
        for edge in graph.edges
    ]

    cyto_graph = {"nodes": nodes, "edges": edges}

    if save_log:
        log_path.mkdir(parents=True, exist_ok=True)
        output_file = log_path / (graph.name + '_graph.json')
        with open(output_file, "w") as f:
            json.dump(cyto_graph, f, indent=4)

    return cyto_graph


def from_cytoscape(cyto: dict) -> KnowledgeGraph:
    """
    Convert a Cytoscape.js graph dict back to a KnowledgeGraph.
    Visual-only fields (e.g. color) are ignored — only knowledge fields are restored.

    Args:
        cyto: A dict with "nodes" and "edges" in Cytoscape.js format.

    Returns:
        A KnowledgeGraph with nodes and edges populated from the Cytoscape data.
    """
    nodes = [
        Node(
            label=n["data"]["label"],
            embedding=n["data"].get("embedding", [])
        )
        for n in cyto["nodes"]
    ]
    edges = [
        Edge(
            source=e["data"]["source"],
            target=e["data"]["target"],
            relationship=e["data"]["label"],
            strength=e["data"]["strength"]
        )
        for e in cyto["edges"]
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)