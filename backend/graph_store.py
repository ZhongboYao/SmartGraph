import json
from pathlib import Path
from merge import merge_by_label
from graph_extraction import KnowledgeGraph
from graph_convert import from_cytoscape, to_cytoscape

class GraphStore:
    """
    Manages the persistent knowledge graph.
    Reads and writes Cytoscape.js format on disk, but operates on KnowledgeGraph internally.
    """

    def __init__(self):
        self.graph = KnowledgeGraph(nodes=[], edges=[])

    def load(self, graph_path: Path):
        """
        Load an existing graph.json into memory.
        If the file does not exist, starts with an empty graph.

        Args:
            graph_path: Path to graph.json.
        """
        if graph_path.exists():
            with open(graph_path) as f:
                self.graph = from_cytoscape(json.load(f))
        else:
            self.graph = KnowledgeGraph(nodes=[], edges=[])

    def merge(self, new_graph: KnowledgeGraph, merge_fn=merge_by_label):
        """
        Merge a new KnowledgeGraph into the current graph.

        Args:
            new_graph: A freshly extracted KnowledgeGraph.
            merge_fn:  The merge strategy to use. Defaults to merge_by_label.
        """
        self.graph = merge_fn(self.graph, new_graph)

    def save(self, path: Path):
        """
        Write the current graph to graph.json.

        Args:
            path: Path to graph.json.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(to_cytoscape(self.graph), f, indent=4)