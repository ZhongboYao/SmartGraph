from graph_extraction import KnowledgeGraph
from embedding import faiss_index

def merge_by_label(existing: KnowledgeGraph, new: KnowledgeGraph, **kwargs) -> KnowledgeGraph:
    """
    Merge two KnowledgeGraphs by exact label matching.
    Nodes and edges from `existing` are kept as-is. Nodes and edges from `new`
    are added only if their label / (source, target) pair is not already present.

    Args:
        existing:  The current KnowledgeGraph.
        new:       A freshly extracted KnowledgeGraph.
        **kwargs:  Ignored. Accepted so all merge functions share the same call signature,
                   allowing them to be swapped in GraphStore.merge() without changes.

    Returns:
        The updated existing KnowledgeGraph with new nodes and edges appended.
    """
    existing_labels = {node.label for node in existing.nodes}
    existing_edge_pairs = {(edge.source, edge.target) for edge in existing.edges}

    existing.nodes += [node for node in new.nodes if node.label not in existing_labels]
    existing.edges += [edge for edge in new.edges if (edge.source, edge.target) not in existing_edge_pairs]

    return existing

def merge_by_embedding(existing: KnowledgeGraph, new: KnowledgeGraph, **kwargs) -> KnowledgeGraph:
    """
    Merge two KnowledgeGraphs using embedding similarity to detect duplicates.

    For each new node, find its nearest neighbour in the existing graph.
    If similarity exceeds the threshold, treat them as the same concept:
    keep the existing node's label and remap any edges in `new` that referenced
    the new node to point to the existing node instead.
    Non-duplicate new nodes and their edges are then merged using exact label matching.

    Args:
        existing:  The current KnowledgeGraph.
        new:       A freshly extracted KnowledgeGraph.
        **kwargs:  Accepts 'threshold' (float, default 0.92) — cosine similarity
                   above which two nodes are considered duplicates.

    Returns:
        The updated existing KnowledgeGraph with deduplicated nodes and edges appended.
    """
    threshold = kwargs.get("threshold", 0.92)

    # Fall back to label merge if existing has no nodes (FAISS index would be empty)
    if not existing.nodes:
        existing.nodes += new.nodes
        existing.edges += new.edges
        return existing

    # Build embedding dicts: label → embedding
    existing_embeddings = {n.label: n.embedding for n in existing.nodes if n.embedding}
    new_embeddings      = {n.label: n.embedding for n in new.nodes if n.embedding}

    # Find nearest existing node for each new node
    results = faiss_index(queries=new_embeddings, documents=existing_embeddings)

    # Build replacements: new_label → existing_label for duplicates
    replacements = {
        query_label: doc_label
        for query_label, doc_label, sim in results
        if sim >= threshold
    }

    # Remap edge source/target in new graph using replacements
    for edge in new.edges:
        if edge.source in replacements:
            edge.source = replacements[edge.source]
        if edge.target in replacements:
            edge.target = replacements[edge.target]

    new.nodes = [n for n in new.nodes if n.label not in replacements]

    return merge_by_label(existing, new)
