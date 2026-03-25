from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def compute_embeddings(labels: list[str], model_name: str = "all-MiniLM-L6-v2") -> dict[str, list[float]]:
    """
    Compute embeddings for a list of node labels.

    Args:
        labels:     List of node label strings to embed.
        model_name: Sentence-transformers model to use.
                    Defaults to "all-MiniLM-L6-v2" (fast, 384-dim, good for short phrases).

    Returns:
        Dict mapping each label to its embedding as a plain Python list of floats,
        ready for JSON serialisation.
    """
    model = SentenceTransformer(model_name)
    vectors = model.encode(labels)
    return {label: vector.tolist() for label, vector in zip(labels, vectors)}


def faiss_index(queries: dict, documents: dict) -> list[tuple[str, str, float]]:
    """
    Find the nearest neighbour in `documents` for each entry in `queries` using FAISS IndexFlatIP
    (exact cosine similarity on L2-normalised vectors).

    Args:
        queries:   Dict of {label: embedding} for the new nodes being searched.
        documents: Dict of {label: embedding} for the existing nodes to search against.

    Returns:
        List of (query_label, nearest_doc_label, cosine_similarity) for every query.
    """
    if not queries or not documents:
        return []

    doc_keys   = list(documents.keys())
    query_keys = list(queries.keys())

    doc_vecs   = np.array(list(documents.values()), dtype="float32")
    query_vecs = np.array(list(queries.values()),   dtype="float32")

    faiss.normalize_L2(doc_vecs)
    faiss.normalize_L2(query_vecs)

    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)

    similarities, indices = index.search(query_vecs, k=1)   # shape: (n_queries, 1)

    return [
        (query_keys[i], doc_keys[indices[i][0]], float(similarities[i][0]))
        for i in range(len(query_keys))
    ]