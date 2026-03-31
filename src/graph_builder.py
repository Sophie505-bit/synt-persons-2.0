from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(embeddings: List[List[float]], k: int = 10) -> Dict[int, List[int]]:
    X = np.array(embeddings, dtype=float)
    n = len(X)
    if n == 0:
        return {}
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    graph = {}
    for i in range(n):
        neigh = [j for j in indices[i].tolist() if j != i]
        graph[i] = neigh[:k]
    return graph
