"""
Graph utilities for upper-layer HRL based on distance-threshold adjacency.
- Uses NetworkX if available; otherwise falls back to NumPy-based adjacency.
- Designed for high-dynamic UAV scenarios (join/leave, fast motion).
"""
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import networkx as nx  # optional
    _HAS_NX = True
except Exception:
    _HAS_NX = False


def _pairwise_dist(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances (N,N)."""
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_distance_graph(
    positions: List[Tuple[float, float]],
    threshold: float,
    directed: bool = False,
    return_graph: bool = True,
):
    """
    Build adjacency by distance threshold.
    - positions: list of (x, y) or (x, y, z)
    - threshold: connect i<->j if dist(i,j) <= threshold
    - directed: if True, returns DiGraph (symmetry still enforced by threshold)
    Returns (adj_matrix, graph or None)
    """
    pos_arr = np.asarray(positions, dtype=np.float32)
    if pos_arr.ndim != 2 or pos_arr.shape[0] == 0:
        return None, None
    N = pos_arr.shape[0]
    d = _pairwise_dist(pos_arr)
    adj = (d <= float(threshold)).astype(np.int32)
    np.fill_diagonal(adj, 0)

    if not return_graph:
        return adj, None

    if _HAS_NX:
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(N))
        # Undirected edges for symmetric adjacency
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j] == 1:
                    G.add_edge(i, j)
        return adj, G
    else:
        return adj, None


def update_graph_dynamic(
    positions: List[Tuple[float, float]],
    threshold: float,
    graph: Optional[object] = None,
    join_ids: Optional[List[int]] = None,
    leave_ids: Optional[List[int]] = None,
    directed: bool = False,
):
    """
    Update or rebuild graph/adjacency under join/leave events.
    - If NetworkX graph provided, updates nodes/edges incrementally; otherwise rebuild adjacency.
    - positions index maps to node id; leave_ids are removed; join_ids assumed present in positions.
    Returns (adj_matrix, graph or None).
    """
    pos_arr = np.asarray(positions, dtype=np.float32)
    if pos_arr.ndim != 2 or pos_arr.shape[0] == 0:
        return None, None
    N = pos_arr.shape[0]

    if _HAS_NX and isinstance(graph, (nx.Graph, nx.DiGraph)):
        # Remove leaving nodes
        for nid in (leave_ids or []):
            if graph.has_node(nid):
                graph.remove_node(nid)
        # Ensure all current ids exist
        for nid in range(N):
            if not graph.has_node(nid):
                graph.add_node(nid)
        # Recompute edges by threshold
        graph.remove_edges_from(list(graph.edges()))
        d = _pairwise_dist(pos_arr)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] <= float(threshold):
                    graph.add_edge(i, j)
        # Build adjacency from graph
        adj = np.zeros((N, N), dtype=np.int32)
        for i, j in graph.edges():
            adj[i, j] = 1
            adj[j, i] = 1
        return adj, graph
    else:
        # Fallback: rebuild adjacency only
        d = _pairwise_dist(pos_arr)
        adj = (d <= float(threshold)).astype(np.int32)
        np.fill_diagonal(adj, 0)
        return adj, None


def smooth_adjacent(adj: np.ndarray, alpha: float = 0.8, prev_adj: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Exponential smoothing to reduce flapping edges in high dynamics.
    - alpha close to 1 favors previous adjacency.
    """
    if prev_adj is None:
        return adj
    return (alpha * prev_adj + (1 - alpha) * adj >= 0.5).astype(np.int32)