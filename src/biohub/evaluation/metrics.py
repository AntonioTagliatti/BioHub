"""Local scoring that mirrors the Kaggle metric, for offline validation.

score = adjusted_edge_jaccard + 0.1 * division_jaccard

This is a best-effort reference implementation — cross-check against the
competition's official scoring notebook/link before trusting it for
model selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PHYSICAL_SCALE = np.array([1.625, 0.40635, 0.40635])
MAX_MATCH_DISTANCE_UM = 7.0


def _scaled_coords(df: pd.DataFrame) -> np.ndarray:
    return df[["z", "y", "x"]].to_numpy(dtype=np.float32) * PHYSICAL_SCALE


def match_nodes_per_timepoint(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    max_distance_um: float = MAX_MATCH_DISTANCE_UM,
) -> pd.DataFrame:
    """Bipartite-match predicted to GT nodes within each timepoint.

    Returns a DataFrame with columns [t, pred_node_id, gt_node_id].
    """
    matches = []
    for t in sorted(set(pred_nodes["t"]) | set(gt_nodes["t"])):
        p = pred_nodes[pred_nodes["t"] == t]
        g = gt_nodes[gt_nodes["t"] == t]
        if p.empty or g.empty:
            continue

        dist = cdist(_scaled_coords(p), _scaled_coords(g))
        row_idx, col_idx = linear_sum_assignment(dist)

        for r, c in zip(row_idx, col_idx):
            if dist[r, c] <= max_distance_um:
                matches.append(
                    {
                        "t": t,
                        "pred_node_id": p.iloc[r]["node_id"],
                        "gt_node_id": g.iloc[c]["node_id"],
                    }
                )
    return pd.DataFrame(matches, columns=["t", "pred_node_id", "gt_node_id"])


def edge_jaccard(
    pred_edges: pd.DataFrame,
    gt_edges: pd.DataFrame,
    node_matches: pd.DataFrame,
    n_pred_nodes: int,
    n_gt_nodes: int,
) -> float:
    """TP / (TP + FP + FN) on edges, with a penalty for over-predicted nodes."""
    pred_to_gt = dict(zip(node_matches["pred_node_id"], node_matches["gt_node_id"]))

    gt_edge_set = {(int(s), int(t)) for s, t in gt_edges.itertuples(index=False)}
    mapped_pred_edges = {
        (pred_to_gt[s], pred_to_gt[t])
        for s, t in pred_edges.itertuples(index=False)
        if s in pred_to_gt and t in pred_to_gt
    }

    tp = len(mapped_pred_edges & gt_edge_set)
    fp = len(mapped_pred_edges - gt_edge_set)
    fn = len(gt_edge_set - mapped_pred_edges)

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Penalize over-predicting total node count.
    over_prediction = max(0, n_pred_nodes - n_gt_nodes)
    penalty = over_prediction / max(n_gt_nodes, 1)
    return max(0.0, jaccard - penalty)


def division_jaccard(
    pred_edges: pd.DataFrame,
    gt_edges: pd.DataFrame,
    node_matches: pd.DataFrame,
) -> float:
    """Micro-averaged Jaccard over division events (nodes with >=2 out-edges)."""
    pred_to_gt = dict(zip(node_matches["pred_node_id"], node_matches["gt_node_id"]))

    def out_edges(edges: pd.DataFrame) -> dict:
        d: dict = {}
        for s, t in edges.itertuples(index=False):
            d.setdefault(s, []).append(t)
        return d

    gt_out = out_edges(gt_edges)
    pred_out = out_edges(pred_edges)

    gt_divisions = {s: set(ts) for s, ts in gt_out.items() if len(ts) >= 2}

    tp = fp = fn = 0
    for gt_parent, gt_daughters in gt_divisions.items():
        pred_parent = next((p for p, g in pred_to_gt.items() if g == gt_parent), None)
        pred_daughters = {pred_to_gt.get(d) for d in pred_out.get(pred_parent, [])}
        pred_daughters.discard(None)

        if len(pred_out.get(pred_parent, [])) >= 2 and gt_daughters.issubset(pred_daughters):
            tp += 1
        else:
            fn += 1

    for pred_parent, pred_daughters in pred_out.items():
        if len(pred_daughters) >= 2:
            gt_parent = pred_to_gt.get(pred_parent)
            if gt_parent not in gt_divisions:
                fp += 1

    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def score_sample(pred_nodes, pred_edges, gt_nodes, gt_edges) -> dict:
    """Compute the combined competition score for a single sample."""
    matches = match_nodes_per_timepoint(pred_nodes, gt_nodes)
    ej = edge_jaccard(pred_edges, gt_edges, matches, len(pred_nodes), len(gt_nodes))
    dj = division_jaccard(pred_edges, gt_edges, matches)
    return {
        "adjusted_edge_jaccard": ej,
        "division_jaccard": dj,
        "score": ej + 0.1 * dj,
    }
