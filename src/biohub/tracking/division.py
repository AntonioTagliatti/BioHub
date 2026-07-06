"""Cell division detection.

A division is a node with two or more outgoing edges (one parent, two+
daughters). The baseline linker only produces one-to-one links, so this
module is where a second pass adds the extra daughter edge once a split is
detected (e.g. by allowing controlled fan-out within a distance/size gate,
or by a learned division classifier on local appearance).
"""
from __future__ import annotations

import pandas as pd


def find_split_candidates(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Placeholder for a division classifier.

    Given the current 1:1 edge set, identify parent nodes whose local
    neighborhood suggests an imminent split, and return candidate extra
    (source_id, target_id, score) rows to merge into the edge table.

    Currently returns an empty frame — implement your division model here.
    """
    return pd.DataFrame(columns=["source_id", "target_id", "score"])


def apply_divisions(edges: pd.DataFrame, candidates: pd.DataFrame, score_threshold: float = 0.5) -> pd.DataFrame:
    """Merge accepted division candidates into the edge table."""
    accepted = candidates[candidates["score"] >= score_threshold][["source_id", "target_id"]]
    return pd.concat([edges, accepted], ignore_index=True).drop_duplicates()
