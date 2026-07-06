"""Frame-to-frame linking of detections into tracks.

Baseline is nearest-neighbor assignment via optimal bipartite matching on
physically-scaled centroid distance, gated at a max distance — mirroring how
the competition matches nodes for scoring. Replace with a learned/optimal
tracker (e.g. an ILP-based linker) behind the same interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PHYSICAL_SCALE = np.array([1.625, 0.40635, 0.40635])  # z, y, x in µm/voxel


def _scaled_coords(df: pd.DataFrame) -> np.ndarray:
    return df[["z", "y", "x"]].to_numpy(dtype=np.float32) * PHYSICAL_SCALE


def link_consecutive_frames(
    nodes: pd.DataFrame,
    max_distance_um: float = 7.0,
) -> pd.DataFrame:
    """Greedy-optimal linking between each pair of consecutive timepoints.

    Parameters
    ----------
    nodes : DataFrame with columns [node_id, t, z, y, x].
    max_distance_um : gate distance in microns; pairs beyond this are never linked.

    Returns
    -------
    edges : DataFrame with columns [source_id, target_id].
    """
    edges = []
    timepoints = sorted(nodes["t"].unique())

    for t0, t1 in zip(timepoints[:-1], timepoints[1:]):
        frame0 = nodes[nodes["t"] == t0]
        frame1 = nodes[nodes["t"] == t1]
        if frame0.empty or frame1.empty:
            continue

        dist = cdist(_scaled_coords(frame0), _scaled_coords(frame1))
        row_idx, col_idx = linear_sum_assignment(dist)

        for r, c in zip(row_idx, col_idx):
            if dist[r, c] <= max_distance_um:
                edges.append(
                    {
                        "source_id": frame0.iloc[r]["node_id"],
                        "target_id": frame1.iloc[c]["node_id"],
                    }
                )

    return pd.DataFrame(edges, columns=["source_id", "target_id"])
