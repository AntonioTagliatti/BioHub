"""Read helpers for ground-truth ``.geff`` tracking graphs (train only).

Layout inside a ``.geff`` directory:
    nodes/ids                     -> node ID array
    nodes/props/{t,z,y,x}/values  -> integer centroid coords per node
    edges/ids                     -> (N, 2) array of (source_id, target_id)

Annotations are sparse; ``estimated_number_of_nodes`` in the ``.geff``
metadata gives an estimate of the true per-sample cell count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def read_geff(geff_path: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load a ground-truth graph.

    Returns
    -------
    nodes : DataFrame with columns [node_id, t, z, y, x]
    edges : (N, 2) int array of (source_id, target_id)
    meta  : dict of graph-level metadata (includes estimated_number_of_nodes)
    """
    root = zarr.open(str(geff_path), mode="r")

    node_ids = np.asarray(root["nodes/ids"])
    props = {
        prop: np.asarray(root[f"nodes/props/{prop}/values"])
        for prop in ("t", "z", "y", "x")
    }
    nodes = pd.DataFrame({"node_id": node_ids, **props})

    edges = np.asarray(root["edges/ids"]) if "edges/ids" in root else np.empty((0, 2), dtype=int)

    meta = dict(root.attrs)
    return nodes, edges, meta
