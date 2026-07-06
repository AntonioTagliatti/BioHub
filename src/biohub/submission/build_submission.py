"""Build a competition-format submission.csv from a per-dataset tracking graph."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["id", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id", "dataset"]


def nodes_to_rows(nodes: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = nodes.copy()
    df["row_type"] = "node"
    df["source_id"] = -1
    df["target_id"] = -1
    df["dataset"] = dataset
    return df[["row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id", "dataset"]]


def edges_to_rows(edges: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = edges.copy()
    df["row_type"] = "edge"
    for col in ("node_id", "t", "z", "y", "x"):
        df[col] = -1
    df["dataset"] = dataset
    return df[["row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id", "dataset"]]


def build_submission(per_dataset_graphs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out_path: str | Path) -> Path:
    """Assemble and write submission.csv.

    Parameters
    ----------
    per_dataset_graphs : mapping of dataset name -> (nodes_df, edges_df),
        where nodes_df has columns [node_id, t, z, y, x] and edges_df has
        columns [source_id, target_id]. Every test dataset must be a key.
    out_path : destination path, e.g. outputs/submissions/submission.csv
    """
    chunks = []
    for dataset, (nodes, edges) in per_dataset_graphs.items():
        chunks.append(nodes_to_rows(nodes, dataset))
        chunks.append(edges_to_rows(edges, dataset))

    full = pd.concat(chunks, ignore_index=True)
    full.insert(0, "id", range(len(full)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return out_path
