#!/usr/bin/env python
"""Run detect -> track -> divide -> submit end-to-end for every test sample.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from biohub.detection.detector import detect_all_timepoints
from biohub.io.zarr_io import iter_timepoints
from biohub.tracking.division import apply_divisions, find_split_candidates
from biohub.tracking.linker import link_consecutive_frames
from biohub.submission.build_submission import build_submission


def run_for_sample(zarr_path: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = detect_all_timepoints(
        iter_timepoints(zarr_path),
        min_distance=cfg["detection"]["min_distance_voxels"],
        intensity_percentile=cfg["detection"]["intensity_percentile_threshold"],
    )
    nodes.insert(0, "node_id", range(len(nodes)))

    edges = link_consecutive_frames(nodes, max_distance_um=cfg["tracking"]["max_link_distance_um"])

    candidates = find_split_candidates(nodes, edges)
    edges = apply_divisions(edges, candidates, cfg["tracking"]["division_score_threshold"])

    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    test_dir = Path(cfg["data"]["test_dir"])
    graphs = {}
    for sample_dir in sorted(test_dir.glob("*.zarr")):
        dataset_name = sample_dir.stem
        nodes, edges = run_for_sample(sample_dir, cfg)
        graphs[dataset_name] = (nodes, edges)
        print(f"{dataset_name}: {len(nodes)} nodes, {len(edges)} edges")

    out_path = build_submission(graphs, cfg["output"]["submission_path"])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
