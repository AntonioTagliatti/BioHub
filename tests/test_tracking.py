import pandas as pd

from biohub.tracking.linker import link_consecutive_frames


def test_link_consecutive_frames_simple_case():
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1],
            "t": [0, 1],
            "z": [10, 10],
            "y": [10, 10],
            "x": [10, 11],
        }
    )
    edges = link_consecutive_frames(nodes, max_distance_um=7.0)
    assert len(edges) == 1
    assert edges.iloc[0]["source_id"] == 0
    assert edges.iloc[0]["target_id"] == 1


def test_link_consecutive_frames_respects_gate():
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1],
            "t": [0, 1],
            "z": [0, 60],
            "y": [0, 0],
            "x": [0, 0],
        }
    )
    edges = link_consecutive_frames(nodes, max_distance_um=7.0)
    assert len(edges) == 0
