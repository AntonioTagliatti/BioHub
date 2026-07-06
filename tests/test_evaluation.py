import pandas as pd

from biohub.evaluation.metrics import score_sample


def test_score_sample_perfect_match():
    nodes = pd.DataFrame(
        {"node_id": [0, 1], "t": [0, 1], "z": [10, 10], "y": [10, 10], "x": [10, 10]}
    )
    edges = pd.DataFrame({"source_id": [0], "target_id": [1]})

    result = score_sample(nodes, edges, nodes, edges)
    assert result["adjusted_edge_jaccard"] == 1.0
    assert result["score"] >= 1.0
