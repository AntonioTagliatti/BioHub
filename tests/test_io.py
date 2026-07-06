"""Smoke tests for io helpers. Extend with a small fixture zarr/geff sample."""
import numpy as np


def test_placeholder_scale_constants():
    from biohub.tracking.linker import PHYSICAL_SCALE

    assert np.allclose(PHYSICAL_SCALE, [1.625, 0.40635, 0.40635])
