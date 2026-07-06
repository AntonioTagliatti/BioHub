"""Read/write helpers for the competition's Zarr v3 image volumes.

Each sample is a single array at path ``0/`` with shape ``(T, Z, Y, X)``
(typically ``(100, 64, 256, 256)``), chunked one timepoint at a time and
compressed with blosc/zstd.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


def open_volume(zarr_path: str | Path) -> zarr.Array:
    """Open a sample's image array without loading it into memory.

    Parameters
    ----------
    zarr_path : path to the ``<sample>.zarr`` directory.

    Returns
    -------
    A lazily-loaded ``zarr.Array`` of shape ``(T, Z, Y, X)``.
    """
    store = zarr.open(str(zarr_path), mode="r")
    return store["0"]


def read_timepoint(zarr_path: str | Path, t: int) -> np.ndarray:
    """Load a single timepoint's ``(Z, Y, X)`` volume into memory."""
    arr = open_volume(zarr_path)
    return np.asarray(arr[t])


def iter_timepoints(zarr_path: str | Path):
    """Yield ``(t, volume)`` pairs for every timepoint in the sample."""
    arr = open_volume(zarr_path)
    for t in range(arr.shape[0]):
        yield t, np.asarray(arr[t])
