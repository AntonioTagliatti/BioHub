"""Per-timepoint cell detection.

Baseline is a local-maxima detector on a smoothed intensity volume; swap in
a trained model (e.g. a 3D U-Net / StarDist-style network) behind the same
interface once you have one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter


def detect_local_maxima(
    volume: np.ndarray,
    min_distance: int = 3,
    intensity_percentile: float = 99.0,
    smoothing_sigma: float = 1.0,
) -> pd.DataFrame:
    """Detect candidate cell centroids in a single ``(Z, Y, X)`` volume.

    Returns a DataFrame with columns [z, y, x] in voxel coordinates.
    """
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=smoothing_sigma)
    footprint = np.ones((min_distance,) * 3)
    local_max = maximum_filter(smoothed, footprint=footprint) == smoothed

    threshold = np.percentile(smoothed, intensity_percentile)
    mask = local_max & (smoothed >= threshold)

    coords = np.argwhere(mask)
    return pd.DataFrame(coords, columns=["z", "y", "x"])


def detect_all_timepoints(volume_iter, **kwargs) -> pd.DataFrame:
    """Run detection over every ``(t, volume)`` pair from an iterator.

    See ``biohub.io.zarr_io.iter_timepoints``. Returns a DataFrame with
    columns [t, z, y, x].
    """
    frames = []
    for t, vol in volume_iter:
        df = detect_local_maxima(vol, **kwargs)
        df.insert(0, "t", t)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["t", "z", "y", "x"]
    )
