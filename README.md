# BioHub Cell Tracking

Algorithms to detect, track, and link cells across time in 3D microscopy data
(zebrafish embryo timelapses), including division detection and lineage
reconstruction. Built for a Kaggle competition benchmark.

## Problem

Each sample is a 3D+time volume (`T, Z, Y, X`, typically `100, 64, 256, 256`)
stored as Zarr v3. The task is to:

1. **Detect** cell centroids in every timepoint.
2. **Link** detections across consecutive timepoints into tracks.
3. **Identify divisions** (one node -> two or more daughter edges) to
   reconstruct full lineages.

Voxel scale: `z=1.625, y=0.40625, x=0.40625` µm/voxel.

## Evaluation

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

- **Edge Jaccard**: nodes matched to ground truth per-timepoint via optimal
  bipartite assignment on scaled centroid distance (max 7.0 µm). Edge TP
  requires both endpoints matched and connected by a GT edge. Adjusted for
  over-prediction of total node count.
- **Division Jaccard**: micro-averaged TP/FP/FN over GT divisions, where a
  predicted division must cover the pre-split node and touch both daughter
  lineages.

See `src/biohub/evaluation/metrics.py` for the local implementation used to
validate submissions before uploading to Kaggle.

## Repo layout

```
configs/            Experiment configs (paths, model/tracker hyperparameters)
data/                Local data (gitignored) — raw .zarr / .geff, processed caches
notebooks/           Exploration + the notebook submitted to Kaggle
scripts/             CLI entry points (download data, run full pipeline)
src/biohub/
    io/              Zarr + GEFF readers/writers
    detection/       Cell detection (per-timepoint centroid/segmentation)
    tracking/        Frame-to-frame linking + division detection
    evaluation/      Local scoring that mirrors the Kaggle metric
    submission/      Build submission.csv from a tracking graph
tests/               Unit tests for io, tracking, evaluation
outputs/             Local run artifacts (gitignored) — figures, submissions
```

## Setup

```bash
conda env create -f environment.yml
conda activate biohub
pip install -e .
```

or, with pip only:

```bash
pip install -r requirements.txt
```

## Data

Competition data isn't committed to this repo. Point `configs/default.yaml`
at your local copy of `train/` and `test/` (see `data/README.md`), or run:

```bash
bash scripts/download_data.sh
```

## Pipeline

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

This runs detection -> tracking -> division detection -> writes
`outputs/submissions/submission.csv` in the required node/edge row format,
then scores it locally against any available ground truth.

## Submission format

`submission.csv` contains two row types, grouped by `dataset`:

| row_type | node_id | t | z | y | x | source_id | target_id | dataset | id |
|---|---|---|---|---|---|---|---|---|---|
| node | int | int | int | int | int | -1 | -1 | name | idx |
| edge | -1 | -1 | -1 | -1 | -1 | int | int | name | idx |

`dataset` must match each test folder name (without `.zarr`), and every test
dataset must be present. `id` is a throwaway consecutive index.

## Notebook / Kaggle constraints

- Must be submitted as a notebook.
- CPU or GPU runtime <= 12 hours.
- No internet access during scored runs — pretrained weights must be
  downloaded ahead of time and checked into the notebook's input dataset.
- Output file must be named `submission.csv`.
