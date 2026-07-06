# data/

Not committed to git — populate locally (see `scripts/download_data.sh`).

Expected layout:

```
data/raw/train/<embryo_id>_<fov>.zarr/
data/raw/train/<embryo_id>_<fov>.geff/
data/raw/test/<embryo_id>_<fov>.zarr/
data/raw/sample_submission.csv
```

`train/` samples are paired `.zarr` (image) + `.geff` (ground-truth graph).
`test/` samples are `.zarr` only. Train and test are embryo-disjoint.
