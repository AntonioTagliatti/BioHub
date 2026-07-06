#!/usr/bin/env bash
# Download competition data into data/raw/.
# Fill in with the Kaggle CLI once the competition slug is known, e.g.:
#
#   kaggle competitions download -c <competition-slug> -p data/raw
#   unzip -q data/raw/<competition-slug>.zip -d data/raw
#
set -euo pipefail

COMPETITION_SLUG="${1:-}"
if [[ -z "$COMPETITION_SLUG" ]]; then
  echo "Usage: $0 <kaggle-competition-slug>" >&2
  exit 1
fi

mkdir -p data/raw
kaggle competitions download -c "$COMPETITION_SLUG" -p data/raw
unzip -q "data/raw/${COMPETITION_SLUG}.zip" -d data/raw
