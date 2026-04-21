#!/usr/bin/env bash
# Sweep every (language × method) combo on the dev split.
#
# Usage:   bash scripts/run_all.sh [vlm_model]
# Example: bash scripts/run_all.sh gpt-4o-mini
#
# Outputs: results/dev/{language}_dev_{method}_{vlm}.{jsonl,csv}
set -euo pipefail

VLM="${1:-gpt-4o-mini}"
LANGUAGES=(bribri guarani maya nahuatl wixarika)

for lang in "${LANGUAGES[@]}"; do
    echo
    echo "================================================================"
    echo "  $lang  /  pipeline  /  $VLM"
    echo "================================================================"
    uv run python -m americasnlp evaluate \
        --language "$lang" --method pipeline --vlm "$VLM" --split dev || true

    echo
    echo "  $lang  /  direct (zero-shot)  /  $VLM"
    echo "----------------------------------------------------------------"
    uv run python -m americasnlp evaluate \
        --language "$lang" --method direct --shots 0 --vlm "$VLM" --split dev || true

    echo
    echo "  $lang  /  direct (3-shot)  /  $VLM"
    echo "----------------------------------------------------------------"
    uv run python -m americasnlp evaluate \
        --language "$lang" --method direct --shots 3 --vlm "$VLM" --split dev || true
done
