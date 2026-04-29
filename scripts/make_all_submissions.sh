#
# Produce final test-set submission JSONLs for all 5 shared-task languages.
#
# Method choices below are fixed. The model is overridable via
# MODEL_OVERRIDE so we can smoke-test the whole flow with gpt-4o-mini before
# burning API credits on the strong-model run.
#
# Usage:
#   # Smoke-test: every language uses gpt-4o-mini, only 5 rows each.
#   MODEL_OVERRIDE=gpt-4o-mini LIMIT=5 bash scripts/make_all_submissions.sh
#
#   # Full smoke-test with the cheap model (still ~990 rows; takes a while).
#   MODEL_OVERRIDE=gpt-4o-mini bash scripts/make_all_submissions.sh
#
#   # Real submission run with the per-language strong-model picks below.
#   bash scripts/make_all_submissions.sh
#
#   # See exactly what would run without executing anything.
#   DRY_RUN=1 bash scripts/make_all_submissions.sh
#
# Outputs (default): results/submissions/{language}_{method-tag}_{model}.jsonl
#
# Env vars:
#   MODEL_OVERRIDE   Force every language to use this model.
#   LIMIT            Pass --limit N to each submit call (smoke testing).
#   WORKERS          --workers value (default 8).
#   DRY_RUN          Print commands instead of executing.
set -euo pipefail

# Parallel arrays (index-aligned). Coleman's T-5 message picks:
#   bribri  (bzd) -> pipeline / claude-sonnet-4-5
#   guarani (grn) -> direct 3-shot / claude-sonnet-4-5
#   maya    (yua) -> pipeline / claude-sonnet-4-5
#   nahuatl (nlv) -> pipeline / gpt-5
#   wixarika(hch) -> direct 3-shot / claude-sonnet-4-5
LANGUAGES=(bribri            guarani           maya              nahuatl  wixarika)
METHODS=(  pipeline          direct            pipeline          pipeline direct)
MODELS=(   claude-sonnet-4-5 claude-sonnet-4-5 claude-sonnet-4-5 gpt-5    claude-sonnet-4-5)
SHOTS=(    0                 3                 0                 0        3)

WORKERS="${WORKERS:-8}"
SUBMIT_LIMIT_FLAG=""
if [[ -n "${LIMIT:-}" ]]; then
    SUBMIT_LIMIT_FLAG="--limit ${LIMIT}"
fi

run() {
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "+ $*"
    else
        # shellcheck disable=SC2294
        eval "$@"
    fi
}

for i in "${!LANGUAGES[@]}"; do
    lang="${LANGUAGES[$i]}"
    method="${METHODS[$i]}"
    model="${MODEL_OVERRIDE:-${MODELS[$i]}}"
    shots="${SHOTS[$i]}"

    extra_flags=""
    if [[ "$method" == "direct" ]]; then
        extra_flags="--shots ${shots}"
    fi

    echo
    echo "================================================================"
    echo "  $lang  /  $method  /  $model  ${SUBMIT_LIMIT_FLAG}"
    echo "================================================================"

    run uv run python -m americasnlp submit \
        --split test \
        --language "$lang" \
        --method "$method" \
        --vlm "$model" \
        --workers "$WORKERS" \
        $extra_flags \
        $SUBMIT_LIMIT_FLAG
done

echo
echo "Done. Submissions in results/submissions/. Validate with:"
echo "  python3 scripts/validate_submission.py --all"
