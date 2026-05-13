#!/usr/bin/env bash
# run_pipeline.sh — Full FrogDQ experiment pipeline
#
# Steps:
#   1. Select Datasets
#   2. Poison Data
#   3. Data Preparation Pipeline (CP)
#   4. Saga++
#   5. AutoGluon
#   6. Main (Optuna experiments)
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --skip-select        Skip step 1 (select datasets)
#   --skip-poison        Skip step 2 (poison data)
#   --skip-cp            Skip step 3 (data preparation pipeline)
#   --skip-saga          Skip step 4 (Saga++)
#   --skip-autogluon     Skip step 5 (AutoGluon)
#   --skip-main          Skip step 6 (main experiments)
#   --start-from N       Start from step N (1–6), skipping earlier steps
#   -y, --yes            Auto-confirm the main experiment prompt
#   -h, --help           Show this help message

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"
CONFIG="${SCRIPT_DIR}/config.yaml"

# ── colours ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

# ── helpers ───────────────────────────────────────────────────────────────────
step_header() {
    local n="$1" label="$2"
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  Step ${n}: ${label}${RESET}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"
}

step_ok() {
    echo -e "${GREEN}  ✓ Step $1 completed successfully.${RESET}"
}

step_skip() {
    echo -e "${YELLOW}  ⊘ Step $1 skipped.${RESET}"
}

die() {
    echo -e "${RED}${BOLD}ERROR: $*${RESET}" >&2
    exit 1
}

# ── argument parsing ──────────────────────────────────────────────────────────
SKIP_SELECT=false
SKIP_POISON=false
SKIP_CP=false
SKIP_SAGA=false
SKIP_AUTOGLUON=false
SKIP_MAIN=false
AUTO_YES=false
START_FROM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-select)    SKIP_SELECT=true    ;;
        --skip-poison)    SKIP_POISON=true    ;;
        --skip-cp)        SKIP_CP=true        ;;
        --skip-saga)      SKIP_SAGA=true      ;;
        --skip-autogluon) SKIP_AUTOGLUON=true ;;
        --skip-main)      SKIP_MAIN=true      ;;
        --start-from)
            START_FROM="$2"; shift
            [[ "$START_FROM" =~ ^[1-6]$ ]] || die "--start-from requires a number between 1 and 6"
            ;;
        -y|--yes)         AUTO_YES=true       ;;
        -h|--help)
            sed -n '/^# Usage/,/^[^#]/{ /^[^#]/d; s/^# \{0,1\}//; p }' "$0"
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done

# Apply --start-from
(( START_FROM > 1 )) && SKIP_SELECT=true
(( START_FROM > 2 )) && SKIP_POISON=true
(( START_FROM > 3 )) && SKIP_CP=true
(( START_FROM > 4 )) && SKIP_SAGA=true
(( START_FROM > 5 )) && SKIP_AUTOGLUON=true

# ── preflight ─────────────────────────────────────────────────────────────────
[[ -x "$PYTHON" ]]  || die "Python not found at: $PYTHON"
[[ -f "$CONFIG" ]]  || die "config.yaml not found at: $CONFIG"
cd "$SCRIPT_DIR"

echo -e "${BOLD}FrogDQ Experiment Pipeline${RESET}"
echo -e "Working directory : ${SCRIPT_DIR}"
echo -e "Python            : ${PYTHON}"
echo -e "Config            : ${CONFIG}"

# ── step 1 — select datasets ──────────────────────────────────────────────────
step_header 1 "Select Datasets"
if $SKIP_SELECT; then
    step_skip 1
else
    "$PYTHON" scripts/select_datasets.py \
        --data-dir data \
        --config "$CONFIG"
    step_ok 1
fi

# ── step 2 — poison data ──────────────────────────────────────────────────────
step_header 2 "Poison Data"
if $SKIP_POISON; then
    step_skip 2
else
    "$PYTHON" scripts/poison_data.py \
        --input_dir  data \
        --output_dir data_poisoned \
        --config     "$CONFIG"
    step_ok 2
fi

# ── step 3 — data preparation pipeline (CP) ───────────────────────────────────
step_header 3 "Data Preparation Pipeline (CP)"
if $SKIP_CP; then
    step_skip 3
else
    "$PYTHON" scripts/data_preparation_pipeline.py \
        --input_dir  data_poisoned \
        --output_dir data_cleaned_cp \
        --config     "$CONFIG"
    step_ok 3
fi

# ── step 4 — saga++ ───────────────────────────────────────────────────────────
step_header 4 "Saga++"
if $SKIP_SAGA; then
    step_skip 4
else
    "$PYTHON" scripts/saga.py \
        --input_dir  data_poisoned \
        --output_dir data_cleaned_saga \
        --config     "$CONFIG"
    step_ok 4
fi

# ── step 5 — autogluon ────────────────────────────────────────────────────────
step_header 5 "AutoGluon"
if $SKIP_AUTOGLUON; then
    step_skip 5
else
    "$PYTHON" scripts/autogluon.py \
        --config      "$CONFIG" \
        --output-dir  data_autogluon \
        --data-dir    data \
        --poisoned-dir data_poisoned
    step_ok 5
fi

# ── step 6 — main (optuna experiments) ───────────────────────────────────────
step_header 6 "Main (Optuna Experiments)"
if $SKIP_MAIN; then
    step_skip 6
else
    MAIN_ARGS=(
        --config "$CONFIG"
        --run-cp
        --run-saga
        --run-autogluon
    )

    # Patch stdin so the "Proceed?" prompt is auto-answered when -y is passed
    if $AUTO_YES; then
        echo "y" | "$PYTHON" main.py "${MAIN_ARGS[@]}"
    else
        "$PYTHON" main.py "${MAIN_ARGS[@]}"
    fi
    step_ok 6
fi

echo ""
echo -e "${BOLD}${GREEN}Pipeline complete.${RESET}"
