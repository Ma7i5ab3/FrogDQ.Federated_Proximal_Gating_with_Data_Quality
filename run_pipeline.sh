#!/usr/bin/env bash
# run_pipeline.sh — Full Quail experiment pipeline, looped over poisoning presets
#
# Steps (per poisoning preset):
#   2. Poison Data        (tier rates for this preset passed via CLI — config.yaml is never edited)
#   3. Data Preparation Pipeline (CP)
#   4. Saga++
#   5. Learn2Clean
#   6. DiffPrep
#   7. CtxPipe
#   8. Main (Optuna experiments)   -> results/<preset>pct/
#   9. Evaluation (Friedman test + critical-difference diagrams) -> results/<preset>pct/evaluation/
#  10. Cleanup of the regenerable poisoning/cleaning dirs (data_poisoned, data_cleaned_cp,
#      data_cleaned_saga, data_cleaned_learn2clean, data_cleaned_diffprep,
#      data_cleaned_ctxpipe) before moving on to the next preset.
#
# Step 1 (select datasets) runs once, before the preset loop — dataset selection
# doesn't depend on the poisoning level.
#
# The presets and their per-tier corruption rates are defined in config.yaml
# under `poisoning.presets` (run order + rates) and `poisoning.noise` (column
# shares) — this script reads them from there, it hardcodes nothing.
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --presets "10 20 30"  Space/comma-separated poisoning presets to run
#                         (default: config.yaml poisoning.presets.run)
#   --skip-select        Skip step 1 (select datasets)
#   --skip-poison        Skip step 2 (poison data) for every preset
#   --skip-cp            Skip step 3 (data preparation pipeline) for every preset
#   --skip-saga          Skip step 4 (Saga++) for every preset
#   --skip-learn2clean   Skip step 5 (Learn2Clean) for every preset
#   --skip-diffprep      Skip step 6 (DiffPrep) for every preset
#   --skip-ctxpipe       Skip step 7 (CtxPipe) for every preset
#   --skip-main          Skip step 8 (main experiments) for every preset
#   --skip-evaluate      Skip step 9 (evaluation) for every preset
#   --no-cleanup         Keep the data_poisoned/data_cleaned_* dirs between presets
#   --eval-output-dir D  Subdirectory (under each preset's results dir) for evaluation plots
#                        (default: evaluation, i.e. results/<preset>pct/evaluation)
#   -y, --yes            Auto-confirm the main experiment prompt
#   -h, --help           Show this help message
#
# Example — run only the 10% and 30% presets:
#   ./run_pipeline.sh --presets "10,30" -y

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
CONFIG="${SCRIPT_DIR}/config.yaml"

# ── colours ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

# ── poisoning presets ────────────────────────────────────────────────────────
# Presets and their per-tier rates live in config.yaml under
# `poisoning.presets` (run order + rates) and `poisoning.noise` (column
# shares) — see POISONING.md. The two readers below pull them from there.

# Presets to run, in order: config.yaml poisoning.presets.run
# (falls back to every preset defined under poisoning.presets.rates).
config_presets() {
    "$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
presets = (cfg.get("poisoning") or {}).get("presets") or {}
rates = {str(k): v for k, v in (presets.get("rates") or {}).items()}
if not rates:
    sys.exit("config.yaml has no poisoning.presets.rates block")
run = presets.get("run") or sorted(rates, key=float)
print(" ".join(str(p) for p in run))
PY
}

# poison_data.py tier flags for one preset (one token per line):
# rates from poisoning.presets.rates[<pct>], column shares from poisoning.noise.
config_preset_flags() {
    "$PYTHON" - "$CONFIG" "$1" <<'PY'
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
pct = sys.argv[2]
poisoning = cfg.get("poisoning") or {}
noise = poisoning.get("noise") or {}
rates = {str(k): v for k, v in ((poisoning.get("presets") or {}).get("rates") or {}).items()}

if pct not in rates:
    known = ", ".join(sorted(rates, key=float)) or "none"
    sys.exit(f"preset '{pct}' not found in config.yaml poisoning.presets.rates (defined: {known})")

tier = rates[pct] or {}
missing = [t for t in ("mild", "moderate", "heavy", "severe") if t not in tier]
if missing:
    sys.exit(f"preset '{pct}' in config.yaml is missing tier rate(s): {', '.join(missing)}")

flags = [
    "--mild-rate",     tier["mild"],
    "--moderate-rate", tier["moderate"],
    "--heavy-rate",    tier["heavy"],
    "--severe-rate",   tier["severe"],
]
# Column shares are shared across presets — read them from poisoning.noise.
for flag, key in (("--moderate-frac", "moderate_frac"),
                  ("--heavy-frac",    "heavy_frac"),
                  ("--severe-frac",   "severe_frac")):
    if key in noise:
        flags += [flag, noise[key]]

print("\n".join(str(f) for f in flags))
PY
}

# Regenerable poisoning/cleaning directories wiped between presets. Never
# touches "data" (the original input) or "results" (the accumulated output).
CLEANUP_DIRS=("data_poisoned" "data_cleaned_cp" "data_cleaned_saga" "data_cleaned_learn2clean"
              "data_cleaned_diffprep" "data_cleaned_ctxpipe")

# ── helpers ───────────────────────────────────────────────────────────────────
step_header() {
    local n="$1" label="$2"
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  Step ${n}: ${label}${RESET}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${RESET}"
}

preset_header() {
    echo ""
    echo -e "${BOLD}${MAGENTA}╔════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${MAGENTA}║  Poisoning preset: ${1}%  ->  results/${1}pct/${RESET}"
    echo -e "${BOLD}${MAGENTA}╚════════════════════════════════════════════════╝${RESET}"
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
SKIP_LEARN2CLEAN=false
SKIP_DIFFPREP=false
SKIP_CTXPIPE=false
SKIP_MAIN=false
SKIP_EVALUATE=false
NO_CLEANUP=false
AUTO_YES=false
EVAL_OUTPUT_SUBDIR="evaluation"
PRESETS_RAW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --presets)            PRESETS_RAW="$2"; shift ;;
        --skip-select)        SKIP_SELECT=true        ;;
        --skip-poison)        SKIP_POISON=true        ;;
        --skip-cp)            SKIP_CP=true            ;;
        --skip-saga)          SKIP_SAGA=true          ;;
        --skip-learn2clean)   SKIP_LEARN2CLEAN=true   ;;
        --skip-diffprep)      SKIP_DIFFPREP=true      ;;
        --skip-ctxpipe)       SKIP_CTXPIPE=true       ;;
        --skip-main)          SKIP_MAIN=true          ;;
        --skip-evaluate)      SKIP_EVALUATE=true      ;;
        --no-cleanup)         NO_CLEANUP=true         ;;
        --eval-output-dir)
            EVAL_OUTPUT_SUBDIR="$2"; shift
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

# ── preflight ─────────────────────────────────────────────────────────────────
[[ -x "$PYTHON" ]]  || die "Python not found at: $PYTHON"
[[ -f "$CONFIG" ]]  || die "config.yaml not found at: $CONFIG"
cd "$SCRIPT_DIR"

# Build the PRESETS array: --presets accepts space- or comma-separated values,
# otherwise the run order comes from config.yaml poisoning.presets.run.
if [[ -n "$PRESETS_RAW" ]]; then
    IFS=', ' read -r -a PRESETS <<< "$PRESETS_RAW"
else
    _presets_raw=""
    _presets_raw="$(config_presets)" || die "Could not read poisoning presets from ${CONFIG}"
    read -r -a PRESETS <<< "$_presets_raw"
fi
(( ${#PRESETS[@]} > 0 )) || die "No poisoning presets to run"

# Fail fast on an unknown/incomplete preset, before any long-running step.
for p in "${PRESETS[@]}"; do
    config_preset_flags "$p" >/dev/null || die "Invalid poisoning preset: ${p}"
done

echo -e "${BOLD}Quail Experiment Pipeline${RESET}"
echo -e "Working directory : ${SCRIPT_DIR}"
echo -e "Python            : ${PYTHON}"
echo -e "Config            : ${CONFIG}"
echo -e "Presets           : ${PRESETS[*]} (%)"

# ── step 1 — select datasets (once, independent of poisoning level) ───────────
step_header 1 "Select Datasets"
if $SKIP_SELECT; then
    step_skip 1
else
    "$PYTHON" scripts/select_datasets.py \
        --data-dir data \
        --config "$CONFIG" \
        --output "$CONFIG"
    step_ok 1
fi

# ── per-preset round ────────────────────────────────────────────────────────
run_round() {
    local pct="$1"
    local output_dir="results/${pct}pct"

    preset_header "$pct"

    # ── step 2 — poison data ───────────────────────────────────────────────
    step_header 2 "Poison Data (${pct}%)"
    if $SKIP_POISON; then
        step_skip 2
    else
        local flags_raw
        flags_raw="$(config_preset_flags "$pct")" \
            || die "Could not read preset ${pct} from ${CONFIG}"
        local -a poison_flags
        mapfile -t poison_flags <<< "$flags_raw"

        "$PYTHON" scripts/poison_data.py \
            --input_dir  data \
            --output_dir data_poisoned \
            --config     "$CONFIG" \
            "${poison_flags[@]}"
        step_ok 2
    fi

    # ── step 3 — data preparation pipeline (CP) ────────────────────────────
    step_header 3 "Data Preparation Pipeline (CP) (${pct}%)"
    if $SKIP_CP; then
        step_skip 3
    else
        "$PYTHON" scripts/data_preparation_pipeline.py \
            --input_dir  data_poisoned \
            --output_dir data_cleaned_cp \
            --config     "$CONFIG"
        step_ok 3
    fi

    # ── step 4 — saga++ ─────────────────────────────────────────────────────
    step_header 4 "Saga++ (${pct}%)"
    if $SKIP_SAGA; then
        step_skip 4
    else
        "$PYTHON" scripts/saga.py \
            --input_dir  data_poisoned \
            --output_dir data_cleaned_saga \
            --config     "$CONFIG"
        step_ok 4
    fi

    # ── step 5 — learn2clean ────────────────────────────────────────────────
    step_header 5 "Learn2Clean (${pct}%)"
    if $SKIP_LEARN2CLEAN; then
        step_skip 5
    else
        "$PYTHON" scripts/learn2clean.py \
            --input_dir  data_poisoned \
            --output_dir data_cleaned_learn2clean \
            --clean_dir  data \
            --config     "$CONFIG"
        step_ok 5
    fi

    # ── step 6 — diffprep ───────────────────────────────────────────────────
    step_header 6 "DiffPrep (${pct}%)"
    if $SKIP_DIFFPREP; then
        step_skip 6
    else
        "$PYTHON" scripts/diffprep.py \
            --input_dir  data_poisoned \
            --output_dir data_cleaned_diffprep \
            --clean_dir  data \
            --config     "$CONFIG"
        step_ok 6
    fi

    # ── step 7 — ctxpipe ────────────────────────────────────────────────────
    step_header 7 "CtxPipe (${pct}%)"
    if $SKIP_CTXPIPE; then
        step_skip 7
    else
        "$PYTHON" scripts/ctxpipe.py \
            --input_dir  data_poisoned \
            --output_dir data_cleaned_ctxpipe \
            --clean_dir  data \
            --config     "$CONFIG"
        step_ok 7
    fi

    # ── step 8 — main (optuna experiments) ─────────────────────────────────
    step_header 8 "Main (Optuna Experiments) (${pct}%)"
    if $SKIP_MAIN; then
        step_skip 8
    else
        _cfg_flag() {
            "$PYTHON" -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print('true' if cfg.get('$1', False) else 'false')"
        }
        MAIN_ARGS=(--config "$CONFIG" --output-dir "$output_dir")
        [ "$(_cfg_flag run_cp)"          = "true" ] && MAIN_ARGS+=(--run-cp)
        [ "$(_cfg_flag run_saga)"        = "true" ] && MAIN_ARGS+=(--run-saga)
        [ "$(_cfg_flag run_learn2clean)" = "true" ] && MAIN_ARGS+=(--run-learn2clean)
        [ "$(_cfg_flag run_diffprep)"    = "true" ] && MAIN_ARGS+=(--run-diffprep)
        [ "$(_cfg_flag run_ctxpipe)"     = "true" ] && MAIN_ARGS+=(--run-ctxpipe)

        if $AUTO_YES; then
            echo "y" | "$PYTHON" main.py "${MAIN_ARGS[@]}"
        else
            "$PYTHON" main.py "${MAIN_ARGS[@]}"
        fi
        step_ok 8
    fi

    # ── step 9 — evaluation (friedman test + cd diagrams) ──────────────────
    step_header 9 "Evaluation (Friedman Test + Critical-Difference Diagrams) (${pct}%)"
    if $SKIP_EVALUATE; then
        step_skip 9
    else
        "$PYTHON" scripts/evaluate.py \
            --config      "$CONFIG" \
            --results-dir "$output_dir" \
            --output-dir  "${output_dir}/${EVAL_OUTPUT_SUBDIR}"
        step_ok 9
    fi

    # ── step 10 — cleanup regenerable poisoning/cleaning dirs ──────────────
    step_header 10 "Cleanup (${pct}%)"
    if $NO_CLEANUP; then
        step_skip 10
    else
        for d in "${CLEANUP_DIRS[@]}"; do
            if [[ -d "$d" ]]; then
                rm -rf -- "$d"
                echo -e "${YELLOW}  Removed ${d}/${RESET}"
            fi
        done
        step_ok 10
    fi
}

for pct in "${PRESETS[@]}"; do
    run_round "$pct"
done

echo ""
echo -e "${BOLD}${GREEN}Pipeline complete for presets: ${PRESETS[*]} (%)${RESET}"
for pct in "${PRESETS[@]}"; do
    echo -e "${GREEN}  results/${pct}pct/${RESET}"
done
