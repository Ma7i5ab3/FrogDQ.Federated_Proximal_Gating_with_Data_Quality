#!/usr/bin/env bash
# run_pipeline.sh — Quail experiment pipeline, looped over poisoning presets
#
# Steps:
#   1. Select Datasets    (once, before the preset loop — dataset selection
#                          doesn't depend on the poisoning level)
#   Per poisoning preset:
#   2. Poison Data        (tier rates for this preset passed via CLI — config.yaml is never edited)
#   3. TODO: Federated training + aggregation   -> results/<preset>pct/
#   4. TODO: Evaluation                          -> results/<preset>pct/evaluation/
#   5. TODO: Cleanup of the regenerable data_poisoned/ dir before the next preset
#
# Until steps 3–5 exist, every preset overwrites data_poisoned/, so only the
# last preset run is left on disk. Run a single preset (--presets 40) to keep
# the one you need.
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
#   -h, --help           Show this help message
#
# Environment:
#   PYTHON               Python interpreter to use (default: ./.venv/bin/python)
#
# Example — poison only at the 30% preset:
#   ./run_pipeline.sh --presets 30

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
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
    echo -e "${BOLD}${MAGENTA}║  Poisoning preset: ${1}%${RESET}"
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
PRESETS_RAW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --presets)            PRESETS_RAW="$2"; shift ;;
        --skip-select)        SKIP_SELECT=true        ;;
        --skip-poison)        SKIP_POISON=true        ;;
        -h|--help)
            awk '/^# Usage/ { f = 1 } f && !/^#/ { exit } f { sub(/^# ?/, ""); print }' "$0"
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done

# ── preflight ─────────────────────────────────────────────────────────────────
[[ -x "$PYTHON" ]]  || die "Python not found at: $PYTHON (set the PYTHON env var)"
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

    # ── step 3 — TODO: federated training + aggregation ────────────────────
    # Partition each dataset's poisoned train split (data_poisoned/{ar,nar}/)
    # across clients, train locally with the quail layer, aggregate on the
    # server with the quality-aware rule, and write per-trial metrics to
    # results/${pct}pct/ (same CSV/Optuna layout the analysis/ scripts read).
    # Add a --skip-train flag once this exists.

    # ── step 4 — TODO: evaluation ──────────────────────────────────────────
    # Once step 3 writes results/${pct}pct/all_experiments_results.csv:
    #   "$PYTHON" scripts/evaluate.py \
    #       --config      "$CONFIG" \
    #       --results-dir "results/${pct}pct" \
    #       --output-dir  "results/${pct}pct/evaluation"

    # ── step 5 — TODO: cleanup ─────────────────────────────────────────────
    # Remove data_poisoned/ before the next preset, once step 3 has consumed
    # it (with a --no-cleanup flag to keep it). Never touch data/ or results/.
}

for pct in "${PRESETS[@]}"; do
    run_round "$pct"
done

echo ""
echo -e "${BOLD}${GREEN}Pipeline complete for presets: ${PRESETS[*]} (%)${RESET}"
if ! $SKIP_POISON; then
    echo -e "${GREEN}  data_poisoned/ holds the last preset run (${PRESETS[${#PRESETS[@]}-1]}%)${RESET}"
fi
