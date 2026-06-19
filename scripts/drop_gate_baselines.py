"""
Drop all curr0_gate1 (gate-only, no curriculum) results so experiments can be re-run.

What this script removes:
  - optuna_studies.db : 4 studies whose names end in _curr0_gate1, plus every
                        dependent row in trials / trial_params / trial_values /
                        trial_user_attributes / trial_system_attributes /
                        trial_intermediate_values / trial_heartbeats /
                        study_directions / study_user_attributes /
                        study_system_attributes.
  - results/          : individual *_curr0_gate1_results.csv files.
  - all_experiments_results.csv : rows where use_gate=True AND use_curriculum=False.

Usage
-----
    python scripts/drop_gate_baselines.py [--results-dir PATH] [--dry-run]

    --results-dir  Path to the results directory (default: results/).
    --dry-run      Print what would be deleted without touching anything.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


STUDY_SUFFIX = "_curr0_gate1"
COMBINED_CSV = "all_experiments_results.csv"
DB_NAME = "optuna_studies.db"


def _get_gate_baseline_study_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT study_id, study_name FROM studies WHERE study_name LIKE ?",
        (f"%{STUDY_SUFFIX}",),
    ).fetchall()
    return rows


def drop_from_db(db_path: Path, dry_run: bool) -> None:
    conn = sqlite3.connect(db_path)
    try:
        studies = _get_gate_baseline_study_ids(conn)
        if not studies:
            print("DB: no curr0_gate1 studies found — nothing to delete.")
            return

        ids = [row[0] for row in studies]
        names = [row[1] for row in studies]
        trial_count = conn.execute(
            f"SELECT COUNT(*) FROM trials WHERE study_id IN ({','.join('?'*len(ids))})",
            ids,
        ).fetchone()[0]

        print(f"DB: found {len(studies)} studies, {trial_count} trials to delete:")
        for name in names:
            print(f"  - {name}")

        if dry_run:
            print("  [dry-run] skipping DB changes.")
            return

        placeholders = ",".join("?" * len(ids))
        conn.execute("BEGIN")
        for table in (
            "trial_user_attributes",
            "trial_system_attributes",
            "trial_params",
            "trial_values",
            "trial_intermediate_values",
            "trial_heartbeats",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE trial_id IN "
                f"(SELECT trial_id FROM trials WHERE study_id IN ({placeholders}))",
                ids,
            )
        conn.execute(f"DELETE FROM trials WHERE study_id IN ({placeholders})", ids)
        for table in ("study_directions", "study_user_attributes", "study_system_attributes"):
            conn.execute(f"DELETE FROM {table} WHERE study_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM studies WHERE study_id IN ({placeholders})", ids)
        conn.execute("COMMIT")
        print(f"DB: deleted {len(studies)} studies and {trial_count} trials.")
    finally:
        conn.close()


def drop_csv_files(results_dir: Path, dry_run: bool) -> None:
    files = sorted(results_dir.glob(f"*{STUDY_SUFFIX}_results.csv"))
    if not files:
        print("CSV files: none found — nothing to delete.")
        return

    print(f"CSV files: {len(files)} file(s) to delete:")
    for f in files:
        print(f"  - {f.name}")

    if dry_run:
        print("  [dry-run] skipping file deletion.")
        return

    for f in files:
        f.unlink()
    print(f"CSV files: deleted {len(files)} file(s).")


def drop_combined_csv_rows(results_dir: Path, dry_run: bool) -> None:
    combined = results_dir / COMBINED_CSV
    if not combined.exists():
        print(f"Combined CSV: {COMBINED_CSV} not found — skipping.")
        return

    df = pd.read_csv(combined)
    mask = (df["use_gate"] == True) & (df["use_curriculum"] == False)
    n_remove = int(mask.sum())

    if n_remove == 0:
        print(f"Combined CSV: no curr0_gate1 rows found — nothing to remove.")
        return

    print(f"Combined CSV: removing {n_remove} rows from {COMBINED_CSV} ({len(df) - n_remove} will remain).")

    if dry_run:
        print("  [dry-run] skipping CSV update.")
        return

    df[~mask].to_csv(combined, index=False)
    print(f"Combined CSV: updated ({len(df) - n_remove} rows kept).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to the results directory (default: results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without making any changes.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    db_path = results_dir / DB_NAME

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    if args.dry_run:
        print("=== DRY RUN — no changes will be made ===\n")

    drop_from_db(db_path, args.dry_run)
    print()
    drop_csv_files(results_dir, args.dry_run)
    print()
    drop_combined_csv_rows(results_dir, args.dry_run)

    if args.dry_run:
        print("\n=== DRY RUN complete ===")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
