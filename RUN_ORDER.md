# Execution order

1. Use the existing `.venv`; direct versions are pinned in `requirements.txt`
   and the full environment is recorded in `requirements-lock.txt`.
2. Validate code: `.venv/bin/python -m unittest discover -s tests -v`.
3. Use the validated `data/snapshots/2026-09-14-v2/monthly.csv`. To acquire new
   observations, run `run_study.py download --as-of DATE --output NEW_DIRECTORY`.
   Downloads and validated snapshots are not overwritten.
4. Run the original benchmark independently:
   `.venv/bin/python run_study.py run --configuration legacy --output results/legacy-2026-09-14`.
   Its isolated workspace runs preprocessing, quantum simulation, LSTM,
   classical reservoirs, and master comparison in dependency order. Source
   data, coupling matrices, and historical artifacts stay unchanged.
5. Run the modern benchmark:
   `.venv/bin/python run_study.py run --output results/modern-2026-09-14 --workers 4 --threads 2`.
   This covers all models, both windows, and seeds 0–4. Deterministic models
   run once. Completed origins resume automatically; use a new directory if
   source, input data, or configuration changes.
6. Regenerate analysis from completed checkpoints:
   `.venv/bin/python run_study.py report --run results/modern-2026-09-14`.
   Incomplete runs fail validation instead of generating a misleading final
   report. Explicit failed-fit records are retained and handled transparently.
7. Read the generated report or open `Current_Market_Results.ipynb` with the
   desired `RUN_DIR`. The notebook validates run identity and prediction dates.

## Artifact roles

- `data/snapshots/.../sources.json` and `manifest.json`: retrieval and data hashes.
- Run `manifest.json`: configuration, scaler, model source hashes, environment.
- `checkpoints/`: atomic per-model/seed/window/origin records, including failures.
- `cache/`: regenerable matrices/features retained locally and excluded from Git.
- `predictions.csv`: combined date-indexed forecasts, including unscored month.
- `metrics.csv`, `metrics_by_seed.csv`, `losses.csv`: accuracy evidence.
- `mcs.csv`, `loss_difference_ci.csv`: dependence-aware comparisons.
- `common_dates_metrics.csv`: incomplete models compared on shared valid dates.
- `failures.csv`: failed fits and numerical error messages.
- `report.md`, plots, `report_receipt.json`: artifact-only results and checksums.

`modern-pilot`, `modern-pilot-v2`, and the first processed snapshot are diagnostic
artifacts, not final evaluation results. They are preserved and described in the
audit. The final run uses only the v2 snapshot and its frozen configuration.

## Source revision and published reports

Both completed experiments used `6e7ddf0`; `execution_revision.json` verifies
their source hashes against that commit. Later maintenance changed a
trailing newline in the exact simulator, with its syntax tree verified identical.
Strict checkpoint resume therefore requires the recorded revision; new source
runs use a new output directory. Published reports regenerate from the aggregate
prediction CSV plus its checksum receipt when local checkpoints are absent.
Fresh-clone training first retrieves a new raw snapshot with the download command.
