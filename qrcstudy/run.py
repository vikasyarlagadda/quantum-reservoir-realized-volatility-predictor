"""Frozen identities, per-origin checkpoints, and resumable local execution."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd

from .data import FEATURES, digest, fit_scaler, inverse_target, transform, write_json
from .models import MODELS, STOCHASTIC, forecast, reservoir_features, sequences


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def checked_manifest(folder, config):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity"] != identity(config):
            raise ValueError("Run identity mismatch; use a new output directory")
    else:
        write_json(path, {"identity": identity(config), "config": config, "created_at": pd.Timestamp.now(tz="UTC").isoformat()})
    return identity(config)


def worker(task):
    out, config, model, seed, limit = task
    out = Path(out)
    run_id = identity(config)
    frame = pd.read_csv(config["data"], index_col=0, parse_dates=True)
    frame = frame.loc[:config["end"]]
    z, y, _ = transform(frame, config["scaler"])
    x = z.to_numpy()
    dates = frame.index.append(pd.DatetimeIndex([frame.index[-1] + pd.offsets.MonthEnd()]))
    y = np.append(y.to_numpy(), np.nan)
    seq = sequences(x)
    positions = [t for t, date in enumerate(dates) if date >= pd.Timestamp(config["start"])]
    features = None
    if model in {"QR1", "QR2", "CRL", "CRLX"}:
        features = reservoir_features(x, model, seed, out / "cache")
    for window in config["windows"]:
        folder = out / "checkpoints" / f"{model}-w{window}-s{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        completed = 0
        for t in positions:
            name = folder / f"{dates[t]:%Y-%m}.json"
            if name.exists():
                row = json.loads(name.read_text())
                if row["run_id"] != run_id or row["target_month"] != str(dates[t].date()):
                    raise ValueError("Checkpoint identity/date mismatch")
                continue
            if limit is not None and completed >= limit:
                break
            started = time.perf_counter()
            row = {"run_id": run_id, "configuration": "modern", "model": model, "seed": seed, "window": window, "target_month": str(dates[t].date()), "forecast_origin": str(dates[t-1].date()), "training_start": str(dates[t-window].date()), "training_end": str(dates[t-1].date()), "actual_log_rv": float(frame.log_rv.iloc[t]) if t < len(frame) else None, "previous_log_rv": float(frame.log_rv.iloc[t-1]), "predicted_log_rv": None, "status": "failed"}
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    per_origin_seed = int(np.random.SeedSequence([seed, dates[t].year, dates[t].month]).generate_state(1)[0])
                    pred = forecast(model, x, y, seq, features, t, window, per_origin_seed, config["threads"])
                pred = float(inverse_target(pred, config["scaler"]))
                if not np.isfinite(pred) or not np.isfinite(np.exp(2*pred)):
                    raise FloatingPointError("Nonfinite forecast or variance")
                row.update(predicted_log_rv=pred, status="ok", warnings=sorted(set(str(w.message) for w in caught)))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter()-started
            write_json(name, row)
            completed += 1
            if completed % 12 == 0:
                print(f"{model} seed={seed} window={window} through {dates[t]:%Y-%m}", flush=True)
    return f"{model} seed={seed} complete"


def collect(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = []
    for path in sorted((folder / "checkpoints").glob("*/*.json")):
        row = json.loads(path.read_text())
        if row["run_id"] != manifest["identity"]:
            raise ValueError(f"Wrong run in {path}")
        rows.append(row)
    if not rows:
        raise ValueError("No forecasts")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["model", "seed", "window", "target_month"]).any():
        raise ValueError("Duplicate forecasts")
    frame.to_csv(folder / "predictions.csv", index=False)
    return frame


def run_modern(data, output, start, end, windows, seeds, models, workers, threads, limit=None):
    data = Path(data).resolve()
    snapshot = json.loads((data.parent / "manifest.json").read_text())
    for file, hash_ in snapshot["artifacts"].items():
        if digest(data.parent / file) != hash_:
            raise ValueError(f"Snapshot artifact changed: {file}")
    frame = pd.read_csv(data, index_col=0, parse_dates=True)
    if pd.Timestamp(end) > frame.index[-1]:
        raise ValueError("Requested evaluation extends beyond completed data")
    scaler = fit_scaler(frame)
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "torch", "statsmodels", "reservoirpy", "qiskit", "arch"]}
    root = Path(__file__).resolve().parents[1]
    config = {"protocol": "modern-v1", "data": str(data), "data_sha256": digest(data), "snapshot_sha256": digest(data.parent / "manifest.json"), "start": start, "end": end, "windows": windows, "seeds": seeds, "models": models, "threads": threads, "scaler": scaler, "features": FEATURES, "epochs": 100, "versions": versions, "python": platform.python_version(), "source_hashes": {p: digest(root / p) for p in ["qrcstudy/data.py", "qrcstudy/models.py", "qrcstudy/run.py", "quantum_reservoir_qiskit.py"]}}
    checked_manifest(output, config)
    output = Path(output)
    _, _, clipped = transform(frame, scaler)
    clipped.loc[start:end].mean().rename("clipped_fraction").to_csv(output / "clipping.csv")
    write_json(output / "scaler.json", scaler)
    tasks = [(str(output), config, model, seed, limit) for model in models for seed in (seeds if model in STOCHASTIC else [0])]
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = str(threads)
    start_time = time.perf_counter()
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(worker, task): (task[2], task[3]) for task in tasks}
        for future in as_completed(pending):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                model, seed = pending[future]
                failures.append({"model": model, "seed": seed, "error": repr(exc)})
                print(f"TASK FAILURE {model} {seed}: {exc}", flush=True)
    write_json(output / "execution.json", {"wall_seconds_this_invocation": time.perf_counter()-start_time, "task_failures": failures, "pilot_limit": limit})
    result = collect(output)
    print(result.groupby(["model", "status"]).size().to_string())
