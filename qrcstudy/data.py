"""Immutable downloads, validated monthly targets, and causal feature construction."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FEATURES = ["log_rv", "log_rv_3", "log_rv_12", "return", "downside_share", "max_abs_return", "close_range"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def validate_prices(prices, expected=None):
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have unique increasing session dates")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError("Missing, nonfinite, or nonpositive close")
    if expected is not None:
        missing = expected.difference(prices.index)
        extra = prices.index.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(f"Session mismatch: missing={missing.tolist()}, unexpected={extra.tolist()}")


def monthly_features(prices, as_of):
    """Completed calendar months only. Caller validates complete daily sessions."""
    validate_prices(prices)
    cutoff = pd.Timestamp(as_of).normalize()
    prices = prices.loc[:cutoff]
    r = np.log(prices).diff()
    sums = r.pow(2).resample("ME").sum(min_count=1)
    # Remove the first return-incomplete month before computing rolling means.
    sums = sums.loc[sums.index > prices.index[0].to_period("M").to_timestamp("M")]
    f = pd.DataFrame({"log_rv": np.log(np.sqrt(sums))})
    f["log_rv_3"] = f.log_rv.rolling(3).mean()
    f["log_rv_12"] = f.log_rv.rolling(12).mean()
    f["return"] = r.resample("ME").sum(min_count=1)
    f["downside_share"] = r.where(r < 0, 0).pow(2).resample("ME").sum() / sums
    f["max_abs_return"] = r.abs().resample("ME").max()
    f["close_range"] = np.log(prices.resample("ME").max() / prices.resample("ME").min())
    # The first price month may start mid-month and lacks the preceding close.
    f = f.loc[f.index < cutoff.to_period("M").to_timestamp()]
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def fit_scaler(frame, cutoff="2017-12-31"):
    train = frame.loc[:cutoff, FEATURES]
    if train.empty:
        raise ValueError("No pre-evaluation calibration data")
    lo, span = train.min(), train.max() - train.min()
    if (span <= 0).any():
        raise ValueError("Constant calibration feature")
    return {"cutoff": cutoff, "min": lo.to_dict(), "span": span.to_dict()}


def transform(frame, scaler):
    z = (frame[FEATURES] - pd.Series(scaler["min"])) / pd.Series(scaler["span"]) - 1
    clipped = ((z < -1) | (z > 0))
    # Target is scaled separately and is never clipped.
    y = (frame.log_rv - scaler["min"]["log_rv"]) / scaler["span"]["log_rv"] - 1
    return z.clip(-1, 0), y, clipped


def inverse_target(y, scaler):
    return (np.asarray(y) + 1) * scaler["span"]["log_rv"] + scaler["min"]["log_rv"]


def download(output, as_of, legacy):
    import exchange_calendars as xcals

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Snapshot exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    end = pd.Timestamp(as_of).normalize()
    start = pd.Timestamp("1949-12-01", tz="UTC")
    end_seconds = int((end.tz_localize("UTC") + pd.Timedelta(days=1)).timestamp())
    urls = {
        "yahoo.json": f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={int(start.timestamp())}&period2={end_seconds}&interval=1d",
        "fred.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
    }
    sources = {}
    for filename, url in urls.items():
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            payload = response.read()
        (output / filename).write_bytes(payload)
        sources[filename] = {"url": url, "sha256": digest(output / filename), "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat()}
    write_json(output / "sources.json", sources)
    prepare_snapshot(output, as_of, legacy, sources)


def prepare_snapshot(output, as_of, legacy, sources=None):
    """Validate an existing raw download without fetching or replacing it."""
    import exchange_calendars as xcals
    output = Path(output)
    end = pd.Timestamp(as_of).normalize()
    if (output / "manifest.json").exists():
        raise FileExistsError("Validated snapshot already exists")
    sources = sources or json.loads((output / "sources.json").read_text())
    for name, meta in sources.items():
        if digest(output / name) != meta["sha256"]:
            raise ValueError("Raw payload checksum mismatch")
    result = json.loads((output / "yahoo.json").read_text())["chart"]["result"][0]
    if result["meta"].get("dataGranularity") != "1d":
        raise ValueError("Provider returned non-daily prices")
    index = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    prices = pd.Series(result["indicators"]["quote"][0]["close"], index=index, name="close").loc[:end]
    validate_prices(prices)
    cal = xcals.get_calendar("XNYS", start=prices.index[0], end=end)
    expected = cal.sessions_in_range(prices.index[0], end)
    if expected.tz is not None:
        expected = expected.tz_localize(None)
    # pandas' default holiday horizon starts in 1970. Explicitly subtract the
    # exchange's historical rules, which otherwise disappear from its offset.
    expected = expected.difference(cal.regular_holidays.holidays(prices.index[0], end))
    # Respect actual exchange close for today's partially completed session.
    now = pd.Timestamp.now(tz="UTC")
    expected = pd.DatetimeIndex([d for d in expected if cal.session_close(d) <= now])
    prices = prices.loc[:expected[-1]]
    validate_prices(prices, expected)
    fred = pd.read_csv(output / "fred.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fred = pd.to_numeric(fred, errors="coerce").dropna()
    overlap = pd.concat([prices, fred.rename("fred")], axis=1).dropna()
    overlap["absolute_difference"] = (overlap.close - overlap.fred).abs()
    overlap.to_csv(output / "crosscheck.csv", index_label="date")
    if len(overlap) == 0:
        raise ValueError("FRED cross-check has no overlapping sessions")
    # FRED identifies S&P Dow Jones Indices as the source. Prefer that feed
    # over Yahoo on discrepancies larger than one-cent/float32 rounding.
    corrections = overlap.loc[overlap.absolute_difference > .011].copy()
    corrections["resolution"] = "use FRED index-provider close; original Yahoo payload preserved"
    corrections.to_csv(output / "price_corrections.csv", index_label="date")
    prices.loc[corrections.index] = corrections.fred
    prices.to_csv(output / "daily.csv", index_label="date")
    frame = monthly_features(prices, end)
    frame.to_csv(output / "monthly.csv", index_label="date")
    from quantum_reservoir_qiskit import MIN_RV, DIF
    old = pd.read_csv(legacy, index_col=0, parse_dates=True).RV
    old = (old + 1) * DIF + MIN_RV
    # Audit every historical target, including warmup months omitted from features.
    rv = np.log(np.sqrt(np.log(prices).diff().pow(2).resample("ME").sum(min_count=1)))
    audit = pd.concat([old.rename("legacy_log_rv"), rv.rename("rebuilt_log_rv")], axis=1).dropna()
    audit["difference"] = audit.rebuilt_log_rv - audit.legacy_log_rv
    audit.to_csv(output / "historical_reconciliation.csv", index_label="date")
    metadata = {"as_of": str(end.date()), "first_session": str(prices.index[0].date()), "last_session": str(prices.index[-1].date()), "last_complete_month": str(frame.index[-1].date()), "sources": sources, "rows_daily": len(prices), "rows_monthly": len(frame), "calendar": "XNYS with explicit pre-1970 regular holidays", "features": FEATURES, "historical_max_abs_log_difference": audit.difference.abs().max(), "historical_differences_above_1e-8": int((audit.difference.abs() > 1e-8).sum()), "fred_overlap_rows": len(overlap), "fred_max_price_difference": overlap.absolute_difference.max(), "corrected_closes": len(corrections), "price_policy": "FRED preferred when overlapping close differs by more than 0.011 index points", "artifacts": {name: digest(output / name) for name in ["daily.csv", "monthly.csv", "crosscheck.csv", "historical_reconciliation.csv", "price_corrections.csv"]}}
    write_json(output / "manifest.json", metadata)
    print(json.dumps(metadata, indent=2))
