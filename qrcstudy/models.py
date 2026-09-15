"""Existing model families with explicit, shared forecast-date contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ["Persistence", "HAR", "HARX", "AR1", "AR3", "ARMAX", "CRL", "CRLX", "QR1", "QR2", "LSTM", "LSTMX"]
STOCHASTIC = {"CRL", "CRLX", "QR1", "QR2", "LSTM", "LSTMX"}


def sequences(x, k=3):
    """Row t holds only x[t-k:t]; includes a final unobserved target row."""
    out = np.full((len(x) + 1, k, x.shape[1]), np.nan)
    for t in range(k, len(out)):
        out[t] = x[t-k:t]
    return out


def reservoir_features(x, model, seed, cache):
    from pathlib import Path
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / f"{model}-{seed}.npy"
    if file.exists():
        from .data import digest
        import json
        metadata = json.loads(file.with_suffix(".json").read_text())
        if digest(file) != metadata["sha256"]:
            raise ValueError("Cached reservoir checksum mismatch")
        a = np.load(file)
        expected_dim = {"QR1": 10, "QR2": 20, "CRL": 50, "CRLX": 20}[model]
        if a.shape != (len(x)+1, expected_dim) or not np.isfinite(a[3:]).all():
            raise ValueError("Invalid cached reservoir features")
        return a
    if model.startswith("QR"):
        from quantum_reservoir_qiskit import generate_coupling_matrix, build_ising_hamiltonian, compute_unitaries, quantum_reservoir
        v = 1 if model == "QR1" else 2
        unitary_file = cache / f"unitary-{seed}-{v}.npz"
        if unitary_file.exists():
            with np.load(unitary_file) as z:
                u, du = z["u"], z["du"]
        else:
            j = generate_coupling_matrix(10, seed=seed)
            h = build_ising_hamiltonian(10, j)
            u, du = compute_unitaries(h, 1., v)
            np.savez(unitary_file, u=u, du=du)
        data = pd.DataFrame(np.vstack([x, np.zeros((1, 7))]))
        a = quantum_reservoir(data, list(range(7)), u, du, 3, v, 10).T
    else:
        from reservoirpy.nodes import Reservoir
        values = x[:, :1] if model == "CRL" else x
        seq = sequences(values)
        n = 50 if model == "CRL" else 20
        r = Reservoir(n, input_dim=values.shape[1], lr=.6, sr=.9, input_scaling=.1, seed=seed)
        r.run(np.zeros((3, values.shape[1])))
        a = np.zeros((len(x)+1, n))
        for t in range(3, len(a)):
            r.reset()
            a[t] = r.run(seq[t])[-1]
    from .data import digest, write_json
    np.save(file, a)
    write_json(file.with_suffix(".json"), {"sha256": digest(file), "shape": list(a.shape)})
    return a


def lstm_forecast(seq, y, start, end, model_name, seed, threads=2):
    import torch
    from torch import nn
    torch.set_num_threads(threads)
    # Seed depends on forecast origin, not task order or resume position.
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    class LSTM(nn.Module):
        def __init__(self, inputs, hidden):
            super().__init__()
            self.lstm = nn.LSTM(inputs, hidden, 2, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    n = 1 if model_name == "LSTM" else 7
    model = LSTM(n, 60 if n == 1 else 50)
    x = torch.tensor(seq[start:end, :, :n], dtype=torch.float32)
    target = torch.tensor(y[start:end, None], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, target), batch_size=64, shuffle=True)
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(seq[end:end+1, :, :n], dtype=torch.float32)).item()


def forecast(model, x, y, seq, features, end, window, seed, threads=2):
    start = end-window
    if start < 3 or not np.isfinite(y[start:end]).all():
        raise ValueError("Insufficient valid training targets")
    if model == "Persistence":
        return y[end-1]
    if model in {"LSTM", "LSTMX"}:
        return lstm_forecast(seq, y, start, end, model, seed, threads)
    if model.startswith("QR"):
        train = features[start:end]
        w = np.linalg.solve(train.T @ train + 1e-8*np.eye(train.shape[1]), train.T @ y[start:end])
        return float(features[end] @ w)
    if model in {"CRL", "CRLX"}:
        from reservoirpy.nodes import Ridge
        readout = Ridge(ridge=1e-7).fit(features[start:end], y[start:end, None])
        return float(readout.run(features[end:end+1]).flat[0])
    if model == "ARMAX":
        from statsmodels.tsa.arima.model import ARIMA
        # Preserve ARIMA(3,0,0), no trend, using the seven lagged inputs.
        fit = ARIMA(y[start:end], exog=x[start-1:end-1], order=(3,0,0), trend="n").fit(method_kwargs={"maxiter": 1000})
        if not fit.mle_retvals.get("converged", True):
            raise RuntimeError("ARMAX optimizer did not converge")
        return float(np.asarray(fit.forecast(exog=x[end-1:end]))[0])
    if model in {"AR1", "AR3"}:
        p = 1 if model == "AR1" else 3
        a = np.array([[y[t-lag] for lag in range(1, p+1)] for t in range(start, end+1)])
    else:
        a = x[start-1:end, :3] if model == "HAR" else x[start-1:end]
    a = np.column_stack([np.ones(len(a)), a])
    w = np.linalg.lstsq(a[:-1], y[start:end], rcond=None)[0]
    return float(a[-1] @ w)
