"""
quantum_reservoir_trotter.py — Trotterized Quantum Reservoir Circuit
=====================================================================

Replaces the matrix exponential evolution in quantum_reservoir_qiskit.py
with a proper Trotterized quantum circuit built from native Qiskit gates.

WHY THIS FILE EXISTS
--------------------
quantum_reservoir_qiskit.py computes e^{-iτH} via scipy.linalg.expm —
mathematically exact, but not a quantum circuit. You cannot submit a matrix
exponential to real hardware. This file decomposes that same evolution into
RXX and RZ gates using the Suzuki-Trotter product formula, producing a
circuit that can run on any gate-based quantum processor.

WHAT CHANGES vs quantum_reservoir_qiskit.py
--------------------------------------------
- Evolution: scipy.linalg.expm  →  Trotterized QuantumCircuit
- Simulation backend: DensityMatrix.evolve(Operator)
                    → DensityMatrix.evolve(QuantumCircuit)
- New parameter: n_trotter — number of Trotter steps (accuracy vs depth)
- Everything else (encoding, partial trace, ridge regression, metrics)
  is imported directly from quantum_reservoir_qiskit.py

TROTTER DECOMPOSITION
---------------------
For H = H_XX + H_Z where:
    H_XX = Σ_{i<j} J_ij X_i X_j
    H_Z  = Σ_i Z_i

First-order (Lie-Trotter):
    e^{-iτH} ≈ [ e^{-i(τ/n)H_XX} · e^{-i(τ/n)H_Z} ]^n

Each factor maps to native Qiskit gates:
    e^{-iδt J_ij X_i X_j} = RXX(2 J_ij δt)   on qubits i, j
    e^{-iδt Z_i}           = RZ(2 δt)          on qubit  i

Trotter error scales as O(τ²/n) — more steps = more accurate, deeper circuit.
For n_trotter=1 on small τ=1 the error is manageable; increase for better
accuracy at the cost of circuit depth.

QUBIT LAYOUT (same as quantum_reservoir_qiskit.py)
--------------------------------------------------
  Qubits 0..n_input-1  (0..6) : input qubits  (7 features, RY encoded)
  Qubits n_input..9    (7..9) : hidden qubits  (3 memory qubits)
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, SparsePauliOp

# Reuse everything that doesn't change from the exact version
from quantum_reservoir_qiskit import (
    MAX_RV, MIN_RV, COE, DIF,
    load_coupling_matrices,
    generate_coupling_matrix,
    build_ising_hamiltonian,       # still used for Trotter accuracy comparison
    build_z_observables,
    encode_input,                  # RY encoding is identical
    rolling_ridge_regression,
    MSE, RMSE, MAE, MAPE, compute_qlike, hitrate,
)


# ============================================================
# Trotter step — one layer of the product formula
# ============================================================

def build_trotter_step(nqubit: int, J: np.ndarray, delta_t: float) -> QuantumCircuit:
    """
    Build one first-order Trotter step as a QuantumCircuit.

    Implements:
        e^{-i delta_t H_XX} · e^{-i delta_t H_Z}

    Gate mapping:
        e^{-i delta_t J[i,j] X_i X_j} = RXX(2 * J[i,j] * delta_t)
        e^{-i delta_t Z_i}             = RZ(2 * delta_t)

    Qiskit gate conventions:
        RXX(θ) = exp(-i θ/2 · X⊗X)   →  θ = 2 * J[i,j] * delta_t
        RZ(θ)  = exp(-i θ/2 · Z)      →  θ = 2 * delta_t

    Parameters
    ----------
    nqubit  : int        — total qubits (10)
    J       : np.ndarray — (nqubit x nqubit) coupling matrix
    delta_t : float      — τ / n_trotter (time per Trotter step)

    Returns
    -------
    QuantumCircuit — one Trotter step, nqubit qubits wide
    """
    qc = QuantumCircuit(nqubit)

    # --- H_XX layer: RXX for every coupled pair ---
    for i in range(nqubit):
        for j in range(i + 1, nqubit):
            if abs(J[i, j]) > 1e-12:          # skip negligible couplings
                theta = 2.0 * J[i, j] * delta_t
                qc.rxx(theta, i, j)

    # --- H_Z layer: RZ on every qubit ---
    theta_z = 2.0 * delta_t                    # coefficient v=1 for all qubits
    for i in range(nqubit):
        qc.rz(theta_z, i)

    return qc


# ============================================================
# Full Trotterized evolution circuits
# ============================================================

def build_trotter_evolution(
    nqubit     : int,
    J          : np.ndarray,
    tau        : float,
    n_trotter  : int,
    virtual_node: int,
) -> tuple:
    """
    Build Trotterized evolution circuits for both U and dU.

    U  approximates e^{-i tau H}           (used in intermediate steps)
    dU approximates e^{-i (tau/V) H}       (used in final virtual-node step)

    Each is n_trotter repetitions of build_trotter_step().

    Parameters
    ----------
    nqubit       : int   — total qubits (10)
    J            : np.ndarray — coupling matrix
    tau          : float — total evolution time (1.0)
    n_trotter    : int   — Trotter steps (higher = more accurate, deeper circuit)
    virtual_node : int   — number of virtual nodes (1 or 2)

    Returns
    -------
    (U_circ, dU_circ) — two QuantumCircuits
    """
    delta_t  = tau / n_trotter
    delta_dt = (tau / virtual_node) / n_trotter

    # Full evolution U = (one step at delta_t) repeated n_trotter times
    U_circ = QuantumCircuit(nqubit)
    for _ in range(n_trotter):
        U_circ.compose(build_trotter_step(nqubit, J, delta_t), inplace=True)

    # Fractional evolution dU = (one step at delta_dt) repeated n_trotter times
    dU_circ = QuantumCircuit(nqubit)
    for _ in range(n_trotter):
        dU_circ.compose(build_trotter_step(nqubit, J, delta_dt), inplace=True)

    return U_circ, dU_circ


# ============================================================
# Trotter accuracy check — compare against exact matrix exponential
# ============================================================

def trotter_accuracy(
    nqubit    : int,
    J         : np.ndarray,
    tau       : float,
    n_trotter : int,
) -> dict:
    """
    Compare the Trotterized unitary against the exact matrix exponential.

    Computes the operator norm ||U_exact - U_trotter|| and the gate count
    so you can tune n_trotter before running the full simulation.

    Parameters
    ----------
    nqubit    : int        — total qubits (10)
    J         : np.ndarray — coupling matrix
    tau       : float      — evolution time
    n_trotter : int        — number of Trotter steps to test

    Returns
    -------
    dict with keys:
        n_trotter        : int
        gate_count_rxx   : int   — number of RXX gates in full circuit
        gate_count_rz    : int   — number of RZ  gates in full circuit
        circuit_depth    : int   — depth of the Trotterized circuit
        operator_error   : float — ||U_exact - U_trotter||_2
    """
    # Exact unitary
    H_matrix = build_ising_hamiltonian(nqubit, J)
    U_exact  = expm(-1j * tau * H_matrix)

    # Trotterized unitary — convert circuit to Operator matrix
    U_circ, _ = build_trotter_evolution(nqubit, J, tau, n_trotter, virtual_node=1)
    U_trotter  = Operator(U_circ).data

    # Operator error (spectral norm via largest singular value)
    diff   = U_exact - U_trotter
    error  = np.linalg.norm(diff, ord=2)

    # Gate counts from the circuit
    ops         = U_circ.count_ops()
    gate_rxx    = ops.get('rxx', 0)
    gate_rz     = ops.get('rz',  0)

    return {
        'n_trotter'      : n_trotter,
        'gate_count_rxx' : gate_rxx,
        'gate_count_rz'  : gate_rz,
        'circuit_depth'  : U_circ.depth(),
        'operator_error' : error,
    }


# ============================================================
# Main Trotterized reservoir simulation
# ============================================================

def quantum_reservoir_trotter(
    data         : pd.DataFrame,
    features     : list,
    J            : np.ndarray,
    tau          : float,
    K_delay      : int,
    virtual_node : int,
    nqubit       : int,
    n_trotter    : int = 1,
) -> np.ndarray:
    """
    Run the quantum reservoir simulation using Trotterized circuits.

    Drop-in replacement for quantum_reservoir() in quantum_reservoir_qiskit.py.
    Identical protocol — only the evolution step changes:
        quantum_reservoir_qiskit.py : DensityMatrix.evolve(Operator(U_matrix))
        this file                   : DensityMatrix.evolve(U_circuit)

    The circuit U_circuit is built from RXX and RZ gates and can be compiled
    for real quantum hardware. The simulation still uses Qiskit's DensityMatrix
    backend for exact statevector-level results.

    Parameters
    ----------
    data         : pd.DataFrame — full dataset (816 rows)
    features     : list of str  — input feature column names (7 features)
    J            : np.ndarray   — (nqubit x nqubit) Ising coupling matrix
    tau          : float        — total evolution time (1.0)
    K_delay      : int          — memory depth (3)
    virtual_node : int          — virtual nodes (1=QR1, 2=QR2)
    nqubit       : int          — total qubits (10)
    n_trotter    : int          — Trotter steps (default 4, increase for accuracy)

    Returns
    -------
    np.ndarray — shape (nqubit * virtual_node, n_samples)
    """
    n_input   = len(features)
    n_hidden  = nqubit - n_input
    N         = nqubit
    n_samples = len(data)

    # Build Trotterized circuits and convert to Operators once —
    # Operator(circuit) computes the unitary matrix upfront so each
    # evolve() call is a single matrix multiply, matching the speed
    # of the exact version in quantum_reservoir_qiskit.py
    U_circ, dU_circ = build_trotter_evolution(nqubit, J, tau, n_trotter, virtual_node)
    U_op  = Operator(U_circ)
    dU_op = Operator(dU_circ)

    observables = build_z_observables(nqubit)
    input_qargs = list(range(n_input))

    output = np.zeros((N * virtual_node, n_samples))

    for l in range(K_delay, n_samples):
        if l % 200 == 0:
            print(f"    time step {l} / {n_samples}")

        rho_hidden = DensityMatrix.from_label('0' * n_hidden)

        for k in range(K_delay, 0, -1):

            input_vals = np.array([data.iloc[l - k][feat] for feat in features])
            rho_input  = encode_input(input_vals, n_input)

            # Form joint state: hidden (higher qubits) ⊗ input (lower qubits)
            rho_joint = rho_hidden.tensor(rho_input)

            if k != 1:
                # Intermediate step — evolve with Trotterized U, partial trace
                rho_joint  = rho_joint.evolve(U_op)
                rho_hidden = partial_trace(rho_joint, input_qargs)

            else:
                # Final step — evolve with Trotterized dU, measure Z on all qubits
                it = 0
                for _ in range(virtual_node):
                    rho_joint = rho_joint.evolve(dU_op)
                    for obs in observables:
                        output[it, l] = rho_joint.expectation_value(obs).real
                        it += 1

    return output


# ============================================================
# Convenience runner — mirrors run_qrc_simulation.py structure
# ============================================================

def run_trotter_simulation(
    data_path  : str = "data/Data.CSV",
    jld2_path  : str = "data/coeff_10.jld2",
    n_trotter  : int = 4,
    save_csv   : bool = True,
) -> dict:
    """
    End-to-end Trotterized QRC simulation.

    Runs QR1 and QR2 with the Trotterized circuit, prints metrics,
    and optionally saves predictions to qrc_trotter_result.csv.

    Parameters
    ----------
    data_path : str  — path to Data.CSV
    jld2_path : str  — path to coeff_10.jld2
    n_trotter : int  — Trotter steps (4 is a good starting point)
    save_csv  : bool — whether to save predictions to CSV

    Returns
    -------
    dict with keys: Pre1, Pre2, signal1, signal2
    """
    NQUBIT = 10
    TAU    = 1.0
    K      = 3
    TOTAL  = 816
    L_OOS  = 245
    WS     = 0
    LAMBDA = 1e-8

    FEATURES_QR1 = ["RV", "MKT", "DP",  "IP",   "RV_q", "STR", "DEF"]
    FEATURES_QR2 = ["RV", "MKT", "STR", "RV_q", "EP",   "INF", "DEF"]

    os.makedirs("results/predictions", exist_ok=True)

    # --- Load data ---
    print("Loading Data.CSV ...")
    data        = pd.read_csv(data_path, header=0, index_col=0)
    target_full = np.array(data["RV"], dtype=np.float64)
    y           = target_full[TOTAL - L_OOS : TOTAL]

    # --- Load coupling matrices ---
    print("Loading coupling matrices ...")
    ms = load_coupling_matrices(jld2_path)
    if ms is None:
        ms = np.stack([generate_coupling_matrix(NQUBIT, seed=i) for i in range(100)])
    J_qr1, J_qr2 = ms[0], ms[1]

    # --- Trotter accuracy check before running ---
    print(f"\nTrotter accuracy check (n_trotter={n_trotter}):")
    for label, J in [("QR1", J_qr1), ("QR2", J_qr2)]:
        acc = trotter_accuracy(NQUBIT, J, TAU, n_trotter)
        print(f"  {label}: depth={acc['circuit_depth']}  "
              f"RXX gates={acc['gate_count_rxx']}  "
              f"RZ gates={acc['gate_count_rz']}  "
              f"operator error={acc['operator_error']:.6f}")

    # --- QR1 ---
    print(f"\nRunning QR1 (Trotterized, n_trotter={n_trotter}) ...")
    signal1 = quantum_reservoir_trotter(
        data, FEATURES_QR1, J_qr1, TAU, K, 1, NQUBIT, n_trotter
    )
    Pre1 = rolling_ridge_regression(signal1, target_full, TOTAL, L_OOS, WS, LAMBDA)

    # --- QR2 ---
    print(f"\nRunning QR2 (Trotterized, n_trotter={n_trotter}) ...")
    signal2 = quantum_reservoir_trotter(
        data, FEATURES_QR2, J_qr2, TAU, K, 2, NQUBIT, n_trotter
    )
    Pre2 = rolling_ridge_regression(signal2, target_full, TOTAL, L_OOS, WS, LAMBDA)

    # --- Metrics ---
    print("\n--- QR1 (Trotterized) ---")
    print(f"  Hit rate : {hitrate(Pre1, y):.4f}   (exact reference: 0.4408)")
    print(f"  MSE      : {MSE(Pre1,     y):.4f}   (exact reference: 0.1051)")
    print(f"  RMSE     : {RMSE(Pre1,    y):.4f}   (exact reference: 0.3242)")
    print(f"  MAE      : {MAE(Pre1,     y):.4f}   (exact reference: 0.2488)")
    print(f"  MAPE     : {MAPE(Pre1,    y):.4f}   (exact reference: 8.2824)")
    print(f"  QLIKE    : {compute_qlike(Pre1, y):.4f}   (exact reference: 1.4428)")

    print("\n--- QR2 (Trotterized) ---")
    print(f"  Hit rate : {hitrate(Pre2, y):.4f}   (exact reference: 0.4571)")
    print(f"  MSE      : {MSE(Pre2,     y):.4f}   (exact reference: 0.1038)")
    print(f"  RMSE     : {RMSE(Pre2,    y):.4f}   (exact reference: 0.3221)")
    print(f"  MAE      : {MAE(Pre2,     y):.4f}   (exact reference: 0.2426)")
    print(f"  MAPE     : {MAPE(Pre2,    y):.4f}   (exact reference: 8.1731)")
    print(f"  QLIKE    : {compute_qlike(Pre2, y):.4f}   (exact reference: 1.4004)")

    # --- Save ---
    if save_csv:
        Pre1_denorm = (Pre1 + 1) * DIF + MIN_RV
        Pre2_denorm = (Pre2 + 1) * DIF + MIN_RV
        out = pd.DataFrame({"QR2": Pre2_denorm, "QR1": Pre1_denorm})
        out.to_csv("results/predictions/qrc_trotter_result.csv", index=False)
        print("\nSaved: results/predictions/qrc_trotter_result.csv")

        # Compare against exact matrix exponential output if available
        try:
            ref = pd.read_csv("results/predictions/qrc_predict_result.csv")
            d1  = np.abs(out["QR1"].values - ref["QR1"].values)
            d2  = np.abs(out["QR2"].values - ref["QR2"].values)
            print("\nDifference vs exact matrix exponential (qrc_predict_result.csv):")
            print(f"  QR1 — max: {d1.max():.6f}   mean: {d1.mean():.6f}")
            print(f"  QR2 — max: {d2.max():.6f}   mean: {d2.mean():.6f}")
        except FileNotFoundError:
            print("  qrc_predict_result.csv not found — run run_qrc_simulation.py first.")

    return {"Pre1": Pre1, "Pre2": Pre2, "signal1": signal1, "signal2": signal2}


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_trotter_simulation(n_trotter=2)
