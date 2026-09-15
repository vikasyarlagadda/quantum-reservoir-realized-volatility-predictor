"""
quantum_reservoir_qiskit.py — Quantum Reservoir Computing in Qiskit
=====================================================================

Direct translation of Time_series.jl into Python/Qiskit.

Paper: "Quantum Reservoir Computing for Realized Volatility Forecasting"
       arXiv:2505.13933, Physical Review Research 8, 023028 (2026)

Every function maps 1-to-1 with its Julia counterpart. Comments call out
the exact Julia line or function name being replicated.

Qiskit qubit-ordering convention (important for correctness):
  - Qubit 0 = rightmost character in a Pauli string = least-significant bit
  - DensityMatrix.tensor(other): self → higher qubit indices,
                                  other → lower qubit indices
  - partial_trace(state, qargs): traces OUT the listed qubit indices

Qubit layout in the joint state (10 qubits total):
  - Qubits 0 .. n_input-1  (0..6)  : input qubits   (7 features)
  - Qubits n_input .. 9    (7..9)  : hidden qubits   (3 memory qubits)
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, SparsePauliOp

# ============================================================
# Normalization constants
# Exact copy of Time_series.jl lines 108-111
# ============================================================
MAX_RV = -1.2543188032019446
MIN_RV = -4.7722718186046515
COE    = (MAX_RV - MIN_RV) ** 2   # MSE scale factor  (~12.376)
DIF    = MAX_RV - MIN_RV           # MAE scale factor  (~-3.518)


# ============================================================
# Coupling matrix utilities
# Matches coeff_matrix() in Time_series.jl and
#   file = JLD2.load("coeff_10.jld2"); ms = file["ms"]
# ============================================================

def load_coupling_matrices(jld2_path: str) -> np.ndarray:
    """
    Load the 100 pre-generated Ising coupling matrices from coeff_10.jld2.

    JLD2 is HDF5-compatible and readable with h5py.
    Returns array of shape (100, 10, 10).
    Returns None if the file cannot be parsed, triggering the fallback.
    """
    try:
        import h5py
        with h5py.File(jld2_path, 'r') as f:
            ms = f['ms'][:]
        # Julia arrays are column-major; HDF5 transposes on write so h5py
        # reads them back in C order.  For a symmetric (10,10) matrix the
        # transpose is identical, so no further fix-up is required.
        # Julia is column-major; HDF5 reverses axis order on write.
        # h5py reads (100,10,10) Julia array as (10,10,100) — transpose back.
        if ms.shape[0] != ms.shape[2]:
            ms = ms.transpose(2, 0, 1)
        return ms.astype(np.float64)
    except Exception as exc:
        print(f"  Warning: could not read {jld2_path} ({exc}).")
        return None


def generate_coupling_matrix(nqubit: int, J_max: float = 1.0,
                             seed: int = None) -> np.ndarray:
    """
    Generate one random symmetric coupling matrix with largest eigenvalue = J_max.

    Matches coeff_matrix(N, J) in Time_series.jl:
        m = rand(N,N)
        m = (m + transpose(m)) / 2
        m[i,i] = 0
        m / max(eigvals(m)...) * J
    """
    rng = np.random.default_rng(seed)
    m = rng.random((nqubit, nqubit))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    eigs = np.linalg.eigvalsh(m)
    return m / np.max(eigs) * J_max


# ============================================================
# Hamiltonian construction
# Matches Qreservoir(nqubit, ps) in Time_series.jl
# ============================================================

def build_ising_hamiltonian(nqubit: int, J: np.ndarray) -> np.ndarray:
    """
    Build the transverse-field Ising Hamiltonian matrix (paper Eq. 1):

        H = sum_{i<j} J[i,j] * X_i X_j  +  sum_i Z_i

    Built via Qiskit's SparsePauliOp to guarantee that the resulting matrix
    uses Qiskit's qubit ordering (qubit 0 = rightmost Pauli character),
    so it is consistent with DensityMatrix.evolve() and expectation_value().

    Parameters
    ----------
    nqubit : int        — total qubits (10)
    J      : np.ndarray — (nqubit x nqubit) symmetric coupling matrix, J[i,i]=0

    Returns
    -------
    np.ndarray — (2^nqubit x 2^nqubit) complex128 Hamiltonian matrix
    """
    pauli_strs = []
    coeffs     = []

    # XX coupling: sum_{i<j} J[i,j] * X_i X_j
    # Qiskit Pauli string: rightmost character = qubit 0
    # So qubit i → position (nqubit-1-i) from the left in the string
    for i in range(nqubit):
        for j in range(i + 1, nqubit):
            chars = ['I'] * nqubit
            chars[nqubit - 1 - i] = 'X'
            chars[nqubit - 1 - j] = 'X'
            pauli_strs.append(''.join(chars))
            coeffs.append(J[i, j])

    # Transverse field: sum_i Z_i  (v = 1, the energy unit in the paper)
    for i in range(nqubit):
        chars = ['I'] * nqubit
        chars[nqubit - 1 - i] = 'Z'
        pauli_strs.append(''.join(chars))
        coeffs.append(1.0)

    H_sparse = SparsePauliOp(pauli_strs, coeffs=np.array(coeffs, dtype=complex))
    return H_sparse.to_matrix()


# ============================================================
# Unitary evolution operators
# Matches the precomputed δU and U in Quantum_Reservoir():
#   U  = CuArray(exp(-im*τ*Matrix(matrix(QR))))
#   δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
# ============================================================

def compute_unitaries(H: np.ndarray, tau: float,
                      virtual_node: int) -> tuple:
    """
    Precompute the two unitary evolution operators via matrix exponential.

        U  = e^{-i * tau * H}           full evolution (intermediate steps)
        dU = e^{-i * (tau/V) * H}       fractional evolution (final step)

    Both are computed ONCE per reservoir and reused for all 816 time steps,
    exactly as in the Julia GPU code.

    Returns
    -------
    (U, dU) — two (2^nqubit x 2^nqubit) complex128 numpy arrays
    """
    U  = expm(-1j * tau * H)
    dU = expm(-1j * (tau / virtual_node) * H)
    return U, dU


# ============================================================
# Z observables
# Matches B=[QubitsTerm(i=>Z) for i in 1:nqubit] in the Julia notebook
# ============================================================

def build_z_observables(nqubit: int) -> list:
    """
    Build a Pauli-Z observable for every qubit in the joint state.

    Returns a list of nqubit SparsePauliOp objects.
    observable[i] = Z on qubit i, I on all others.
    """
    observables = []
    for i in range(nqubit):
        chars = ['I'] * nqubit
        chars[nqubit - 1 - i] = 'Z'
        observables.append(SparsePauliOp(''.join(chars)))
    return observables


# ============================================================
# Input encoding
# Matches:
#   cir = QCircuit()
#   push!(cir, RyGate(i, rand(), isparas=true))
#   cir(para .* pi)
#   ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize, 1)
#   cir(ρ₁)           ← applies encoding to |0...0>
# ============================================================

def encode_input(feature_values: np.ndarray, n_input: int) -> DensityMatrix:
    """
    Encode feature values into n_input qubits via RY rotation gates,
    starting from the |0...0> state.

    RY(pi * x_i) on qubit i, for each input feature x_i.
    Features are in [-1, 0], so RY angles are in [-pi, 0].

    Returns
    -------
    DensityMatrix — n_input-qubit pure state after RY encoding
    """
    qc = QuantumCircuit(n_input)
    for i, val in enumerate(feature_values):
        qc.ry(float(val) * np.pi, i)
    # DensityMatrix(qc) automatically applies the circuit to |0...0>
    return DensityMatrix(qc)


# ============================================================
# Core quantum reservoir simulation
# Matches Quantum_Reservoir() in Time_series.jl exactly
# ============================================================

def quantum_reservoir(
    data          : pd.DataFrame,
    features      : list,
    U             : np.ndarray,
    dU            : np.ndarray,
    K_delay       : int,
    virtual_node  : int,
    nqubit        : int,
) -> np.ndarray:
    """
    Run the quantum reservoir over all data points.

    Direct translation of Quantum_Reservoir() in Time_series.jl.

    Protocol for each time step l (from K_delay to n_samples-1):

      Initialize: rho_hidden = |0...0><0...0|  (n_hidden qubits)

      For k = K_delay, K_delay-1, ..., 2  [intermediate steps]:
        1. Encode data[l-k] → rho_input  (RY gates on n_input qubits)
        2. Form joint state: rho_hidden.tensor(rho_input)
              hidden on qubits (n_input..nqubit-1), input on (0..n_input-1)
        3. Evolve: rho_joint = U @ rho_joint @ U†
        4. Partial trace out input qubits (0..n_input-1)
              → updated rho_hidden  (n_hidden qubits)

      For k = 1  [final step]:
        1. Encode data[l-1] → rho_input
        2. Form joint state
        3. For each virtual node v in 0..V-1:
              a. Evolve: rho_joint = dU @ rho_joint @ dU†
              b. For each qubit i: output[v*N + i, l] = <Z_i>

    Parameters
    ----------
    data         — full DataFrame (816 rows), must contain all columns in features
    features     — list of n_input column names (7 features)
    U            — full evolution unitary,      shape (2^nqubit, 2^nqubit)
    dU           — fractional evolution unitary, shape (2^nqubit, 2^nqubit)
    K_delay      — memory depth (3)
    virtual_node — number of virtual nodes (1 for QR1, 2 for QR2)
    nqubit       — total qubits (10)

    Returns
    -------
    np.ndarray — shape (nqubit * virtual_node, n_samples)
                 Columns 0..K_delay-1 are zeros (no valid input history).
    """
    n_input  = len(features)
    n_hidden = nqubit - n_input
    N        = nqubit                    # one Z observable per qubit
    n_samples = len(data)

    U_op    = Operator(U)
    dU_op   = Operator(dU)
    observables  = build_z_observables(nqubit)
    input_qargs  = list(range(n_input))  # qubit indices to trace out (0..6)

    output = np.zeros((N * virtual_node, n_samples))

    for l in range(K_delay, n_samples):
        if l % 200 == 0:
            print(f"    time step {l} / {n_samples}")

        # ---------------------------------------------------------
        # Initialize hidden state: ρᵣ = |0...0><0...0|  (n_hidden qubits)
        # Matches: ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize, 1)
        # ---------------------------------------------------------
        rho_hidden = DensityMatrix.from_label('0' * n_hidden)

        # ---------------------------------------------------------
        # Iterate k from K_delay down to 1
        # Matches: for k in K_delay:-1:1
        # ---------------------------------------------------------
        for k in range(K_delay, 0, -1):

            # Encode input at lag k
            # Matches: para = [Data[l-k, str] for str in features]; cir(para .* pi)
            input_vals = np.array([data.iloc[l - k][feat] for feat in features])
            rho_input  = encode_input(input_vals, n_input)

            # Form joint state: rho_hidden ⊗ rho_input
            # Matches: ρ = ρᵣ ⊗ (cir(ρ₁))
            # Qiskit: tensor(other) puts self (hidden) on higher qubit indices
            #         and other (input) on lower qubit indices
            rho_joint = rho_hidden.tensor(rho_input)

            if k != 1:
                # -------------------------------------------------
                # Intermediate step (Steps I / II in paper)
                # Matches:
                #   ρ = U * ρ * U'
                #   ρᵣ = partial_tr(ρ, Vector(1:InputSize))
                # -------------------------------------------------
                rho_joint  = rho_joint.evolve(U_op)
                rho_hidden = partial_trace(rho_joint, input_qargs)

            else:
                # -------------------------------------------------
                # Final step (Step III in paper) — virtual node readout
                # Matches:
                #   for v in 1:VirtualNode
                #     ρ = δU * ρ * δU'
                #     Output[it, l] = expectation(B[n], ρ)
                # -------------------------------------------------
                it = 0
                for _ in range(virtual_node):
                    rho_joint = rho_joint.evolve(dU_op)
                    for obs in observables:
                        output[it, l] = rho_joint.expectation_value(obs).real
                        it += 1

    return output


# ============================================================
# Rolling ridge regression readout
# Matches the for-loop in cells 12 and 13 of
#   Time_serial_Finance_regression.ipynb
# ============================================================

def rolling_ridge_regression(
    signal     : np.ndarray,
    target     : np.ndarray,
    total      : int,
    L_oos      : int,
    ws         : int   = 0,
    lambda_reg : float = 1e-8,
) -> np.ndarray:
    """
    Rolling-window ridge regression readout.

    For j in 0..L_oos-1:
      y_train = target[ws+j : ws+wi+j]            (wi = total - L_oos = 571 rows)
      x_train = signal[:, ws+j : ws+wi+j]
      W_j = y_train @ x_train.T @ inv(x_train @ x_train.T + lambda * I)
      pred_j = W_j @ signal[:, ws+wi+j]

    Matches Julia notebook:
      W_paras[j,:] = y_train' * transpose(x_train) *
                     inv(x_train * transpose(x_train) + 0.00000001 * I)
      Pre[j] = dot(W_paras[j,:], signal[:, ws+wi+j])

    Parameters
    ----------
    signal     — (out_len, total) reservoir output from quantum_reservoir()
    target     — (total,) RV values for all 816 months
    total      — 816
    L_oos      — 245  (out-of-sample length)
    ws         — 0    (window start; 277 for subsample)
    lambda_reg — 1e-8 (paper: 0.00000001)

    Returns
    -------
    np.ndarray — shape (L_oos,) out-of-sample predictions
    """
    wi      = total - L_oos           # training window size = 571
    out_len = signal.shape[0]
    I_reg   = lambda_reg * np.eye(out_len)
    preds   = np.zeros(L_oos)

    for j in range(L_oos):
        y_train = target[ws + j : ws + wi + j]
        x_train = signal[:, ws + j : ws + wi + j]
        W_j     = y_train @ x_train.T @ np.linalg.inv(
                      x_train @ x_train.T + I_reg)
        preds[j] = W_j @ signal[:, ws + wi + j]

    return preds


# ============================================================
# Metrics
# Exact translations of the metric functions in Time_series.jl
# ============================================================

def MSE(pred, actual):
    """
    Mean Squared Error in the original log-RV scale.
    Matches: MSE(x,y) = mean(((x-y).^2) .* coe)
    """
    return float(np.mean(((pred - actual) ** 2) * COE))


def RMSE(pred, actual):
    """Root MSE. Matches: RMSE(x,y) = sqrt(mean(((x-y).^2) .* coe))"""
    return float(np.sqrt(MSE(pred, actual)))


def MAE(pred, actual):
    """
    Mean Absolute Error in the original log-RV scale.
    Matches: MAE(x,y) = mean(abs.((x-y) .* dif))
    """
    return float(np.mean(np.abs((pred - actual) * DIF)))


def MAPE(pred, actual):
    """
    Mean Absolute Percentage Error (%).
    Matches: MAPE(x,y) = mean(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100
    """
    return float(
        np.mean(np.abs(
            (pred - actual) * DIF / ((actual + 1) * DIF + MIN_RV)
        )) * 100
    )


def compute_qlike(pred, actual):
    """
    QLIKE (Quasi-Likelihood) loss in denormalized log-RV space.
    Matches compute_qlike() in Time_series.jl:
        forecasts = abs.((forecasts.+1)*dif.+Min_RV)
        actuals   = abs.((actuals.+1)*dif.+Min_RV)
        ratio = actuals ./ forecasts
        qlike = sum(ratio - log.(ratio) .- 1)
    """
    f     = np.abs((pred   + 1) * DIF + MIN_RV)
    a     = np.abs((actual + 1) * DIF + MIN_RV)
    ratio = a / f
    return float(np.sum(ratio - np.log(ratio) - 1))


def hitrate(pred, actual):
    """
    Directional accuracy (hit rate).
    Matches hitrate() in Time_series.jl, including the hardcoded anchor
    value that prepends the last training observation.

    Julia:
        x = vcat(-0.5704088242386152, x)
        y = vcat(-0.5704088242386152, y)
        wx = wave(x);  wy = wave(y)
        return sum(wx .== wy) / L
    """
    ANCHOR      = -0.5704088242386152   # last in-sample RV value
    pred_full   = np.concatenate([[ANCHOR], pred])
    actual_full = np.concatenate([[ANCHOR], actual])
    pred_dir    = np.sign(np.diff(pred_full))
    actual_dir  = np.sign(np.diff(actual_full))
    return float(np.mean(pred_dir == actual_dir))

