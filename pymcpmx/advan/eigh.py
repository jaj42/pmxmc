import jax
import jax.numpy as jnp
import numpy as np
from pytensor import wrap_jax


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = int(np.clip(idx, 0, len(infu_rate) - 1))
    return float(infu_rate[idx])


def eigendecomposition(k10, k12, k13, k21, k31, V1, V2, V3):
    """
    Eigendecompose the 3-compartment PK concentration matrix via its
    symmetrized form.

    The concentration dynamics matrix A_conc (dC/dt = A_conc @ C + [1/V1,0,0]*rate)
    is related to the symmetric matrix A_sym = D^{1/2} A_conc D^{-1/2},
    D = diag(V1,V2,V3), by a similarity transform that preserves eigenvalues.

    Using jnp.linalg.eigh on A_sym avoids:
      - complex arithmetic (jnp.linalg.eig returns complex arrays)
      - non-differentiable argsort (eigh always returns sorted real eigenvalues)
      - explicit matrix inversion (the orthonormality of eigh eigenvectors
        gives p_coef[i] = w[0,i]^2 / (V1 * lambda_i) directly)

    Returns:
        lambdas : (3,) positive decay constants (-eigenvalues, descending)
        p_coef  : (3,) plasma-concentration coefficients, sum(p_coef) = 1/CL
    """
    k123 = k10 + k12 + k13

    # Off-diagonal coupling terms of the symmetrized matrix:
    #   A_sym[0,1] = Q2 / sqrt(V1*V2) = k12 * sqrt(V1/V2) = k21 * sqrt(V2/V1)
    a01 = k12 * jnp.sqrt(V1 / V2)
    a02 = k13 * jnp.sqrt(V1 / V3)

    A_sym = jnp.asarray(
        [
            [-k123, a01, a02],
            [a01, -k21, 0],
            [a02, 0, -k31],
        ]
    )

    # eigh: real symmetric solver — real outputs, eigenvalues ascending,
    # orthonormal eigenvectors, fully differentiable via standard Hermitian JVP.
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)

    lambdas = -eigvals  # positive decay constants (descending order)

    # Plasma coefficients derived from the symmetry transform:
    #   V_conc[:,i] = D^{-1/2} w_i  →  V_conc[0,i] = w_i[0] / sqrt(V1)
    #   (V_conc^{-1})[i,0] = (W^T D^{1/2})[i,0] = w_i[0] * sqrt(V1)
    # so  p_coef[i] = V_conc[0,i] * (V_conc^{-1})[i,0] / (V1 * lambda_i)
    #              = w_i[0]^2 / (V1 * lambda_i)
    p_coef = eigvecs[0, :] ** 2 / (V1 * lambdas)

    #   V_vec = jnp.array([V1, V2, V3])            # (3,)
    #   coefs = (eigvecs * eigvecs[0, :]            # (3,3) * (3,)  →  w[j,i] * w[0,i]
    #            / (jnp.sqrt(V1 * V_vec[:, None])  # (3,1)          →  / sqrt(V1·Vj)
    #               * lambdas))                     # (3,)           →  / lambda_i

    #   coefs[j, i] is the concentration contribution of mode i to compartment j. The state in the scan then becomes shape (3, 3) and you get all
    #    compartment concentrations from jnp.sum(states, axis=-1) — a (n_meas, 3) array with columns [Cp, C2, C3].

    return lambdas, p_coef


@wrap_jax
def eigen_wrapper(y0, meas_time, infu_time, infu_rate, params):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k13 = p["k13"]
    k21 = p["k21"]
    k31 = p["k31"]
    V1 = p["V1"]
    V2 = p["V2"]
    V3 = p["V3"]

    lambdas, p_coef = eigendecomposition(k10, k12, k13, k21, k31, V1, V2, V3)

    # Build time grid (identical logic to model.py) ---------------------------
    _meas = np.asarray(meas_time)
    _itimes = np.asarray(infu_time)
    _irates = np.asarray(infu_rate)

    _start = min(float(_meas[0]), float(_itimes[0]))
    _end = float(_meas[-1])

    _relevant_itimes = _itimes[(_itimes >= _start) & (_itimes <= _end)]
    _all_times = np.unique(np.concatenate([_relevant_itimes, _meas]))
    _dts = np.diff(_all_times)
    _rates = np.array([rate_at(t, _itimes, _irates) for t in _all_times[:-1]])

    dts = jnp.array(_dts)
    rates = jnp.array(_rates)

    state0 = jnp.zeros(3, dtype=jnp.float64)  # one entry per eigenmode

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)  # (3,)
        A_new = A * decay + p_coef * rate * (1 - decay)  # (3,)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates))
    # Prepend the initial state so indexing aligns with _all_times
    all_states_with_init = jnp.concatenate(
        [state0[None, :], all_states], axis=0
    )  # (n_steps+1, 3)

    _meas_indices = np.where(np.isin(_all_times, _meas))[0]
    states_at_meas = all_states_with_init[_meas_indices]  # (n_meas, 3)

    Cp = jnp.sum(states_at_meas, axis=-1)  # (n_meas,)
    return Cp


# Cp = eigen_wrapper(
#     y0=[0, 0, 0],
#     meas_time=meas_time,
#     infu_time=infu_time,
#     infu_rate=infu_rate,
#     params=pk_params,
# )
# C_preds.append(Cp)
