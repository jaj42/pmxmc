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
    a2 = k12 * jnp.sqrt(V1 / V2)
    a3 = k13 * jnp.sqrt(V1 / V3)
    k123 = k10 + k12 + k13
    A_sym = jnp.asarray(
        [
            [-k123, a2, a3],
            [a2, -k21, 0],
            [a3, 0, -k31],
        ]
    )
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)
    lambdas = -eigvals
    p_coef = eigvecs[0, :] ** 2 / (V1 * lambdas)
    return lambdas, p_coef


@wrap_jax
def eigh_solver(y0, meas_time, infu_time, infu_rate, params):
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

    state0 = jnp.asarray(y0, dtype=jnp.float64)

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
