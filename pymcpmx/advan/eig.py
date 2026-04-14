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


def eigendecomposition(k10, k12, k13, k21, k31, V1):
    """
    Eigendecompose the 3-compartment PK concentration matrix using jnp.linalg.eig.

    A_conc is the concentration dynamics matrix (dC/dt = A_conc @ C + [1/V1,0,0]*rate):
        [[-k123,  k12,  k13],
         [  k21, -k21,    0],
         [  k31,    0, -k31]]

    Returns:
        lambdas : (3,) positive decay constants (-eigenvalues)
        p_coef  : (3,) plasma-concentration coefficients, sum(p_coef) = 1/CL
    """
    k123 = k10 + k12 + k13

    A_conc = jnp.asarray(
        [
            [-k123, k12, k13],
            [k21, -k21, 0.0],
            [k31, 0.0, -k31],
        ]
    )

    # eig returns complex arrays; for a stable PK system eigenvalues are real.
    eigvals_c, eigvecs_c = jnp.linalg.eig(A_conc, allow_eigvec_deriv=True)
    eigvals = eigvals_c.real
    eigvecs = eigvecs_c.real

    lambdas = -eigvals  # positive decay constants

    # Modal coordinates for a unit concentration bolus in compartment 1.
    C0 = jnp.asarray([1.0 / V1, 0.0, 0.0])
    # alpha = jnp.linalg.solve(eigvecs, C0)
    alpha = jnp.linalg.inv(eigvecs) @ C0  # µg/L

    # p_coef[i] = eigvecs[0,i] * alpha[i] / lambda[i]
    # → sum(p_coef) = 1/CL at steady state (unit infusion rate).
    p_coef = eigvecs[0, :] * alpha / lambdas

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

    lambdas, p_coef = eigendecomposition(k10, k12, k13, k21, k31, V1)

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

    state0 = jnp.zeros(3, dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + p_coef * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates))
    all_states_with_init = jnp.concatenate([state0[None, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(_all_times, _meas))[0]
    states_at_meas = all_states_with_init[_meas_indices]

    Cp = jnp.sum(states_at_meas, axis=-1)
    return Cp
