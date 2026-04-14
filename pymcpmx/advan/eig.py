import jax
import jax.numpy as jnp
import numpy as np
from pytensor import wrap_jax


def rate_at(t, infu_time, infu_rate):
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = int(np.clip(idx, 0, len(infu_rate) - 1))
    return float(infu_rate[idx])


def eigendecomposition(k10, k12, k13, k21, k31, V1):
    k123 = k10 + k12 + k13
    A_conc = jnp.asarray(
        [
            [-k123, k12, k13],
            [k21, -k21, 0.0],
            [k31, 0.0, -k31],
        ]
    )
    eigvals_c, eigvecs_c = jnp.linalg.eig(A_conc, allow_eigvec_deriv=True)
    eigvals = eigvals_c.real
    eigvecs = eigvecs_c.real
    lambdas = -eigvals  # positive decay constants

    # Modal coordinates for a unit concentration bolus in compartment 1.
    C0 = jnp.asarray([1.0 / V1, 0.0, 0.0])
    alpha = jnp.linalg.inv(eigvecs) @ C0  # µg/L

    # p_coef[i] = eigvecs[0,i] * alpha[i] / lambda[i]
    # → sum(p_coef) = 1/CL at steady state (unit infusion rate).
    p_coef = eigvecs[0, :] * alpha / lambdas

    return lambdas, p_coef


@wrap_jax
def eig_solver(y0, meas_time, infu_time, infu_rate, params):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k13 = p["k13"]
    k21 = p["k21"]
    k31 = p["k31"]
    V1 = p["V1"]

    lambdas, p_coef = eigendecomposition(k10, k12, k13, k21, k31, V1)

    # _start = min(float(meas_time[0]), float(infu_time[0]))
    # _end = float(meas_time[-1])
    tbeg = min(meas_time[0], infu_time[0])
    tend = meas_time[-1]

    _relevant_itimes = infu_time[(infu_time >= tbeg) & (infu_time <= tend)]
    _all_times = np.unique(np.concatenate([_relevant_itimes, meas_time]))
    _dts = np.diff(_all_times)
    _rates = np.array([rate_at(t, infu_time, infu_rate) for t in _all_times[:-1]])

    dts = jnp.array(_dts)
    rates = jnp.array(_rates)

    state0 = jnp.asarray(y0, dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + p_coef * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates))
    all_states_with_init = jnp.concatenate([state0[None, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(_all_times, meas_time))[0]
    states_at_meas = all_states_with_init[_meas_indices]

    Cp = jnp.sum(states_at_meas, axis=-1)
    return Cp
