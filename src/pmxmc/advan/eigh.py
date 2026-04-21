import jax
import jax.numpy as jnp
import numpy as np
from pytensor import wrap_jax
from pmxmc.utils import rate_at


@wrap_jax
def twocomp_advan(meas_time, infu_time, infu_rate, params, y0=None):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k21 = p["k21"]
    V1 = p["V1"]
    V2 = p["V2"]

    a2 = k12 * jnp.sqrt(V1 / V2)
    k12 = k10 + k12
    S = jnp.asarray(
        [
            [-k12,   a2],
            [  a2, -k21],
        ]
    )  # fmt: skip
    return eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0, scale=V1)


@wrap_jax
def threecomp_advan(meas_time, infu_time, infu_rate, params, y0=None):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k13 = p["k13"]
    k21 = p["k21"]
    k31 = p["k31"]
    V1 = p["V1"]
    V2 = p["V2"]
    V3 = p["V3"]

    a2 = k12 * jnp.sqrt(V1 / V2)
    a3 = k13 * jnp.sqrt(V1 / V3)
    k123 = k10 + k12 + k13
    S = jnp.asarray(
        [
            [-k123,   a2,   a3],
            [   a2, -k21,    0],
            [   a3,    0, -k31],
        ]
    )  # fmt: skip

    lambdas, p_coef = eigendecomposition(S, V1, 0)
    return eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0, scale=V1)


@wrap_jax
def eigh_advan(S, meas_time, infu_time, infu_rate, y0=None, scale=1.0):
    return eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0, scale)


def eigendecomposition(S, scale, cmt=0):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    lambdas = -eigvals
    coef = eigvecs[cmt, :] ** 2 / (scale * lambdas)
    return lambdas, coef


def eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0=None, scale=1.0):
    lambdas, p_coef = eigendecomposition(S, scale, 0)
    if y0 is None:
        y0 = jnp.zeros_like(lambdas)

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
        decay = jnp.exp(-lambdas * dt)  # (3,)
        A_new = A * decay + p_coef * rate * (1 - decay)  # (3,)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates))
    # Prepend the initial state so indexing aligns with _all_times
    all_states_with_init = jnp.concatenate(
        [state0[None, :], all_states], axis=0
    )  # (n_steps+1, 3)

    _meas_indices = np.where(np.isin(_all_times, meas_time))[0]
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
