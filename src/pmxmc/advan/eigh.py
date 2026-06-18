import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor import wrap_jax

from pmxmc.utils import rate_at_numpy as rate_at

pytensor.config.floatX = "float64"
jax.config.update("jax_enable_x64", True)


@wrap_jax
def twocomp_advan(meas_time, infu_time, infu_rate, params, y0=None):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k21 = p["k21"]
    V1 = p["V1"]
    V2 = p["V2"]

    a2 = k12 * jnp.sqrt(V1 / V2)
    k123 = k10 + k12
    S = jnp.asarray(
        [
            [-k123,   a2],
            [   a2, -k21],
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

    lambdas, p_coef = eigendecomposition(S)
    return eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0, scale=V1)


@wrap_jax
def eigh_advan(S, meas_time, infu_time, infu_rate, y0=None, scale=1.0):
    return eigh_advan_worker(S, meas_time, infu_time, infu_rate, y0, scale)


def eigendecomposition(S):
    eigvals, eigvecs = jnp.linalg.eigh(S)
    lambdas = -eigvals
    coefs = eigvecs**2 / lambdas
    return lambdas, coefs


@wrap_jax
def _threecomp_single_occasion(
    dts, rates, boluses, meas_indices, k10, k12, k21, k13, k31, V1
):
    """Single-occasion 3-compartment PK solver (pure JAX, no batching)."""
    a2 = k12 * jnp.sqrt(k21 / k12)
    a3 = k13 * jnp.sqrt(k31 / k13)
    k123 = k10 + k12 + k13
    S = jnp.array(
        [
            [-k123,   a2,   a3],
            [   a2, -k21,    0],
            [   a3,    0, -k31],
        ]
    )  # fmt: skip
    lambdas, coefs = eigendecomposition(S)
    p_coef = coefs[0, :] / V1
    state0 = jnp.zeros(3, dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate, bolus = inputs
        A = A + bolus * lambdas * p_coef
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + p_coef * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates, boluses))
    all_states = jnp.concatenate([state0[None, :], all_states], axis=0)
    return jnp.sum(all_states[meas_indices], axis=-1)


threecomp_advan_vec = pt.vectorize(
    _threecomp_single_occasion,
    signature="(n),(n),(n),(m),(),(),(),(),(),()->(m)",
)


def eigh_advan_worker(
    S,
    meas_time,
    infu_time,
    infu_rate,
    y0=None,
    scale=1.0,
    bolus_time=None,
    bolus_amt=None,
):
    lambdas, coefs = eigendecomposition(S)
    p_coef = coefs[0, :] / scale
    if y0 is None:
        y0 = jnp.zeros_like(lambdas)

    bolus_time = np.asarray([] if bolus_time is None else bolus_time)
    bolus_amt = np.asarray([] if bolus_amt is None else bolus_amt)

    tbeg = min(meas_time[0], infu_time[0])
    if len(bolus_time):
        tbeg = min(tbeg, bolus_time[0])
    tend = meas_time[-1]

    _relevant_itimes = infu_time[(infu_time >= tbeg) & (infu_time <= tend)]
    _relevant_btimes = bolus_time[(bolus_time >= tbeg) & (bolus_time <= tend)]
    _all_times = np.unique(
        np.concatenate([_relevant_itimes, _relevant_btimes, meas_time])
    )
    _dts = np.diff(_all_times)
    _rates = np.array([rate_at(t, infu_time, infu_rate) for t in _all_times[:-1]])

    # Bolus amounts aligned with _all_times[:-1]: applied at interval start.
    _boluses = np.zeros(len(_dts))
    for bt, ba in zip(bolus_time, bolus_amt):
        idxs = np.where(_all_times[:-1] == bt)[0]
        if len(idxs):
            _boluses[idxs[0]] += ba

    dts = jnp.array(_dts)
    rates = jnp.array(_rates)
    boluses = jnp.array(_boluses)

    state0 = jnp.asarray(y0, dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate, bolus = inputs
        A = A + bolus * lambdas * p_coef  # instantaneous bolus at interval start
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + p_coef * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates, boluses))
    # Prepend the initial state so indexing aligns with _all_times
    all_states_with_init = jnp.concatenate(
        [state0[None, :], all_states], axis=0
    )  # (n_steps+1, n_cmt)

    _meas_indices = np.where(np.isin(_all_times, meas_time))[0]
    states_at_meas = all_states_with_init[_meas_indices]  # (n_meas, n_cmt)

    Cp = jnp.sum(states_at_meas, axis=-1)  # (n_meas,)
    return Cp
