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


@wrap_jax
def expm_advan(y0, meas_time, infu_time, infu_rate, params):
    p = params
    k10 = p["k10"]
    k12 = p["k12"]
    k13 = p["k13"]
    k21 = p["k21"]
    k31 = p["k31"]

    S = jnp.zeros((3, 3))
    S = S.at[0, 0].set(-(k10 + k12 + k13))
    S = S.at[0, 1].set(k21)
    S = S.at[0, 2].set(k31)
    S = S.at[1, 0].set(k12)
    S = S.at[1, 1].set(-k21)
    S = S.at[2, 0].set(k13)
    S = S.at[2, 2].set(-k31)

    B = jnp.array([1.0, 0.0, 0.0])
    S_inv = jnp.linalg.inv(S)
    I3 = jnp.eye(3)

    # Build time grid ----------------------------------------------------------
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

    def step_fn(state, inputs):
        dt, rate = inputs
        exp_Sdt = jax.scipy.linalg.expm(S * dt)
        state_new = exp_Sdt @ state + S_inv @ (exp_Sdt - I3) @ B * rate
        return state_new, state_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates))
    all_states_with_init = jnp.concatenate([state0[None, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(_all_times, _meas))[0]
    return all_states_with_init[_meas_indices, 0]  # A1 at measurement times


# A1 = matrix_exp_wrapper(
#     y0=[0, 0, 0],
#     meas_time=meas_time,
#     infu_time=infu_time,
#     infu_rate=infu_rate,
#     params=pk_params,
# )
# C_preds.append(A1 / V1_i[bio_idx])
