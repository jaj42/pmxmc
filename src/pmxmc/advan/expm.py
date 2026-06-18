import jax
import jax.numpy as jnp
import numpy as np
from pytensor import wrap_jax

from pmxmc.utils import rate_at


@wrap_jax
def expm_advan(
    system_matrix, input_matrix, meas_time, infu_time, infu_rate, y0=None, lag=0.0
):
    S = system_matrix
    B = input_matrix
    S_inv = jnp.linalg.inv(S)
    Eye = jnp.eye(S.shape[0])

    if y0 is not None:
        state0 = jnp.asarray(y0, dtype=jnp.float64)
    else:
        state0 = jnp.zeros_like(B, dtype=jnp.float64)

    all_times = np.unique(np.concatenate([infu_time, meas_time]))
    starts = jnp.array(all_times[:-1])
    steps = np.diff(all_times)

    rates = rate_at(starts - lag, infu_time, infu_rate)

    def step_fn(state, inputs):
        dt, rate = inputs
        exp_Sdt = jax.scipy.linalg.expm(S * dt)
        state_new = exp_Sdt @ state + S_inv @ (exp_Sdt - Eye) @ B * rate
        return state_new, state_new

    _, all_states = jax.lax.scan(step_fn, state0, (steps, rates))
    all_states_with_init = jnp.concatenate([state0[None, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(all_times, meas_time))[0]
    # return all_states_with_init[_meas_indices, 0]  # A1 at measurement times
    return all_states_with_init[_meas_indices, :]
