import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from pytensor import wrap_jax

from pmxmc.utils import rate_at


def eigendecomposition(S, B):
    eigvals_c, eigvecs_c = lax.linalg.eig(
        S,
        enable_eigvec_derivs=True,
        compute_right_eigenvectors=True,
        compute_left_eigenvectors=False,
    )
    eigvals = eigvals_c.real
    eigvecs = eigvecs_c.real
    lambdas = -eigvals

    alpha = jnp.linalg.inv(eigvecs) @ B
    coefs = eigvecs @ jnp.diag(alpha / lambdas).T

    return lambdas, coefs


@wrap_jax
def eig_advan(system_matrix, input_matrix, meas_time, infu_time, infu_rate, y0=None):
    lambdas, coefs = eigendecomposition(system_matrix, input_matrix)

    tbeg = min(meas_time[0], infu_time[0])
    tend = meas_time[-1]

    _relevant_itimes = infu_time[(infu_time >= tbeg) & (infu_time <= tend)]
    _all_times = np.unique(np.concatenate([_relevant_itimes, meas_time]))
    _dts = np.diff(_all_times)
    _rates = np.array([rate_at(t, infu_time, infu_rate) for t in _all_times[:-1]])

    dts = jnp.array(_dts)
    rates = jnp.array(_rates)

    if y0 is None:
        n_cmt = len(lambdas)
        y0 = jnp.zeros((n_cmt, n_cmt), dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + coefs * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, y0, (dts, rates))
    all_states_with_init = jnp.concatenate([y0[None, :, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(_all_times, meas_time))[0]
    states_at_meas = all_states_with_init[_meas_indices]  # (n_meas, n_cmt, n_cmt)

    return jnp.sum(states_at_meas, axis=-1)  # (n_meas, n_cmt)
