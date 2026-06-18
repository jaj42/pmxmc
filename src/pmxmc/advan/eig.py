import jax.numpy as jnp
import numpy as np
from jax import lax
from pytensor import wrap_jax

from pmxmc.utils import rate_at


def eigendecomposition(S, B, f=None, y0=None, real_eigenvalues=True):
    """Decompose the linear ODE system x' = Sx + B·r + f into eigenmode form.

    Returns decay rates (lambdas), input coefficients (coefs), forcing coefficients
    (coefs_f), and the initial state y0 projected into eigenmode space (n_cmt, n_cmt).
    Each column of the state matrix corresponds to one eigenmode's contribution.
    """
    eigvals, eigvecs = lax.linalg.eig(
        S,
        enable_eigvec_derivs=True,
        compute_right_eigenvectors=True,
        compute_left_eigenvectors=False,
    )
    if real_eigenvalues:
        eigvals = eigvals.real
        eigvecs = eigvecs.real
    lambdas = -eigvals

    Vinv = jnp.linalg.inv(eigvecs)
    alpha_B = Vinv @ B
    coefs = eigvecs @ jnp.diag(alpha_B / lambdas).T

    if f is not None:
        alpha_f = Vinv @ f
        coefs_f = eigvecs @ jnp.diag(alpha_f / lambdas).T
    else:
        coefs_f = jnp.zeros_like(coefs)

    state_dtype = jnp.float64 if real_eigenvalues else jnp.complex128
    if y0 is not None:
        x0 = jnp.asarray(y0, dtype=state_dtype)
        y0_mod = eigvecs * (Vinv @ x0)  # (n_cmt, n_cmt)
    else:
        n_cmt = len(lambdas)
        y0_mod = jnp.zeros((n_cmt, n_cmt), dtype=state_dtype)

    return lambdas, coefs, coefs_f, y0_mod


@wrap_jax
def eig_advan(
    system_matrix,
    input_matrix,
    meas_time,
    infu_time,
    infu_rate,
    y0=None,
    forcing=None,
    real_eigenvalues=True,
    lag=0.0,
):
    """Solve a linear system via eigendecomposition and return estimates at meas_time.

    The system is x' = S·x + B·r(t) + f, where r(t) is a piecewise-constant infusion
    rate. The solution is propagated with lax.scan over the union of infu_time and
    meas_time breakpoints. Supports a constant forcing term (f), complex eigenvalues,
    a lag time applied to the infusion rate, and a compartment-space initial condition y0.

    Returns an array of shape (n_meas, n_cmt).
    """
    lambdas, coefs, coefs_f, y0_mod = eigendecomposition(
        system_matrix, input_matrix, forcing, y0, real_eigenvalues
    )

    all_times = np.unique(np.concatenate([infu_time, meas_time]))
    starts = jnp.array(all_times[:-1])
    steps = np.diff(all_times)

    rates = rate_at(starts - lag, infu_time, infu_rate)

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + (coefs * rate + coefs_f) * (1 - decay)
        return A_new, A_new

    _, all_states = lax.scan(step_fn, y0_mod, (steps, rates))
    all_states_with_init = jnp.concatenate([y0_mod[None, :, :], all_states], axis=0)

    meas_idx = np.searchsorted(all_times, meas_time)
    states_at_meas = all_states_with_init[meas_idx]  # (n_meas, n_cmt, n_cmt)

    res = jnp.sum(states_at_meas, axis=-1)  # (n_meas, n_cmt)
    if real_eigenvalues:
        return res
    else:
        return res.real
