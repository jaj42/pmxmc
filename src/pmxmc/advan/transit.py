import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import gammainc, gammaln
from pytensor import wrap_jax

from pmxmc.advan.eig import eigendecomposition

# alpha = KTR - lambda. Three regimes (see notes 2-4):
#   alpha >  _ALPHA_EPS : main branch, (KTR/alpha)^n * P(n, alpha*t) via gammainc.
#   |alpha| <= _ALPHA_EPS: analytic limit, (KTR*t)^n / Gamma(n+1).
#   alpha < -_ALPHA_EPS : flip-flop (KTR < lambda). gammainc cannot take the negative
#       argument, so we use the all-positive convergent series
#       int_0^t s^{n-1} e^{beta s} ds = t^n * sum_k (beta t)^k / (k! (n+k)), beta = -alpha,
#       with the e^{-lambda t} decay folded in per term for numerical stability.
_ALPHA_EPS = 1e-6

# Number of series terms for the alpha<0 branch. The series converges once k > beta*t, so
# this is accurate for beta*t up to roughly _FLIPFLOP_TERMS/2; well beyond the point where
# the response (~ (KTR*t)^n e^{-KTR t}) has decayed to numerical noise for realistic PK.
_FLIPFLOP_TERMS = 200


@wrap_jax
def transit_advan(
    system_matrix,
    input_matrix,
    meas_time,
    n,
    MTT,
    infu_time=None,
    infu_rate=None,
    bolus_time=None,
    bolus_amt=None,
    real_eigenvalues=True,
):
    """Closed-form transit-compartment absorption coupled to a linear PK eigensystem.

    Models delayed absorption as a Gamma-distributed transit chain (continuous shape
    ``n``, mean transit time ``MTT``) convolved against the eigensystem of
    ``x' = S·x + B·r``. The whole response is an exact, ODE-free, differentiable sum of
    regularised incomplete-Gamma terms (one per eigenmode, see
    ``notes/transit_compartment_closed_form_insights.md``).

    Drug enters either as instantaneous boluses (``bolus_time`` / ``bolus_amt``) or as a
    piecewise-constant infusion (``infu_time`` / ``infu_rate``); both are supported and may
    be combined by superposition. ``meas_time`` and the dose timing arrays are static
    numpy; only ``n``, ``MTT`` and the eigensystem are traced, so the response is fully
    differentiable in the transit and PK parameters.

    Parameters
    ----------
    system_matrix, input_matrix : the PK system ``S`` and input matrix ``B``.
    meas_time : numpy array of measurement times, shape ``(n_meas,)``.
    n : Gamma shape of the transit delay (continuous, ``> 1`` recommended; see notes 3.3/4.5).
        The transit input to the PK system is ``Gamma(shape=n, rate=KTR)``.
    MTT : mean transit time (the exact mean of the delay). ``KTR = n / MTT``. Note this
        differs from Savic's ``(n + 1) / MTT``: that convention counts an extra absorption
        compartment, which this closed form does not model (transit feeds the PK system
        directly). See ``notes/transit_compartment_closed_form_insights.md``.
    infu_time, infu_rate : piecewise-constant infusion breakpoints / rates.
    bolus_time, bolus_amt : bolus dose times / amounts.
    real_eigenvalues : must be True — ``gammainc`` requires real decay rates. Always true
        for standard mammillary PK models.

    Returns
    -------
    Array of shape ``(n_meas, n_cmt)``, consistent with ``eig_advan``.

    Notes
    -----
    input_matrix ``B`` follows ``eig_advan``'s convention: a 1-D vector of shape
    ``(n_cmt,)`` (single input channel, e.g. ``[1/V1, 0, 0]``), not a 2-D column.

    Each mode uses ``alpha_i = KTR - lambda_i`` and is evaluated on the appropriate branch:
    the incomplete-gamma form for ``alpha_i > 0``, the analytic ``alpha -> 0`` limit near
    zero, and an all-positive convergent series for ``alpha_i < 0`` (the flip-flop /
    slow-transit regime, ``KTR < lambda_i``). All signs are supported; see the notes file
    and ``tests/test_transit.py::TestFlipFlop``.
    """
    if not real_eigenvalues:
        raise ValueError(
            "transit_advan requires real_eigenvalues=True (gammainc needs real rates)."
        )

    # Reuse the eig.py eigendecomposition. lambdas[i] are positive decay rates; coefs[j,i]
    # is the infusion-gain coefficient for compartment j, mode i (already folds in B's
    # volume scaling and divides by lambdas). The bolus residue is w[j,i] = coefs[j,i]*r_i.
    lambdas, coefs, _, _ = eigendecomposition(
        system_matrix, input_matrix, None, None, real_eigenvalues
    )
    w = coefs * lambdas[None, :]  # (n_cmt, n_cmt) indexed [compartment j, mode i]

    # The transit chain feeds the PK system directly (no separate ka/absorption
    # compartment), so the delay is Gamma(shape=n, rate=KTR) with mean n/KTR. Setting
    # KTR = n / MTT makes MTT the exact mean transit time (see notes 2/3). This differs
    # from Savic's KTR = (n+1)/MTT, which counts an extra absorption compartment we don't
    # model here.
    KTR = n / MTT
    alpha = KTR - lambdas  # (n_modes,)
    log_norm = gammaln(n + 1.0)  # for the alpha -> 0 limit branch

    a = alpha[:, None, None]  # (i, 1, 1)
    lam = lambdas[:, None, None]
    # Per-mode regime masks (see the header comment). All branches are evaluated with
    # "safe" arguments so the unused ones stay finite (no NaN/Inf to poison gradients
    # through jnp.where), then selected.
    pos = a > _ALPHA_EPS
    neg = a < -_ALPHA_EPS
    any_neg = jnp.any(neg)  # scalar; independent of dt, so it gates both _kernels calls
    a_pos = jnp.where(pos, a, 1.0)  # (KTR/alpha)^n for the main branch, finite where unused
    beta = jnp.where(neg, -a, 1.0)  # -alpha > 0 for the flip-flop series, finite elsewhere

    ks = jnp.arange(_FLIPFLOP_TERMS)  # series index k = 0 .. K-1
    log_series_den = gammaln(ks + 1.0) + jnp.log(n + ks)  # log(k! (n+k)), (K,)

    n_meas = len(meas_time)
    n_cmt = lambdas.shape[0]

    def _kernels(dt):
        """Per-mode bolus and infusion-step kernels for static offsets dt (n_meas, n_ev).

        Returns two (n_modes, n_meas, n_ev) arrays:
          bolus_K = e^{-r_i dt} (KTR/alpha_i)^n P(n, alpha_i dt)         [phi_i / w]
          step_K  = P(n, KTR dt) - bolus_K                              [S_i / coefs]
        valid for all signs of alpha_i (main / limit / flip-flop-series branches).
        """
        dt = jnp.asarray(dt)
        mask = (dt > 0)[None]  # (1, n_meas, n_ev)
        dtp = jnp.maximum(dt, 0.0)[None]  # clamp so gammainc args stay non-negative
        dtp_safe = jnp.where(dtp > 0.0, dtp, 1.0)  # avoid log(0); cells discarded by mask
        decay = jnp.exp(-lam * dtp)

        # alpha > 0 : bolus_K = e^{-lambda t} (KTR/alpha)^n P(n, alpha t).
        gi = gammainc(n, jnp.where(pos, a * dtp, 0.0))
        bolus_pos = decay * (KTR / a_pos) ** n * gi

        # |alpha| ~ 0 : analytic limit e^{-lambda t} (KTR t)^n / Gamma(n+1).
        bolus_lim = decay * (KTR * dtp) ** n * jnp.exp(-log_norm)

        # alpha < 0 : e^{-lambda t} (KTR t)^n / Gamma(n) * sum_k (beta t)^k / (k! (n+k)).
        # Fold e^{-lambda t} into each term (log-space) so nothing overflows: the summand
        # e^{-lambda t} (beta t)^k / k! peaks near k = beta t at ~e^{-KTR t}, which is
        # representable, while the bare sum ~ e^{beta t} would not be.
        # Guarded by lax.cond on any_neg (constant across dt): when no mode is in the
        # flip-flop regime the whole K-term series is skipped at runtime (fwd and grad),
        # so the common KTR > max(lambda) case pays nothing for it.
        def _series_bolus_neg(_):
            log_bd = jnp.log(jnp.where(neg, beta * dtp_safe, 1.0))  # log(beta t), (1,M,E)
            log_terms = (
                -lam * dtp  # (i,M,E) decay, broadcast over k below
                + ks[:, None, None, None] * log_bd[None]  # k log(beta t)
                - log_series_den[:, None, None, None]  # log(k! (n+k))
            )  # (K, i, M, E)
            series = jnp.sum(jnp.exp(log_terms), axis=0)  # (i, M, E)
            return jnp.exp(n * jnp.log(KTR * dtp_safe) - gammaln(n)) * series

        bolus_neg = lax.cond(
            any_neg, _series_bolus_neg, lambda _: jnp.zeros_like(decay), None
        )

        bolus_K = jnp.where(pos, bolus_pos, jnp.where(neg, bolus_neg, bolus_lim)) * mask
        step_K = (gammainc(n, KTR * dtp) - bolus_K) * mask
        return bolus_K, step_K

    result = jnp.zeros((n_meas, n_cmt))

    # Piecewise-constant infusion -> rate-change events dR_k at tau_k, summed by
    # superposition of step responses S_i (gain coefs[j,i]).
    if infu_time is not None and infu_rate is not None:
        infu_time = np.asarray(infu_time, dtype=float).ravel()
        infu_rate = np.asarray(infu_rate, dtype=float).ravel()
        dR = np.empty_like(infu_rate)
        dR[0] = infu_rate[0]
        dR[1:] = np.diff(infu_rate)

        dt_inf = meas_time[:, None] - infu_time[None, :]  # (n_meas, n_inf) static
        _, step_K = _kernels(dt_inf)
        Ks = jnp.einsum("e,ime->im", jnp.asarray(dR), step_K)
        result = result + jnp.einsum("ji,im->mj", coefs, Ks)

    # Boluses -> superposition of impulse responses phi_i (residue weight w[j,i]).
    if bolus_time is not None and bolus_amt is not None:
        bolus_time = np.asarray(bolus_time, dtype=float).ravel()
        bolus_amt = np.asarray(bolus_amt, dtype=float).ravel()

        dt_bol = meas_time[:, None] - bolus_time[None, :]  # (n_meas, n_bol) static
        bolus_K, _ = _kernels(dt_bol)
        Kb = jnp.einsum("e,ime->im", jnp.asarray(bolus_amt), bolus_K)
        result = result + jnp.einsum("ji,im->mj", w, Kb)

    return result
