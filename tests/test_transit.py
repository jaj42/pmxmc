"""Tests for the closed-form transit-compartment absorption model (``transit_advan``).

The closed form in ``pmxmc.advan.transit`` models delayed absorption as a
Gamma(shape=n, rate=KTR) transit input convolved against the linear-PK eigensystem,
returning a sum of regularised incomplete-Gamma terms with no ODE solve.

We validate it against an **independent numerical oracle**: a diffrax linear chain of
``k`` transit sub-compartments (each with per-compartment rate ``kt``), matching the
classic "linear chain trick" (cf. icomo's ``erlang_kernel``). For integer shape ``k`` the
chain output is a Gamma(shape=k, rate=kt) input to the PK system, so ``transit_advan`` with
``n = k`` and ``MTT = n / kt`` (i.e. ``KTR = kt``) must reproduce the chain trajectory.

Covers:
  - Bolus dosing, 1- and 2-compartment PK, several integer n (equivalence to the chain).
  - Multiple boluses (superposition).
  - Piecewise-constant infusion (the dR step decomposition).
  - Convention pinning: the effective mean transit time of the modelled input equals MTT
    (and would NOT under Savic's (n+1)/MTT convention) — this is the test that guards the
    KTR = n / MTT parameterisation.
  - The alpha -> 0 (KTR ~= lambda) analytic-limit branch: continuity and finite gradients.
"""

import unittest

import jax

# x64 must be enabled before jax.numpy is imported, so the imports below follow it (E402).
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pmxmc.advan.transit import transit_advan  # noqa: E402


# ---------------------------------------------------------------------------
# Independent numerical oracle: linear chain of transit compartments -> PK system
# ---------------------------------------------------------------------------
def chain_oracle(S, B, k, kt, times, *, bolus_D=None, infusion=None):
    """Integrate a k-stage transit chain (per-comp rate kt) feeding PK system x'=Sx+B*out.

    Drug enters transit compartment 0 either as a t=0 bolus of ``bolus_D`` or as a
    piecewise-constant ``infusion=(breakpoints, rates)`` (rate held from each breakpoint).
    The last transit compartment's outflow ``kt * A_{k-1}`` is the input rate to the PK
    system. Returns the PK-state trajectory of shape ``(len(times), n_cmt)``.
    """
    S = jnp.asarray(S, dtype=float)
    b = jnp.asarray(B, dtype=float).ravel()
    n_cmt = S.shape[0]

    if infusion is not None:
        bp = jnp.asarray(infusion[0], dtype=float)
        rt = jnp.asarray(infusion[1], dtype=float)

        def inflow(t):
            idx = jnp.sum(t >= bp) - 1
            return jnp.where(idx >= 0, rt[jnp.clip(idx, 0, rt.size - 1)], 0.0)
    else:
        def inflow(t):
            return 0.0

    def rhs(t, y, args):
        trans = y[:k]
        pk = y[k:]
        d = jnp.zeros_like(y)
        d = d.at[0].add(inflow(t) - kt * trans[0])
        d = d.at[1:k].add(kt * trans[:-1] - kt * trans[1:])
        outflow = kt * trans[-1]
        d = d.at[k:].add(S @ pk + b * outflow)
        return d

    y0 = jnp.zeros(k + n_cmt)
    if bolus_D is not None:
        y0 = y0.at[0].set(bolus_D)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=float(times[-1]),
        dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.asarray(times, dtype=float)),
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-13),
        max_steps=200000,
    )
    return np.asarray(sol.ys[:, k:])


def call_transit(S, B, times, n, MTT, **kw):
    """Evaluate the wrap_jax'd transit_advan to a concrete numpy array."""
    out = transit_advan(
        np.asarray(S, dtype=float),
        np.asarray(B, dtype=float),
        np.asarray(times, dtype=float),
        float(n),
        float(MTT),
        **kw,
    )
    return np.asarray(out.eval())


# Standard mammillary PK systems (real eigenvalues), input into the central compartment.
# The eigendecomposition expects the input matrix B as a 1-D vector of shape (n_cmt,)
# (single input channel), matching eig_advan's convention, e.g. B = [1/V1, 0, 0].
def one_cmt(ke=0.3):
    return np.array([[-ke]]), np.array([1.0])


def two_cmt(k10=0.25, k12=0.5, k21=0.2):
    S = np.array([[-(k10 + k12), k21], [k12, -k21]])
    B = np.array([1.0, 0.0])
    return S, B


class TestTransitAdvanEquivalence(unittest.TestCase):
    """transit_advan reproduces the diffrax linear-chain oracle for integer shapes."""

    times = np.linspace(0.05, 30.0, 80)

    def _check_bolus(self, S, B, n, kt, D=100.0, rtol=2e-4):
        # KTR = n / MTT, so MTT = n / kt makes KTR == kt (the chain's per-comp rate).
        # NB: kt must exceed every decay rate lambda so that alpha = KTR - lambda > 0 for
        # all modes; the closed form's incomplete-gamma branch is only valid there (the
        # alpha < 0 flip-flop regime is a known limitation, see TestFlipFlopLimitation).
        MTT = n / kt
        got = call_transit(
            S, B, self.times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        ref = chain_oracle(S, B, n, kt, self.times, bolus_D=D)
        denom = np.max(np.abs(ref))
        self.assertLess(np.max(np.abs(got - ref)) / denom, rtol)

    def test_bolus_one_cmt(self):
        S, B = one_cmt(ke=0.3)
        for n in (2, 5, 10):
            with self.subTest(n=n):
                self._check_bolus(S, B, n, kt=0.8)

    def test_bolus_two_cmt(self):
        S, B = two_cmt()  # decay rates lambda ~= {0.894, 0.056}
        for n in (2, 5):
            with self.subTest(n=n):
                self._check_bolus(S, B, n, kt=2.0)  # KTR = 2.0 > max lambda

    def test_multiple_boluses_superposition(self):
        S, B = one_cmt(ke=0.25)
        n, kt = 4, 0.9
        MTT = n / kt
        times = np.linspace(0.05, 40.0, 120)
        b_t = np.array([0.0, 8.0, 20.0])
        b_a = np.array([100.0, 60.0, 80.0])

        got = call_transit(
            S, B, times, n, MTT, bolus_time=b_t, bolus_amt=b_a,
        )
        # Oracle superposition: shift each bolus response in time and sum.
        ref = np.zeros_like(got)
        for t0, amt in zip(b_t, b_a):
            mask = times >= t0
            local = chain_oracle(S, B, n, kt, times[mask] - t0, bolus_D=amt)
            ref[mask] += local
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 3e-4)

    def test_infusion_one_cmt(self):
        S, B = one_cmt(ke=0.3)
        n, kt = 5, 0.8
        MTT = n / kt
        times = np.linspace(0.05, 40.0, 120)
        # Zero-order infusion of rate R over [0, T), then off.
        R, T = 10.0, 12.0
        infu_time = np.array([0.0, T])
        infu_rate = np.array([R, 0.0])

        got = call_transit(
            S, B, times, n, MTT, infu_time=infu_time, infu_rate=infu_rate,
        )
        ref = chain_oracle(
            S, B, n, kt, times, infusion=(infu_time, infu_rate),
        )
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 5e-4)


class TestTransitConvention(unittest.TestCase):
    """Guards KTR = n / MTT: MTT is the *exact* mean transit time of the modelled input."""

    def _chain_input_mean(self, kt, k):
        """First moment of a k-stage chain's output rate (impulse response), from diffrax.

        Independent of the closed form / gammainc: integrates kt*A_{k-1}(t) directly.
        """
        times = np.linspace(0.0, 400.0 / kt, 40000)

        def rhs(t, y, args):
            trans = y
            d = jnp.zeros_like(y)
            d = d.at[0].add(-kt * trans[0])
            d = d.at[1:].add(kt * trans[:-1] - kt * trans[1:])
            return d

        y0 = jnp.zeros(k).at[0].set(1.0)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=0.0, t1=float(times[-1]),
            dt0=0.001, y0=y0, saveat=diffrax.SaveAt(ts=jnp.asarray(times)),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
            max_steps=400000,
        )
        out = np.asarray(kt * sol.ys[:, -1])  # outflow rate; a normalised density in t
        # mean = ∫ t f(t) dt / ∫ f(t) dt  (f already integrates to ~1)
        num = np.trapz(times * out, times) if hasattr(np, "trapz") else np.trapezoid(times * out, times)
        den = np.trapz(out, times) if hasattr(np, "trapz") else np.trapezoid(out, times)
        return num / den

    def test_mean_transit_time_equals_MTT(self):
        # For the modelled Gamma(shape=n, rate=KTR=n/MTT), mean = n/KTR = MTT.
        # The oracle chain that transit_advan matches has per-comp rate kt = KTR = n/MTT;
        # its measured first moment must equal MTT (not Savic's (n+1)/MTT).
        for n, MTT in ((3, 6.0), (8, 10.0)):
            with self.subTest(n=n, MTT=MTT):
                kt = n / MTT  # == KTR under the corrected convention
                measured = self._chain_input_mean(kt, n)
                self.assertAlmostEqual(measured / MTT, 1.0, delta=2e-3)
                # Sanity: the OLD convention (kt=(n+1)/MTT) would give mean n/(n+1)*MTT.
                self.assertLess(abs(measured / MTT - 1.0), abs(n / (n + 1.0) - 1.0) / 2)

    def test_old_convention_would_mismatch_chain(self):
        # transit_advan(MTT) matches a chain at rate n/MTT; it must NOT match one at the
        # old (n+1)/MTT rate — demonstrating the correction changes behaviour.
        S, B = one_cmt(ke=0.3)
        n, MTT, D = 4, 5.0, 100.0
        times = np.linspace(0.05, 30.0, 80)
        got = call_transit(
            S, B, times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        ref_new = chain_oracle(S, B, n, n / MTT, times, bolus_D=D)
        ref_old = chain_oracle(S, B, n, (n + 1) / MTT, times, bolus_D=D)
        err_new = np.max(np.abs(got - ref_new)) / np.max(np.abs(ref_new))
        err_old = np.max(np.abs(got - ref_old)) / np.max(np.abs(ref_old))
        self.assertLess(err_new, 5e-4)
        self.assertGreater(err_old, 1e-2)


class TestAlphaLimit(unittest.TestCase):
    """The alpha = KTR - lambda -> 0 analytic-limit branch (transit.py _ALPHA_EPS)."""

    def test_limit_branch_matches_chain(self):
        # Choose ke == KTR so alpha == 0 exactly for the single mode -> limit branch.
        n, MTT, D = 5, 6.0, 100.0
        kt = n / MTT
        S, B = one_cmt(ke=kt)  # ke == KTR
        times = np.linspace(0.05, 30.0, 80)
        got = call_transit(
            S, B, times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        ref = chain_oracle(S, B, n, kt, times, bolus_D=D)
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 5e-4)

    def test_gradient_finite_across_branch(self):
        import pytensor
        import pytensor.tensor as pt

        n, D = 5.0, 100.0
        times = np.linspace(0.05, 30.0, 40)
        # ke fixed so that KTR = n / MTT crosses ke as MTT varies -> exercises both branches.
        ke = 0.8
        S, B = one_cmt(ke=ke)

        MTT = pt.scalar("MTT")
        out = transit_advan(
            S, B, times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        g = pt.grad(out.sum(), MTT)
        f = pytensor.function([MTT], g)
        for MTT_val in (n / ke - 0.5, n / ke, n / ke + 0.5):  # straddle alpha == 0
            with self.subTest(MTT=MTT_val):
                gv = np.asarray(f(MTT_val))
                self.assertTrue(np.all(np.isfinite(gv)))


class TestFlipFlop(unittest.TestCase):
    """The alpha = KTR - lambda < 0 (flip-flop) branch: transit slower than disposition.

    ``gammainc`` cannot take the negative argument the main branch would need, so the code
    switches to the all-positive convergent series (transit.py, _FLIPFLOP_TERMS). These
    tests exercise both a fully-negative single mode and a mixed system where one mode is
    positive and one negative (so pos and neg branches run simultaneously).
    """

    times = np.linspace(0.05, 30.0, 80)

    def test_flip_flop_one_cmt_matches_chain(self):
        # ke > KTR so alpha < 0 for the single mode.
        n, MTT, D = 5, 12.5, 100.0  # KTR = n/MTT = 0.4
        kt = n / MTT
        S, B = one_cmt(ke=1.2)  # lambda = 1.2 > KTR = 0.4  -> alpha = -0.8
        got = call_transit(
            S, B, self.times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        ref = chain_oracle(S, B, n, kt, self.times, bolus_D=D)
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 5e-4)

    def test_mixed_regime_two_cmt_matches_chain(self):
        # two_cmt lambdas ~= {0.894, 0.056}; kt = KTR = 0.7 -> alpha = {-0.194, +0.644}.
        S, B = two_cmt()
        n, kt, D = 5, 0.7, 100.0
        MTT = n / kt
        got = call_transit(
            S, B, self.times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        ref = chain_oracle(S, B, n, kt, self.times, bolus_D=D)
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 5e-4)

    def test_flip_flop_infusion_matches_chain(self):
        S, B = one_cmt(ke=1.0)  # lambda = 1.0 > KTR = 0.5 -> alpha < 0
        n, kt = 4, 0.5
        MTT = n / kt
        times = np.linspace(0.05, 40.0, 120)
        R, T = 10.0, 12.0
        infu_time = np.array([0.0, T])
        infu_rate = np.array([R, 0.0])
        got = call_transit(
            S, B, times, n, MTT, infu_time=infu_time, infu_rate=infu_rate,
        )
        ref = chain_oracle(S, B, n, kt, times, infusion=(infu_time, infu_rate))
        self.assertLess(np.max(np.abs(got - ref)) / np.max(np.abs(ref)), 5e-4)

    def test_gradient_finite_in_flip_flop(self):
        import pytensor
        import pytensor.tensor as pt

        n, D = 5.0, 100.0
        times = np.linspace(0.05, 30.0, 40)
        S, B = one_cmt(ke=1.2)  # lambda = 1.2
        MTT = pt.scalar("MTT")
        out = transit_advan(
            S, B, times, n, MTT,
            bolus_time=np.array([0.0]), bolus_amt=np.array([D]),
        )
        g = pt.grad(out.sum(), MTT)
        f = pytensor.function([MTT], g)
        # KTR = n/MTT crosses lambda=1.2 at MTT = n/1.2 ~= 4.17; straddle it.
        for MTT_val in (3.0, n / 1.2, 6.0):
            with self.subTest(MTT=MTT_val):
                gv = np.asarray(f(MTT_val))
                self.assertTrue(np.all(np.isfinite(gv)))


if __name__ == "__main__":
    unittest.main()
