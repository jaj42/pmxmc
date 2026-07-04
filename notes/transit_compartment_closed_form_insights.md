# Closed-form transit-compartment absorption: derivation & insights

Reference for `src/pmxmc/advan/transit.py` (`transit_advan`). It models delayed absorption
as a Gamma-distributed transit input convolved against the linear-PK eigensystem, giving an
exact, ODE-free, differentiable response — a sum of regularised incomplete-Gamma terms.

Related: Savic et al. 2007 (DOI `s10928-007-9066-0`), the pharmacometrics skill's
`structural-models.md` (transit-compartment section), and icomo's `erlang_kernel`
(`/home/jaj/devel/upstream/icomo/icomo/comp_model.py`), which realises the same delay by
the linear-chain trick integrated numerically with diffrax — our test oracle.

---

## 1. Model

The PK system is linear: `x'(t) = S·x(t) + B·r(t)`, with `r(t)` the drug input rate into
the system (via the 1-D input vector `B`, shape `(n_cmt,)`, e.g. `[1/V1, 0, 0]`).

Instead of a first-order or bolus input, absorption is *delayed* by a transit process whose
arrival-time density is Gamma with shape `n` and rate `KTR`:

```
g(t) = KTR^n · t^(n-1) · e^(-KTR·t) / Γ(n)          (Gamma(shape=n, rate=KTR) pdf)
```

A dose `D` therefore drives the system with input rate `r(t) = D·g(t)` (bolus), or, for a
piecewise-constant infusion, the delayed version of each rate step.

### Parameterisation (KTR = n / MTT), and why it differs from Savic

The mean of `Gamma(n, KTR)` is `n / KTR`. We *define* the mean transit time `MTT` to be
exactly that mean, hence:

```
KTR = n / MTT          (this codebase)
```

Savic uses `KTR = (n+1) / MTT` because the classic transit model has `n` transit
compartments **plus a separate absorption (ka) compartment** — `n+1` stages in total, mean
`(n+1)/KTR`. Our closed form has **no separate absorption compartment**: the transit output
feeds the PK system directly, so the delay is `Gamma(shape=n)` and its mean is `n/KTR`.
Using `(n+1)/MTT` here would make `MTT` overstate the true mean by a factor `(n+1)/n`
(~9% at n=10, ~2% at n=50). This was found and fixed by numerical comparison against the
diffrax chain oracle; see `tests/test_transit.py::TestTransitConvention`.

---

## 2. Per-mode convolution (the core identity)

Diagonalise `S = V·diag(-λ)·V⁻¹` with positive decay rates `λ_i` (real for mammillary PK).
Compartment `j`'s response is a sum over modes of the input convolved with `e^(-λ_i·t)`.
The essential integral, for one mode with `α = KTR − λ`:

```
∫₀ᵗ e^(-λ(t-s)) · g(s) ds
    = e^(-λt) · KTR^n/Γ(n) · ∫₀ᵗ s^(n-1) e^(-α s) ds
    = e^(-λt) · KTR^n/Γ(n) · α^(-n) · γ(n, α t)
    = e^(-λt) · (KTR/α)^n · P(n, α t)                    ( P = regularised lower inc. gamma )
```

So define, per mode:

```
G_i(t)      = (KTR/α_i)^n · P(n, α_i·t)
bolus_K_i   = e^(-λ_i·t) · G_i(t)          # impulse (bolus) response kernel   [phi_i]
step_K_i    = P(n, KTR·t) − bolus_K_i      # unit rate-step (infusion) response [S_i]
```

- **Boluses** (impulse of area `D`) enter with residue weight `w[j,i] = V_ji·(V⁻¹B)_i`
  (in code `w = coefs · λ`). Contribution: `Σ_i w[j,i]·D·bolus_K_i`.
- **Infusions** (piecewise-constant rate) are decomposed into rate *steps* `dR_k` at times
  `τ_k` (`dR[0]=rate[0]`, `dR[k]=rate[k]−rate[k-1]`); each step's response is `step_K` with
  gain `coefs[j,i]`. Contribution: `Σ_i coefs[j,i]·(Σ_k dR_k·step_K_i(t−τ_k))`.
- **Superposition** over doses/steps (all times clamped so `t−t_dose ≥ 0`) assembles the
  full multi-dose response. Verified to machine precision (~1e-12) against the chain oracle
  for integer `n`.

---

## 3. The α → 0 limit branch

When `α = KTR − λ → 0`, both `(KTR/α)^n` and `P(n, α t)` blow up / vanish, but their product
has a finite limit. Since `P(n, x) = γ(n, x)/Γ(n) → x^n/Γ(n+1)` as `x → 0`:

```
G_i(t) → (KTR·t)^n / Γ(n+1)          (α → 0)
```

The code (`transit.py`) switches to this analytic limit when `|α| < _ALPHA_EPS` via the
double-`where` trick (`a_safe = where(safe, a, 1.0)`) so the unused main branch stays finite
and gradients propagate cleanly. The exactly-`α=0` case (`KTR = λ`) is covered by
`tests/test_transit.py::TestAlphaLimit`.

---

## 4. α < 0 (flip-flop / slow transit): the series branch

`jax.scipy.special.gammainc` only accepts a **non-negative** argument, so the main
`(KTR/α)^n P(n, αt)` form cannot be evaluated when `α < 0` — i.e. when some mode has
`λ_i > KTR` (transit slower than that disposition mode, the flip-flop regime). For those
modes we compute the integral directly. With `β = −α > 0` the integral is real and positive:

```
∫₀ᵗ s^(n-1) e^(β s) ds = tⁿ · Σ_{k≥0} (β t)^k / (k! · (n+k))         (all terms > 0)
```

so, with the decay folded back in,

```
bolus_K = e^(-λt) · (KTR·t)ⁿ / Γ(n) · Σ_{k≥0} (β t)^k / (k! (n+k))
```

This series is entire (converges for all `βt`) and, crucially, has **no sign cancellation**
(unlike the `γ(n, z)` series at negative `z`, whose `z^n` would also go complex for
non-integer `n`). The `β → 0` limit recovers §3: the `k=0` term gives `1/n`, so
`bolus_K → e^(-λt)(KTR·t)ⁿ/Γ(n+1)`.

**Numerical stability.** The bare sum grows like `e^(βt)`, which overflows float64 for
`βt ≳ 700`, yet the physical response `~ e^(-λt)·e^(βt) = e^(-KTR·t)` stays bounded. So we
fold `e^(-λt)` into each term and work in log-space: the summand
`e^(-λt) (βt)^k / k!` peaks near `k = βt` at `~e^(-KTR·t)`, always representable.

**Implementation** (`transit.py`): `_FLIPFLOP_TERMS` (=200) fixed series terms summed in
log-space; per-mode masks (`pos` / `neg`) select this branch only where `α < −ε`, with all
branches evaluated on "safe" arguments so gradients stay finite through `jnp.where`. Accurate
for `βt` up to roughly `_FLIPFLOP_TERMS / 2`, far beyond where the response has decayed to
noise for realistic PK. Validated (including a mixed pos/neg two-compartment system and the
infusion path) by `tests/test_transit.py::TestFlipFlop`.

The whole series is wrapped in `lax.cond(any_neg, …)` (`any_neg = jnp.any(neg)`, constant
across `dt`). Since the mask cost is data-independent and most PK models satisfy
`KTR > max λ`, this skips the series entirely at runtime — forward *and* gradient — in the
common no-flip-flop case, recovering ~all of its overhead (benchmarked: a guarded no-flip
call returns to the series-absent baseline; the cost is paid only when a mode is actually in
flip-flop).

---

## 5. Why closed form (vs. icomo's linear chain)

- icomo integrates a chain of `k` sub-compartments (each rate `k·rate`) with diffrax —
  general, works for nonlinear systems, but `k` (the shape) must be an **integer** and every
  evaluation is an ODE solve.
- Our closed form is exact and ODE-free, `O(modes × dose-events)`, differentiable in a
  **continuous** `n` — delivering Savic's "estimate a non-integer `n`" goal, which the
  integer-shape chain cannot. The trade-off is that it is specific to **linear** PK with
  **real** eigenvalues (all signs of `α = KTR − λ` are supported, incl. flip-flop; see §4).
