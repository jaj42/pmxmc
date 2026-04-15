import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve
from pytensor import wrap_jax


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = int(np.clip(idx, 0, len(infu_rate) - 1))
    return float(infu_rate[idx])


def pk_ode(t, y, p):
    A1, A2, A3 = y
    ddt_A1 = (
        A2 * p["k21"]
        + A3 * p["k31"]
        - A1 * (p["k10"] + p["k12"] + p["k13"])
        + p["rate"]
    )
    ddt_A2 = A1 * p["k12"] - A2 * p["k21"]
    ddt_A3 = A1 * p["k13"] - A3 * p["k31"]
    return jnp.array([ddt_A1, ddt_A2, ddt_A3])


@wrap_jax
def ode_scan_solver(y0, meas_time, infu_time, infu_rate, params):
    _meas = np.asarray(meas_time)
    _itimes = np.asarray(infu_time)
    _irates = np.asarray(infu_rate)

    t_start = min(float(_meas[0]), float(_itimes[0]))

    _relevant = _itimes[(_itimes >= t_start) & (_itimes <= float(_meas[-1]))]
    all_times = np.unique(np.concatenate([_relevant, _meas]))

    seg_t0s = all_times[:-1]
    seg_t1s = all_times[1:]
    _rates = np.array([rate_at(t, _itimes, _irates) for t in seg_t0s])

    _itime_set = {float(t) for t in _itimes}
    _is_jump = np.array(
        [(i > 0) and (float(seg_t0s[i]) in _itime_set) for i in range(len(seg_t0s))]
    )
    _at_meas = np.isin(seg_t1s, _meas)

    ode_term = ODETerm(pk_ode)
    solver = diffrax.Tsit5()
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

    def step_fn(state, inputs):
        seg_t0, seg_t1, rate, jump = inputs
        sol = diffeqsolve(
            terms=ode_term,
            solver=solver,
            t0=seg_t0,
            t1=seg_t1,
            y0=state,
            dt0=None,
            stepsize_controller=stepsize_controller,
            max_steps=100_000,
            saveat=SaveAt(t1=True),
            args={**params, "rate": rate},
            made_jump=jump,
        )
        new_state = sol.ys[-1]
        return new_state, new_state

    _, all_states = jax.lax.scan(
        step_fn,
        jnp.asarray(y0, dtype=jnp.float64),
        (
            jnp.array(seg_t0s),
            jnp.array(seg_t1s),
            jnp.array(_rates),
            jnp.array(_is_jump),
        ),
    )

    # all_states: (n_segs, 3) — extract only segments ending at a measurement time
    return all_states[_at_meas, 0]  # A1 compartment
