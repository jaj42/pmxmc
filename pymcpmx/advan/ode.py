import diffrax
import jax.numpy as jnp
import numpy as np
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
def ode_wrapper(y0, meas_time, infu_time, infu_rate, params):
    _meas = np.asarray(meas_time)
    _itimes = np.asarray(infu_time)
    _irates = np.asarray(infu_rate)

    t_start = min(float(_meas[0]), float(_itimes[0]))
    t_end = float(_meas[-1])

    # Time grid: union of measurement times and infusion breakpoints in [t_start, t_end].
    # Splitting here ensures the ODE is restarted at every infusion discontinuity.
    _relevant = _itimes[(_itimes >= t_start) & (_itimes <= t_end)]
    all_times = np.unique(np.concatenate([_relevant, _meas]))

    ode_term = ODETerm(pk_ode)
    solver = diffrax.Tsit5()
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

    _itime_set = set(float(t) for t in _itimes)
    _meas_set = set(float(t) for t in _meas)

    state = jnp.asarray(y0, dtype=jnp.float64)
    solver_state = None
    controller_state = None
    results = []

    for i in range(len(all_times) - 1):
        seg_t0 = float(all_times[i])
        seg_t1 = float(all_times[i + 1])

        # Constant infusion rate for this segment
        rate = rate_at(seg_t0, _itimes, _irates)

        # Signal to the solver that the RHS is discontinuous at seg_t0 so it
        # resets its step-size estimate rather than stepping over the jump.
        is_jump = (i > 0) and (seg_t0 in _itime_set)

        sol = diffeqsolve(
            terms=ode_term,
            solver=solver,
            t0=seg_t0,
            t1=seg_t1,
            y0=state,
            dt0=None,
            stepsize_controller=stepsize_controller,
            max_steps=100_000,
            saveat=SaveAt(t1=True, solver_state=True, controller_state=True),
            args={**params, "rate": rate},
            made_jump=is_jump,
            solver_state=solver_state,
            controller_state=controller_state,
        )

        # sol.ys has shape (1, 3) with SaveAt(t1=True)
        state = sol.ys[-1]
        solver_state = sol.solver_state
        controller_state = sol.controller_state

        if seg_t1 in _meas_set:
            results.append(state[0])  # A1 compartment

    return jnp.stack(results)


# A1 = ode_wrapper(
#     y0=[0, 0, 0],
#     meas_time=meas_time,
#     infu_time=infu_time,
#     infu_rate=infu_rate,
#     params=pk_params,
# )
# C_preds.append(A1 / V1_i[bio_idx])
