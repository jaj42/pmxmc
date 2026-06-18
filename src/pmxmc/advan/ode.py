import diffrax as dfx
import jax.numpy as jnp
from pytensor import wrap_jax

from pmxmc.utils import build_rate_func


@wrap_jax
def ode_advan(meas_time, infu_time, infu_rate, pk_ode, params, y0):
    ode_term = dfx.ODETerm(pk_ode)
    solver = dfx.Tsit5()
    pid_controller = dfx.PIDController(rtol=1e-7, atol=1e-9)
    stepsize_controller = dfx.ClipStepSizeController(pid_controller, jump_ts=infu_time)
    max_steps = 100_000
    saveat = dfx.SaveAt(ts=meas_time)
    rate = build_rate_func(infu_time, infu_rate)
    tbeg = min(meas_time[0], infu_time[0])
    tend = meas_time[-1]

    sol = dfx.diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=tbeg,
        t1=tend,
        y0=jnp.asarray(y0),
        dt0=meas_time[1] - meas_time[0],
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=saveat,
        args={**params, "rate": rate},
        throw=False,
    )

    return sol.ys[:, 0]
