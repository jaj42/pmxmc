import pymc as pm
import numpy as np
import jax.numpy as jnp


def add_omegas(model):
    with model:
        for name, var in model.named_vars.copy().items():
            if name.startswith("sd_"):
                pm.Deterministic(f"omega_{name[3:]}", var**2)


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = np.clip(idx, 0, len(infu_rate) - 1)
    return infu_rate[idx]


def build_rate_func(infu_time, infu_rate):
    def worker(t):
        _infu_time = jnp.asarray(infu_time)
        _infu_rate = jnp.asarray(infu_rate)
        idx = jnp.searchsorted(_infu_time[1:], t, side="right")
        idx = jnp.clip(idx, 0, len(infu_rate) - 1)
        return jnp.where(t < _infu_time[0], 0.0, _infu_rate[idx])

    return worker
