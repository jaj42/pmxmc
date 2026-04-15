import pymc as pm


def add_omegas(model):
    with model:
        for name, var in model.named_vars.copy().items():
            if name.startswith("sd_"):
                pm.Deterministic(f"omega_{name[3:]}", var**2)
