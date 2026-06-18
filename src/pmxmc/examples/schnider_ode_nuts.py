import os
from importlib import resources
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import nutpie
import pymc as pm
import pytensor.tensor as pt
from pmxmc import assets
from pmxmc.advan import ode_advan
from pmxmc.diagnostics import plot_idata, print_table
from pmxmc.io import read_nonmem_dataset
from pmxmc.utils import add_omegas, load_parameters

jax.config.update("jax_enable_x64", True)


def pk_ode(t, y, p):
    A1, A2, A3 = y
    k123 = p["k10"] + p["k12"] + p["k13"]
    ddt_A1 = -k123 * A1 + p["k21"] * A2 + p["k31"] * A3 + p["rate"](t)
    ddt_A2 = p["k12"] * A1 - p["k21"] * A2
    ddt_A3 = p["k13"] * A1 - p["k31"] * A3
    return jnp.array([ddt_A1, ddt_A2, ddt_A3])


def build_model(ds, prior_idata) -> pm.Model:
    n_subj = ds["n_subj"]

    with pm.Model() as model:
        p = load_parameters(prior_idata, sigma=True)

        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        # eta_V3 = pm.Normal("eta_V3", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)
        # eta_Q3 = pm.Normal("eta_Q3", mu=0, sigma=1, shape=n_subj)

        V1_i = p["theta_V1"] * pt.exp(p["sd_V1"] * eta_V1)
        V2_i = p["theta_V2"] * pt.exp(p["sd_V2"] * eta_V2)
        V3 = p["theta_V3"]
        CL_i = p["theta_CL"] * pt.exp(p["sd_CL"] * eta_CL)
        Q2_i = p["theta_Q2"] * pt.exp(p["sd_Q2"] * eta_Q2)
        Q3 = p["theta_Q3"]

        C_preds = []
        for subj_id in ds["subj"]:
            idx = ds["subj_idx"][subj_id]

            meas_time = ds["dv"].xs(subj_id, level="ID").index
            # rate = ds["rate"].xs(subj_id, level="ID")
            rate = ds["rate"]["RATE"].xs(subj_id, level="ID")

            V1 = V1_i[idx]
            V2 = V2_i[idx]
            CL = CL_i[idx]
            Q2 = Q2_i[idx]

            k10 = CL / V1
            k12 = Q2 / V1
            k21 = Q2 / V2
            k13 = Q3 / V1
            k31 = Q3 / V3
            params = {
                "k10": k10,
                "k12": k12, "k21": k21,
                "k13": k13, "k31": k31,
                "V1": V1, "V2": V2, "V3": V3,
            }  # fmt: skip

            Ap = ode_advan(
                meas_time.to_numpy(),
                rate.index.to_numpy(),
                rate.to_numpy(),
                pk_ode,
                params,
                y0=[0, 0, 0],
            )
            C_preds.append(Ap / V1)

        IPRED = pt.concatenate(C_preds)
        ERR = IPRED * p["sigma_prop"]
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=np.exp(ds["dv"]))

    return model


def main():
    with resources.open_text(assets, "eleveld.csv") as fd:
        dataset = read_nonmem_dataset(
            fd,
            sep=" ",
            filter="STDY==13",  # Schnider
        )
    prior_idata = az.from_netcdf("idata.nc")
    model = build_model(dataset, prior_idata)
    with model:
        add_omegas()
        compiled = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(compiled, chains=4)
    idata.to_netcdf("idata_ode_nuts.nc")

    # plot_model_criticism(idata, "Cp_obs", subject_per_obs,time_per_obs)
    plot_idata(idata, "pk.pdf")
    print_table(idata)


if __name__ == "__main__":
    main()
