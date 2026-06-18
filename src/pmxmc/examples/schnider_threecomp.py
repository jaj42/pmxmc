import os
from importlib import resources
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import nutpie
import pymc as pm
import pytensor.tensor as pt
from pmxmc import assets
from pmxmc.advan import threecomp_advan as advan
from pmxmc.diagnostics import plot_idata, print_table
from pmxmc.io import read_nonmem_dataset
from pmxmc.utils import add_omegas, sample_predictive

jax.config.update("jax_enable_x64", True)


def build_model(ds) -> pm.Model:
    n_subj = ds["n_subj"]

    with pm.Model() as model:
        theta_V1 = pm.LogNormal("theta_V1", mu=np.log(4.5), sigma=0.5)
        theta_V2 = pm.LogNormal("theta_V2", mu=np.log(15), sigma=0.7)
        theta_V3 = pm.LogNormal("theta_V3", mu=np.log(250), sigma=1.0)
        theta_CL = pm.LogNormal("theta_CL", mu=np.log(1.5), sigma=0.5)
        theta_Q2 = pm.LogNormal("theta_Q2", mu=np.log(1.5), sigma=0.5)
        theta_Q3 = pm.LogNormal("theta_Q3", mu=np.log(0.8), sigma=0.5)

        sd_CL = pm.HalfNormal("sd_CL", sigma=0.5)
        sd_V1 = pm.HalfNormal("sd_V1", sigma=0.5)
        sd_V2 = pm.HalfNormal("sd_V2", sigma=0.5)
        sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)

        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        V3 = theta_V3
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)
        Q3 = theta_Q3

        C_preds = []
        for subj_id in ds["subj"]:
            idx = ds["subj_idx"][subj_id]

            meas_time = ds["dv"].xs(subj_id, level="ID").index
            rate = ds["rate"].xs(subj_id, level="ID")

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

            Cp = advan(
                meas_time.to_numpy(), rate.index.to_numpy(), rate.to_numpy(), params
            )
            C_preds.append(Cp)

        IPRED = pt.concatenate(C_preds)
        ERR = IPRED * sigma_prop
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=np.exp(ds["dv"]))

    return model


def main():
    with resources.open_text(assets, "eleveld.csv") as fd:
        dataset = read_nonmem_dataset(
            fd,
            sep=" ",
            filter="STDY==13",  # Schnider
        )

    model = build_model(dataset)
    add_omegas(model)
    with model:
        compiled = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(compiled)
        idata = sample_predictive(idata)
    idata.to_netcdf("idata.nc")

    # plot_model_criticism(idata, "Cp_obs", subject_per_obs,time_per_obs)
    plot_idata(idata, "pk.pdf", prior_predictive=True, posterior_predictive=True)
    print_table(idata)


if __name__ == "__main__":
    main()
