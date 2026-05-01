import os
from multiprocessing import cpu_count
from importlib import resources

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

from pmxmc import assets
import arviz as az
import jax
import numpy as np
import nutpie
import pymc as pm
import pytensor.tensor as pt
from pmxmc.advan import threecomp_advan as advan
from pmxmc.io import read_nonmem_dataset
from pmxmc.utils import add_omegas

jax.config.update("jax_enable_x64", True)


def build_model(rates, dv, covar, bio_map) -> pm.Model:
    unique_occ_ids = dv.index.get_level_values("ID").unique()
    unique_bio_ids = sorted(bio_map.unique())
    n_subj = len(unique_bio_ids)
    bio_idx_map = {int(bid): i for i, bid in enumerate(unique_bio_ids)}

    DV = dv.to_numpy()

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
        # sd_V3 = pm.HalfNormal("sd_V3", sigma=0.5)
        sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)
        # sd_Q3 = pm.HalfNormal("sd_Q3", sigma=0.5)

        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        # eta_V3 = pm.Normal("eta_V3", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)
        # eta_Q3 = pm.Normal("eta_Q3", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        V3 = theta_V3
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)
        Q3 = theta_Q3

        C_preds = []
        for occ_id in unique_occ_ids:
            bio_id = int(bio_map.loc[occ_id])
            bio_idx = bio_idx_map[bio_id]

            meas_time = dv.xs(occ_id, level="ID").index.to_numpy().flatten()
            patient_rate = rates.xs(occ_id, level="ID")
            infu_time = patient_rate.index.to_numpy().flatten()
            infu_rate = patient_rate.to_numpy().flatten()

            V1 = V1_i[bio_idx]
            V2 = V2_i[bio_idx]
            # V3 = V3_i[bio_idx]
            CL = CL_i[bio_idx]
            Q2 = Q2_i[bio_idx]
            # Q3 = Q3_i[bio_idx]

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

            Cp = advan(meas_time, infu_time, infu_rate, params)
            C_preds.append(Cp)

        IPRED = pt.concatenate(C_preds)
        ERR = IPRED * sigma_prop
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=DV)

    return model


def main():
    with resources.open_text(assets, "schnider.csv") as fd:
        rate, dv, covar, bio_map, _bolus = read_nonmem_dataset(fd, sep=",", dv_col="CP")

    model = build_model(rate, dv, covar, bio_map)
    add_omegas(model)
    with model:
        compiled = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(compiled)
        # idata = fit_dadvi(gradient_backend="jax")
    az.to_netcdf(idata, "idata.nc")


if __name__ == "__main__":
    main()
