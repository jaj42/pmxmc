import os
from importlib import resources
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc_extras import inference

from pmxmc import assets
from pmxmc.advan.eigh import threecomp_advan_vec
from pmxmc.diagnostics import print_table
from pmxmc.io import read_nonmem_dataset_padded
from pmxmc.utils import add_omegas

jax.config.update("jax_enable_x64", True)
pytensor.config.floatX = "float64"


def build_model(ds) -> pm.Model:
    n_subj = len(set(ds["occid_map"].values()))
    with pm.Model() as model:
        theta_V1 = pm.LogNormal("theta_V1", mu=np.log(4.5), sigma=0.5)
        theta_V2 = pm.LogNormal("theta_V2", mu=np.log(15), sigma=0.7)
        theta_V3 = pm.LogNormal("theta_V3", mu=np.log(250), sigma=1.0)
        theta_CL = pm.LogNormal("theta_CL", mu=np.log(1.5), sigma=0.5)
        theta_Q2 = pm.LogNormal("theta_Q2", mu=np.log(1.5), sigma=0.5)
        theta_Q3 = pm.LogNormal("theta_Q3", mu=np.log(0.8), sigma=0.5)

        sd_V1 = pm.HalfNormal("sd_V1", sigma=0.5)
        sd_V2 = pm.HalfNormal("sd_V2", sigma=0.5)
        sd_CL = pm.HalfNormal("sd_CL", sigma=0.5)
        sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)

        # sigma_add = pm.HalfNormal("sigma_add", sigma=5)
        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)

        V1 = V1_i[ds["id"]]
        V2 = V2_i[ds["id"]]
        V3 = theta_V3
        CL = CL_i[ds["id"]]
        Q2 = Q2_i[ds["id"]]
        Q3 = theta_Q3

        all_Cp = threecomp_advan_vec(
            ds["dt"],
            ds["rate"],
            ds["bolus"],
            ds["meas_idx"],
            CL / V1,  # k10
            Q2 / V1,  # k12
            Q2 / V2,  # k21
            Q3 / V1,  # k13
            Q3 / V3,  # k31
            V1,
        )
        IPRED = pt.flatten(all_Cp)[ds["valid_idx"]]
        ERR = IPRED * sigma_prop
        # ERR = pt.sqrt((IPRED * sigma_prop )**2 + sigma_add**2)
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=ds["dv"])

    return model


def main():
    with resources.open_text(assets, "schnider.csv") as fd:
        dataset = read_nonmem_dataset_padded(fd, sep=",", dv_col="CP")
    model = build_model(dataset)
    add_omegas(model)
    with model:
        # compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
        # idata = nutpie.sample(compiled)
        idata = inference.fit_laplace(model=model, gradient_backend="jax")
    # az.to_netcdf(idata, "idata.nc")
    print_table(idata)


if __name__ == "__main__":
    main()
