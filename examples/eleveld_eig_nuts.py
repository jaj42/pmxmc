import os
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import nutpie
from pmxmc.advan.eigh import eigendecomposition
from pmxmc.io import read_nonmem_dataset
from pmxmc.utils import add_omegas, rate_at
from pymc_extras import inference
from pytensor import wrap_jax

jax.config.update("jax_enable_x64", True)


@wrap_jax
def threecomp_advan_vmap(all_dts, all_rates, all_boluses, all_meas_indices, params):
    """
    Batched 3-compartment solver via vmap over occasions.

    all_dts          : (n_occ, max_steps)  zero-padded time deltas
    all_rates        : (n_occ, max_steps)  zero-padded infusion rates
    all_boluses      : (n_occ, max_steps)  bolus amounts at interval start (0 if none)
    all_meas_indices : (n_occ, max_meas)   zero-padded indices into the
                       state array at which concentrations are read out
    params           : dict; per-occasion keys (k10,k12,k21,k13,V1,V2)
                       are 1-D arrays of length n_occ; scalar keys
                       (k31, V3) are broadcast over all occasions.

    Returns (n_occ, max_meas) array; first n_meas[i] entries of row i
    are valid, the rest are zero-padded artefacts.
    """

    def single_occasion(dts, rates, boluses, meas_indices, p):
        a2 = p["k12"] * jnp.sqrt(p["V1"] / p["V2"])
        a3 = p["k13"] * jnp.sqrt(p["V1"] / p["V3"])
        k123 = p["k10"] + p["k12"] + p["k13"]
        S = jnp.array(
            [
                [-k123,      a2,       a3  ],
                [  a2,  -p["k21"],    0.0  ],
                [  a3,      0.0,  -p["k31"]],
            ]
        )  # fmt: skip
        lambdas, p_coef = eigendecomposition(S, p["V1"], 0)
        state0 = jnp.zeros(3, dtype=jnp.float64)

        def step_fn(A, inputs):
            dt, rate, bolus = inputs
            A = A + bolus * lambdas * p_coef   # instantaneous bolus at interval start
            decay = jnp.exp(-lambdas * dt)
            A_new = A * decay + p_coef * rate * (1 - decay)
            return A_new, A_new

        _, all_states = jax.lax.scan(step_fn, state0, (dts, rates, boluses))
        all_states_with_init = jnp.concatenate([state0[None, :], all_states], axis=0)
        return jnp.sum(all_states_with_init[meas_indices], axis=-1)

    per_occ = {
        "k10": 0,
        "k12": 0,
        "k21": 0,
        "k13": 0,
        "V1": 0,
        "V2": 0,
        # "k31": None,
        # "V3": None,
        "k31": 0,
        "V3": 0,
    }
    return jax.vmap(single_occasion, in_axes=(0, 0, 0, 0, per_occ))(
        all_dts, all_rates, all_boluses, all_meas_indices, params
    )


def precompute_occasion_data(rates, dv, bolus, bio_map, bio_idx_map, unique_occ_ids):
    """Precompute padded numpy arrays for all occasions."""
    bio_indices, dts_list, rates_list, boluses_list, meas_idx_list, n_meas_list = (
        [], [], [], [], [], []
    )

    for occ_id in unique_occ_ids:
        bio_indices.append(bio_idx_map[int(bio_map.loc[occ_id])])

        meas_time = dv.xs(occ_id, level="ID").index.to_numpy().flatten()
        patient_rate = rates.xs(occ_id, level="ID")
        infu_time = patient_rate.index.to_numpy().flatten()
        infu_rate = patient_rate.to_numpy().flatten()

        try:
            patient_bolus = bolus.xs(occ_id, level="ID")
            bolus_time_i = patient_bolus.index.to_numpy().flatten()
            bolus_amt_i = patient_bolus["AMT"].to_numpy().flatten()
        except KeyError:
            bolus_time_i = np.array([])
            bolus_amt_i = np.array([])

        tbeg = min(meas_time[0], infu_time[0])
        if len(bolus_time_i):
            tbeg = min(tbeg, bolus_time_i[0])
        tend = meas_time[-1]

        rel_itimes = infu_time[(infu_time >= tbeg) & (infu_time <= tend)]
        rel_btimes = bolus_time_i[(bolus_time_i >= tbeg) & (bolus_time_i <= tend)]
        all_times = np.unique(np.concatenate([rel_itimes, rel_btimes, meas_time]))
        dts_i = np.diff(all_times)
        rates_i = np.array([rate_at(t, infu_time, infu_rate) for t in all_times[:-1]])

        boluses_i = np.zeros(len(dts_i))
        for bt, ba in zip(bolus_time_i, bolus_amt_i):
            idxs = np.where(all_times[:-1] == bt)[0]
            if len(idxs):
                boluses_i[idxs[0]] += ba

        meas_idx_i = np.where(np.isin(all_times, meas_time))[0]

        dts_list.append(dts_i)
        rates_list.append(rates_i)
        boluses_list.append(boluses_i)
        meas_idx_list.append(meas_idx_i)
        n_meas_list.append(len(meas_idx_i))

    n_occ = len(unique_occ_ids)
    max_steps = max(len(d) for d in dts_list)
    max_meas = max(len(m) for m in meas_idx_list)

    dts_padded     = np.zeros((n_occ, max_steps), dtype=np.float64)
    rates_padded   = np.zeros((n_occ, max_steps), dtype=np.float64)
    boluses_padded = np.zeros((n_occ, max_steps), dtype=np.float64)
    meas_idx_padded = np.zeros((n_occ, max_meas), dtype=np.int32)
    for i, (d, r, b, mi) in enumerate(zip(dts_list, rates_list, boluses_list, meas_idx_list)):
        dts_padded[i, : len(d)] = d
        rates_padded[i, : len(r)] = r
        boluses_padded[i, : len(b)] = b
        meas_idx_padded[i, : len(mi)] = mi

    valid_flat_indices = np.concatenate(
        [i * max_meas + np.arange(n) for i, n in enumerate(n_meas_list)]
    )

    return (
        np.array(bio_indices),
        dts_padded,
        rates_padded,
        boluses_padded,
        meas_idx_padded,
        valid_flat_indices,
    )


def build_model(rates, dv, covar, bio_map, bolus) -> pm.Model:
    unique_occ_ids = dv.index.get_level_values("ID").unique()
    unique_bio_ids = sorted(bio_map.unique())
    n_subj = len(unique_bio_ids)
    bio_idx_map = {int(bid): i for i, bid in enumerate(unique_bio_ids)}

    DV = dv.to_numpy()

    bio_indices, dts_padded, rates_padded, boluses_padded, meas_idx_padded, valid_flat_indices = (
        precompute_occasion_data(rates, dv, bolus, bio_map, bio_idx_map, unique_occ_ids)
    )

    with pm.Model() as model:
        theta_V1 = pm.LogNormal("theta_V1", mu=np.log(4.5), sigma=0.5)
        theta_V2 = pm.LogNormal("theta_V2", mu=np.log(15), sigma=0.7)
        theta_V3 = pm.LogNormal("theta_V3", mu=np.log(250), sigma=1.0)
        theta_CL = pm.LogNormal("theta_CL", mu=np.log(1.5), sigma=0.5)
        theta_Q2 = pm.LogNormal("theta_Q2", mu=np.log(1.5), sigma=0.5)
        theta_Q3 = pm.LogNormal("theta_Q3", mu=np.log(0.8), sigma=0.5)

        sd_V1 = pm.HalfNormal("sd_V1", sigma=0.5)
        sd_V2 = pm.HalfNormal("sd_V2", sigma=0.5)
        sd_V3 = pm.HalfNormal("sd_V3", sigma=0.5)
        sd_CL = pm.HalfNormal("sd_CL", sigma=0.5)
        sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)
        sd_Q3 = pm.HalfNormal("sd_Q3", sigma=0.5)

        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        eta_V3 = pm.Normal("eta_V3", mu=0, sigma=1, shape=n_subj)
        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)
        eta_Q3 = pm.Normal("eta_Q3", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        V3_i = theta_V3 * pt.exp(sd_V3 * eta_V3)
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)
        Q3_i = theta_Q3 * pt.exp(sd_Q3 * eta_Q3)

        V1 = V1_i[bio_indices]
        V2 = V2_i[bio_indices]
        V3 = V3_i[bio_indices]
        # V3 = theta_V3
        CL = CL_i[bio_indices]
        Q2 = Q2_i[bio_indices]
        Q3 = Q3_i[bio_indices]
        # Q3 = theta_Q3

        params = {
            "k10": CL / V1,
            "k12": Q2 / V1,
            "k21": Q2 / V2,
            "k13": Q3 / V1,
            "k31": Q3 / V3,
            "V1":  V1,
            "V2":  V2,
            "V3":  V3,
        }  # fmt: skip

        all_Cp = threecomp_advan_vmap(dts_padded, rates_padded, boluses_padded, meas_idx_padded, params)
        IPRED = pt.flatten(all_Cp)[valid_flat_indices]
        ERR = IPRED * sigma_prop
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=DV)

    return model


def main():
    rate, dv, covar, bio_map, bolus = read_nonmem_dataset("./eleveld.csv")
    # rate, dv, covar, bio_map, bolus = read_nonmem_dataset("./schnider.csv",sep=',',dv_col='CP')
    model = build_model(rate, dv, covar, bio_map, bolus)
    add_omegas(model)
    with model:
        compiled = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(compiled)
        # idata = inference.fit_laplace(model=model, gradient_backend="jax")
    az.to_netcdf(idata, "idata.nc")


if __name__ == "__main__":
    main()
