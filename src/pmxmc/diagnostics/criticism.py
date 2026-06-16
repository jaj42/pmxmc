"""Per-subject model criticism plots.

These diagnostics help detect whether individual-level (random) effects are
warranted.  The key signal is systematic structure correlated with subject
identity: consistent residual bias, poor predictive coverage, or elevated
LOO Pareto-k values for the same subjects.

``subject_per_obs`` must be aligned with the observation dimension of the
idata variable — i.e. filtered to the same rows the model used, not the full
dataset.  Example for a model that only uses DVID == 2::

    valid = ds["tv_covariates"]["DVID"]
    valid = valid[valid > 0]
    subject_per_obs = valid[valid == 2].index.get_level_values("ID").to_numpy()
    plot_model_criticism(idata, "E_obs", subject_per_obs)
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _group_by_subject(values, subject_per_obs):
    if len(values) != len(subject_per_obs):
        raise ValueError(
            f"subject_per_obs has {len(subject_per_obs)} entries but the "
            f"observation array has {len(values)}. Filter subject_per_obs to "
            "the same rows the model used (e.g. only DVID == 2 observations)."
        )
    subjects = np.unique(subject_per_obs)
    groups = [values[subject_per_obs == s] for s in subjects]
    return subjects, groups


def plot_residuals_by_subject(idata, obs_var, subject_per_obs, ax=None):
    """Boxplot of (observed - posterior-predictive mean) per subject.

    Systematic bias in specific subjects — some boxes entirely above or below
    zero — is the primary signal that individual variation is not captured.
    """
    subject_per_obs = np.asarray(subject_per_obs)
    ppc_mean = idata.posterior_predictive[obs_var].mean(("chain", "draw")).values
    obs = idata.observed_data[obs_var].values
    resid = obs - ppc_mean
    subjects, groups = _group_by_subject(resid, subject_per_obs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    else:
        fig = ax.get_figure()

    ax.boxplot(groups, tick_labels=subjects)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Observed − Predicted")
    ax.set_title("Per-subject residuals")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig, ax


def plot_coverage_by_subject(idata, obs_var, subject_per_obs, prob=0.89, ax=None):
    """Bar chart of per-subject posterior predictive coverage.

    Bars far below the target line indicate subjects the model cannot
    accommodate at the population level.
    """
    subject_per_obs = np.asarray(subject_per_obs)
    ppc = idata.posterior_predictive[obs_var]
    q_lo, q_hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    lower = ppc.quantile(q_lo, ("chain", "draw")).values
    upper = ppc.quantile(q_hi, ("chain", "draw")).values
    obs = idata.observed_data[obs_var].values
    within = (obs >= lower) & (obs <= upper)
    subjects = np.unique(subject_per_obs)
    coverage = np.array([within[subject_per_obs == s].mean() for s in subjects])

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    else:
        fig = ax.get_figure()

    colors = ["steelblue" if c >= prob * 0.85 else "tomato" for c in coverage]
    ax.bar(range(len(subjects)), coverage, color=colors)
    ax.axhline(
        prob,
        color="k",
        linestyle="--",
        linewidth=0.8,
        label=f"Target {int(prob * 100)}%",
    )
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Coverage fraction")
    ax.set_title(f"Per-subject {int(prob * 100)}% predictive coverage")
    ax.legend()
    return fig, ax


def plot_pareto_k_by_subject(idata, subject_per_obs, ax=None):
    """Bar chart of mean LOO Pareto-k per subject.

    Clusters of high-k observations within the same subjects indicate that
    those subjects are outliers the model cannot pool with the rest of the
    population — a strong signal for hierarchical priors.

    Requires log_likelihood group in idata (pm.compute_log_likelihood).
    """
    subject_per_obs = np.asarray(subject_per_obs)
    loo = az.loo(idata, pointwise=True)
    pk = loo.pareto_k.values
    subjects = np.unique(subject_per_obs)
    mean_pk = np.array([pk[subject_per_obs == s].mean() for s in subjects])

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.5), 4))
    else:
        fig = ax.get_figure()

    colors = [
        "tomato" if v > 0.7 else "orange" if v > 0.5 else "steelblue" for v in mean_pk
    ]
    ax.bar(range(len(subjects)), mean_pk, color=colors)
    ax.axhline(
        0.7, color="tomato", linestyle="--", linewidth=0.8, label="Poor  k > 0.7"
    )
    ax.axhline(
        0.5, color="orange", linestyle="--", linewidth=0.8, label="Marginal  k > 0.5"
    )
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Mean Pareto-k")
    ax.set_title("LOO Pareto-k by subject")
    ax.legend()
    return fig, ax


def plot_ppc_by_subject(
    idata, obs_var, subject_per_obs, time_per_obs, prob=0.89, ncols=4
):
    """Grid of per-subject posterior predictive plots.

    Each panel shows the posterior predictive mean + credible band and the
    observed data points for one subject.

    Parameters
    ----------
    time_per_obs : array-like
        Time value for each observation row, same length as the obs dimension.
    ncols : int
        Number of columns in the grid.

    Returns
    -------
    fig : matplotlib Figure
    """
    subject_per_obs = np.asarray(subject_per_obs)
    time_per_obs = np.asarray(time_per_obs)

    ppc = idata.posterior_predictive[obs_var]
    q_lo, q_hi = (1 - prob) / 2, 1 - (1 - prob) / 2
    ppc_mean = ppc.mean(("chain", "draw")).values
    ppc_lo = ppc.quantile(q_lo, ("chain", "draw")).values
    ppc_hi = ppc.quantile(q_hi, ("chain", "draw")).values
    obs = idata.observed_data[obs_var].values

    subjects = np.unique(subject_per_obs)
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False
    )

    for i, subj in enumerate(subjects):
        ax = axes[i // ncols][i % ncols]
        mask = subject_per_obs == subj
        order = np.argsort(time_per_obs[mask])
        t = time_per_obs[mask][order]

        ax.fill_between(
            t,
            ppc_lo[mask][order],
            ppc_hi[mask][order],
            alpha=0.3,
            color="steelblue",
        )
        ax.plot(t, ppc_mean[mask][order], color="steelblue", linewidth=1)
        ax.scatter(t, obs[mask][order], color="k", s=8, zorder=5)
        ax.set_title(str(subj), fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(len(subjects), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.supxlabel("Time", fontsize=9)
    fig.supylabel(obs_var, fontsize=9)
    fig.suptitle(f"Posterior predictive — {int(prob * 100)}% CI", fontsize=10)
    plt.tight_layout()
    return fig


def plot_model_criticism(
    idata,
    obs_var,
    subject_per_obs,
    time_per_obs=None,
    output="criticism.pdf",
    prob=0.89,
):
    """Save residual, coverage, Pareto-k, and (optionally) per-subject PPC plots to a PDF.

    Parameters
    ----------
    obs_var : str
        Name of the observed variable in idata.posterior_predictive.
    subject_per_obs : array-like
        Subject ID for each observation row, same length as the obs dimension.
    time_per_obs : array-like, optional
        Time value per observation. When provided, per-subject PPC panels are included.
    """
    with PdfPages(output) as pdf:
        fig, _ = plot_residuals_by_subject(idata, obs_var, subject_per_obs)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        fig, _ = plot_coverage_by_subject(idata, obs_var, subject_per_obs, prob=prob)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        try:
            fig, _ = plot_pareto_k_by_subject(idata, subject_per_obs)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        except Exception as exc:
            print(f"Pareto-k skipped (need log_likelihood in idata): {exc}")

        if time_per_obs is not None:
            fig = plot_ppc_by_subject(
                idata, obs_var, subject_per_obs, time_per_obs, prob=prob
            )
            pdf.savefig(fig)
            plt.close()
