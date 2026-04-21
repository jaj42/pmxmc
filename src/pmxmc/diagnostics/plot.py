from sys import argv

import arviz as az
import arviz_base as azb
import arviz_stats as azs
import matplotlib.pyplot as plt
import pymc as pm
from matplotlib.backends.backend_pdf import PdfPages

azb.rcParams["plot.max_subplots"] = 500


def sample(model, trace):
    with model:
        prior = pm.sample_prior_predictive(draws=1000)
        posterior = pm.sample_posterior_predictive(trace)
    idata = trace.copy()
    idata.extend(prior)
    idata.extend(posterior)
    return idata


def main():
    # rate, dv, covar, bio_map = read_dataset("./data.csv")
    # model = build_model(rate, dv, covar, bio_map)

    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)
    # idata = sample(model, idata)

    with PdfPages("output.pdf") as pdf:
        # azp.plot_ppc_dist(idata, group="prior")
        # pdf.savefig()
        # plt.close()

        # az.plot_ppc(idata, group="posterior")
        # pdf.savefig()
        # plt.close()

        # THETA
        az.plot_trace(idata, var_names=["^theta.*[^_]$"], filter_vars="regex")
        plt.suptitle("THETA Trace")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        az.plot_posterior(idata, var_names=["^theta.*[^_]$"], filter_vars="regex")
        plt.suptitle("THETA posterior")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # azp.plot_prior_posterior(idata, var_names=["^theta.*[^_]$"], filter_vars="regex")
        # plt.suptitle("THETA prior vs posterior")
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        ax.table(
            azs.summary(idata, var_names=["^theta.*[^_]$"], filter_vars="regex"),
            loc="center",
        )
        fig.suptitle("THETA")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # OMEGA
        az.plot_trace(idata, var_names=["^omega.*[^_]$"], filter_vars="regex")
        plt.suptitle("OMEGA Trace")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        az.plot_posterior(idata, var_names=["^omega.*[^_]$"], filter_vars="regex")
        plt.suptitle("OMEGA posterior")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # azp.plot_prior_posterior(idata, var_names=["^omega.*[^_]$"], filter_vars="regex")
        # plt.suptitle("OMEGA prior vs posterior")
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        ax.table(
            azs.summary(idata, var_names=["^omega.*[^_]$"], filter_vars="regex"),
            loc="center",
        )
        fig.suptitle("OMEGA")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # SIGMA
        az.plot_trace(idata, var_names=["^sigma.*[^_]$"], filter_vars="regex")
        plt.suptitle("SIGMA Trace")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ETA
        az.plot_trace(
            idata,
            var_names=["^eta.*[^_]$"],
            filter_vars="regex",
        )
        plt.suptitle("ETA Trace")
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    main()
