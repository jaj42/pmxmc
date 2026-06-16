import re
from sys import argv

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

az.rcParams["plot.max_subplots"] = 500

patterns = {
    "omega": "^omega.*[^_]$",
    "theta": "^theta.*[^_]$",
    "sigma": "^sigma.*[^_]$",
    "eta": "^eta.*[^_]$",
}


def _available_parameters(idata):
    params = {k: False for k in patterns.keys()}
    for param_type, param_pattern in patterns.items():
        if any(re.search(param_pattern, v) for v in idata.posterior.data_vars):
            params[param_type] = True
    return params


def plot_param_type(idata, name, pattern, device):
    az.plot_trace_dist(idata, var_names=[pattern], filter_vars="regex")
    plt.suptitle(f"{name} Trace")
    plt.tight_layout()
    device.savefig()
    plt.close()

    az.plot_dist(idata, var_names=[pattern], filter_vars="regex")
    plt.suptitle(f"{name} posterior")
    plt.tight_layout()
    device.savefig()
    plt.close()

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(
        az.summary(idata, var_names=[pattern], filter_vars="regex"),
        loc="center",
    )
    fig.suptitle(name)
    plt.tight_layout()
    device.savefig(fig)
    plt.close()


def plot_idata(
    idata,
    output="output.pdf",
    model=None,
    prior_predictive=False,
    posterior_predictive=False,
):
    params = _available_parameters(idata)

    with PdfPages(output) as pdf:
        if prior_predictive:
            az.plot_ppc_dist(idata, group="prior_predictive")
            plt.suptitle("Prior Predictive")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if posterior_predictive:
            az.plot_ppc_dist(idata, group="posterior_predictive")
            plt.suptitle("Posterior Predictive")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # THETA
        if params["theta"]:
            plot_param_type(idata, "THETA", patterns["theta"], pdf)

        # OMEGA
        if params["omega"]:
            plot_param_type(idata, "OMEGA", patterns["omega"], pdf)

        # SIGMA
        if params["sigma"]:
            plot_param_type(idata, "SIGMA", patterns["sigma"], pdf)

        # ETA
        if params["eta"]:
            az.plot_trace_dist(
                idata,
                var_names=[patterns["eta"]],
                filter_vars="regex",
            )
            plt.suptitle("ETA Trace")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def main():
    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)

    plot_idata(idata)


if __name__ == "__main__":
    main()
