from sys import argv

import arviz as az


def main():
    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)
    # print(az.summary(idata, var_names=["^eta"], filter_vars="regex"))
    print(az.summary(idata, var_names=["^omega.*[^_]$"], filter_vars="regex"))
    print(az.summary(idata, var_names=["^theta.*[^_]$"], filter_vars="regex"))
    print(az.summary(idata, var_names=["^sigma.*[^_]$"], filter_vars="regex"))


if __name__ == "__main__":
    main()
