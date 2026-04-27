# pmxmc

**Pharmacometric Modeling with PyMC and JAX**

`pmxmc` is a Python package for Bayesian pharmacometric modeling, combining the power of [PyMC](https://www.pymc.io/) for probabilistic programming with [JAX](https://jax.readthedocs.io/) for efficient numerical computing. It provides tools for reading NONMEM datasets, fitting pharmacokinetic models, and performing MCMC sampling for population PK/PD modeling.

## Features

- **NONMEM Dataset I/O**: Read and process standard NONMEM-formatted CSV files with support for multiple dosing occasions
- **Solvers**: Solve pharmacokinetic differential equations using eigenvalue decomposition methods for efficiency
- **Vectorized Computation**: Leverage JAX's vmap/jit for fast batch processing of individual predictions
- **PyMC Integration**: Seamless integration with PyMC for full Bayesian inference via NUTS, Laplace approximation, or other samplers
- **Diagnostics**: Built-in diagnostic tools for model assessment (WIP)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pmxmc.git
cd pmxmc

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with uv
uv pip install -e .
```

## Dataset Format

`pmxmc` works with NONMEM-style tabular data. Required columns:

| Column | Description | Values |
|--------|-------------|---------|
| ID | Subject identifier | Integer |
| TIME | Time of observation/dose | Float |
| EVID | Event ID | 0=observation, 1=bolus, 2=infusion, 4=reset+new occasion |
| AMT | Dose amount | Float |
| RATE | Infusion rate | Float |
| DV | Dependent variable (observed concentration) | Float |

## Examples

See the `examples/` directory for complete workflow examples:

- `schnider_ode_nuts.py`: 3-compartment Schnider model with NUTS sampling
- `schnider_vectorized.py`: Vectorized implementation for efficiency
- `schnider_eig.py`: Eigenvalue-based solver

## License

ISC

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
