# Scalable Data Assimilation via Message Passing

A message passing algorithm, implemented in JAX, for approximately computing the posterior marginal distribution of a Gaussian Markov random field (GMRF).
It can scale to fields with millions of variables.
The algorithm is designed for data assimilation in weather models, which is the process of updating a model of the atmosphere based on observations.

For more details, see our accompanying paper [Scalable Data Assimilation with Message Passing; Oscar Key*, So Takao*, Daniel Giles*, Marc Peter Deisenroth](https://arxiv.org/abs/2404.12968).

In this repository we include:
- Tools for defining a Matérn Gaussian process prior on a rectangle or sphere and generating a corresponding GMRF prior
- Three methods for performing inference with this prior
  - Ours: a re-weighted message passing algorithm with multi grid support based on [Ruozzi 2013](https://jmlr.org/papers/v14/ruozzi13a.html) (GPU accelerated)
  - Baseline 1: 3D-Var, computes the posterior mean using optimisation (GPU accelerated)
  - Baseline 2: exact inference, by launching [R-INLA](https://www.r-inla.org/) (CPU only)
- Code to reproduce the experiments in the paper


## Message passing implementation
As our message passing implementation is specialised to GMRFs, we can make some assumptions which improve the efficiency:
- Variables are connected by at most pairwise factors: we do not explicity represent factors, only variables and connections between variables (which implies connection via a pairwise factor)
- Almost all variables have the same degree, as the graph representing the GMRF has a regular structure except at the boundaries: we can use a regular, GPU-ameanable data structure for the factor graph


## To set up the environment
- Install Python 3.11 (e.g. using [pyenv](https://github.com/pyenv/pyenv))
- Install [Poetry](https://python-poetry.org/)
- Run `poetry install`, or `poetry install --with plotting` to also include the dependencies required for plotting
- Optionally: install R and R-INLA if you'd like to run the baseline experiments

## Experiments
To run an experiment: `python src/experiments/[script].py`, where `script` is:
- Try out the methods on simulated data: `mp_demo.py`, `mp_multigrid_demo.py`, `threedvar_demo.py`, `inla_demo.py`
- Reproduce Table 1: `comparison_table.py`
- Reproduce the grid search: `lr_c_grid_search.py`

The experiments on spherical temperature data are in `temperature.py`.
Unfortunately they depend on data from the Met Office's Unified Model which we are unable to include in this repository.
Thus, this code is for reference only.


## Development tools
- Format:
  - `black **/*.py`
  - `isort **/*.py`
- Run tests: `pytest`
- Run type checker:
  - `mypy -p damp`
  - `mypy src/experiments`


## License
We release this code under the MIT license, see [LICENSE](LICENSE).
