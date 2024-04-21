"""Compares message passing, 3D-Var, and INLA on simulated data."""
import pickle
from pathlib import Path
from time import time
from typing import Any, Union

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array
from numpy.random import default_rng

from damp import gp, ground_truth_cache, inla_bridge, metrics, multigrid, threedvar
from damp.gp import Shape


class Ours:
    def run(self, prior: gp.Prior, obs: gp.Obs, obs_noise: float) -> Array:
        output = multigrid.run_rect(
            prior, obs, obs_noise, min_grid_size=32, c=10.0, lr=0.6
        )
        level, marginals = output[-1]
        return marginals.mean

    @property
    def name(self) -> str:
        return "mp_multigrid"


class INLA:
    def run(self, prior: gp.Prior, obs: gp.Obs, obs_noise: float) -> Array:
        mean, std = inla_bridge.run(prior, obs)
        return jnp.array(mean)

    @property
    def name(self) -> str:
        return "inla"


class ThreeDVar:
    def run(self, prior: gp.Prior, obs: gp.Obs, obs_noise: float) -> Array:
        return threedvar.run_optimizer(
            jax.random.key(seed=2343499), prior, obs, obs_noise
        )

    @property
    def name(self) -> str:
        return "map_lbfgs"


Method = Union[Ours, INLA, ThreeDVar]


def main() -> None:
    save_path = Path("outputs/comparison_table/")
    save_path.mkdir(parents=True, exist_ok=True)

    n_repeats = 3
    grid_sizes = [256, 512, 1024]
    obs_fractions = [0.01, 0.05, 0.1]
    methods: list[Method] = [Ours(), ThreeDVar(), INLA()]

    results = []
    for method in methods:
        for grid_size in grid_sizes:
            for obs_fraction in obs_fractions:
                for repeat in range(n_repeats):
                    variables = {
                        "grid size": grid_size,
                        "obs fraction": obs_fraction,
                        "method": method.name,
                        "repeat": repeat,
                    }
                    file_name = f"{grid_size}_{obs_fraction}_{method.name}_{repeat}"
                    output_path = save_path / file_name
                    if output_path.exists():
                        print(
                            f"Load {method.name}: {grid_size} {obs_fraction} {repeat}"
                        )
                        with open(output_path, "rb") as f:
                            outputs = pickle.load(f)
                    else:
                        print(f"Run {method.name}: {grid_size} {obs_fraction} {repeat}")
                        outputs = _run_repeat(repeat, grid_size, obs_fraction, method)
                        with open(output_path, "wb") as f:
                            pickle.dump(outputs, f)

                    results.append(variables | outputs)

    _print_results(pd.DataFrame(results))


def _print_results(df: pd.DataFrame) -> None:
    def sort_key(s: pd.Series) -> pd.Series:
        if s.name == "method":
            return s.replace({"inla": 0, "map_lbfgs": 1, "mp_multigrid": 2})
        else:
            return s

    df = df.sort_values(by=["grid size", "obs fraction", "method"], key=sort_key)

    df = df.replace({"inla": "INLA", "map_lbfgs": "3D-Var", "mp_multigrid": "ours"})
    df = df.groupby(["grid size", "obs fraction", "method"], sort=False).aggregate(
        {"duration": ["mean"], "rmse": ["mean"]}
    )
    df["RMSE"] = df["rmse"].apply(lambda row: f"{row['mean']:.2f}", axis="columns")
    df["duration (seconds)"] = df["duration"].apply(
        lambda row: f"{row['mean']:.0f}", axis="columns"
    )
    df = df.drop(["duration", "rmse"], axis="columns")
    df.columns = ["".join(x) for x in df.columns.to_flat_index()]
    df = df.reset_index().pivot(index=["grid size", "obs fraction"], columns=["method"])
    df = df.reindex(columns=df.columns.reindex(["INLA", "3D-Var", "ours"], level=1)[0])
    df.index = df.index.map(lambda x: (x[0], f"{x[1] * 100:.0f}\%"))
    print(df)
    print(df.to_latex())


def _run_repeat(
    repeat_i: int, grid_size: int, obs_fraction: float, method: Method
) -> dict[str, Any]:
    n_obs = round(grid_size**2 * obs_fraction)
    numpy_rng = default_rng(seed=n_obs)
    prior = gp.get_prior(Shape(grid_size, grid_size))
    ground_truth = next(ground_truth_cache.load_or_gen(prior, start_at=repeat_i))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng, n_obs, ground_truth=ground_truth, obs_noise=obs_noise
    )

    start_time = time()
    mean = method.run(prior, obs, obs_noise)
    duration = time() - start_time

    return {
        "duration": duration,
        "rmse": metrics.rmse(mean, jnp.array(ground_truth)).item(),
    }


if __name__ == "__main__":
    main()
