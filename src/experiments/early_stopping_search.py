"""Grid search to compare the early stopping hyperparameters for 3D-Var and MP."""
import pickle
from itertools import product
from pathlib import Path
from time import time
from typing import Any, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import Array
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.random import default_rng

import plotting
from damp import gp, ground_truth_cache, metrics, multigrid, threedvar
from damp.gp import Shape


class Ours:
    def run(
        self,
        prior: gp.Prior,
        obs: gp.Obs,
        obs_noise: float,
        early_stopping_threshold: float,
    ) -> Array:
        output = multigrid.run_rect(
            prior,
            obs,
            obs_noise,
            min_grid_size=32,
            stopping_threshold=early_stopping_threshold,
            c=10.0,
            lr=0.6,
        )
        level, marginals = output[-1]
        return marginals.mean

    @property
    def name(self) -> str:
        return "mp_multigrid"


class ThreeDVar:
    def run(
        self,
        prior: gp.Prior,
        obs: gp.Obs,
        obs_noise: float,
        early_stopping_threshold: float,
    ) -> Array:
        return threedvar.run_optimizer(
            jax.random.key(seed=123456),
            prior,
            obs,
            obs_noise,
            early_stopping_threshold,
        )

    @property
    def name(self) -> str:
        return "map_lbfgs"


Method = Union[Ours, ThreeDVar]


def main() -> None:
    save_path = Path("outputs/early_stopping/")
    save_path.mkdir(parents=True, exist_ok=True)

    n_repeats = 3
    grid_sizes = [256, 512]
    obs_fractions = [0.01, 0.05, 0.1]
    early_stopping_threshold = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
    methods: list[Method] = [Ours(), ThreeDVar()]
    for method in methods:
        results = []
        configurations = product(
            grid_sizes, obs_fractions, early_stopping_threshold, range(n_repeats)
        )
        for grid_size, obs_fraction, stopping_threshold, repeat in configurations:
            variables = {
                "grid size": grid_size,
                "obs fraction": obs_fraction,
                "method": method.name,
                "repeat": repeat,
                "early stopping threshold": stopping_threshold,
            }
            file_name = f"{grid_size}_{obs_fraction}_{method.name}_{stopping_threshold}_{repeat}"
            output_path = save_path / file_name
            info = f"{method.name}: {grid_size} {obs_fraction} {stopping_threshold} {repeat}"
            if output_path.exists():
                print(f"Load {info}")
                with open(output_path, "rb") as f:
                    outputs = pickle.load(f)
            else:
                print(f"Run {info}")
                outputs = _run_repeat(
                    repeat,
                    grid_size,
                    obs_fraction,
                    stopping_threshold,
                    method,
                )
                with open(output_path, "wb") as f:
                    pickle.dump(outputs, f)

            results.append(variables | outputs)

        df = _print_results(pd.DataFrame(results))
        df.to_csv(save_path / f"pareto_{method.name}.csv")

        plot_pareto(
            method.name, grid_sizes, obs_fractions, early_stopping_threshold, save_path
        )
    return


def plot_pareto(
    method: str,
    grid_sizes: list[int],
    obs_fractions: list[float],
    thresholds: list[float],
    save_path: Path,
):
    plotting.configure_matplotlib()
    df = pd.read_csv(save_path / f"pareto_{method}.csv", skiprows=2)
    df.columns = list(df.columns[:-2]) + ["RMSE", "runtime"]
    c = ["r", "b", "k"]
    marker_styles = ["*", "o", "^", "s", "<", "p"]
    percent_observed = [round(frac * 100) for frac in obs_fractions]
    args = ([0], [0])
    param_name = "$\\tau$" if method == "mp_multigrid" else "tolerance"
    legend_elements = [
        Line2D(*args, color="r", lw=2, label=f"${percent_observed[0]}$\\% observed"),
        Line2D(*args, color="b", lw=2, label=f"${percent_observed[1]}$\\% observed"),
        Line2D(*args, color="k", lw=2, label=f"${percent_observed[2]}$\\% observed"),
    ] + [
        Line2D(
            *args,
            marker=marker_style,
            color="w",
            markerfacecolor="k",
            markersize=8,
            label=f"{param_name} = {threshold:.0e}",
        )
        for threshold, marker_style in zip(thresholds, marker_styles, strict=True)
    ]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(plotting.FULL_WIDTH, 2.6))
    plt.title(f"{method}")
    for i, (grid, ax) in enumerate(zip(grid_sizes, axes, strict=True)):
        ax.title.set_text(f"${grid} \\times {grid}$")
        for obs_index, obs in enumerate(obs_fractions):
            for index, stopping in enumerate(thresholds):
                data = df[
                    (df["grid size"] == grid)
                    & (df["obs fraction"] == obs)
                    & (df["early stopping threshold"] == stopping)
                ]
                ax.plot(
                    data["runtime"],
                    data["RMSE"],
                    color=c[obs_index],
                    marker=marker_styles[index],
                )
            line_data = df[(df["grid size"] == grid) & (df["obs fraction"] == obs)]
            ax.plot(
                line_data["runtime"], line_data["RMSE"], color=c[obs_index], linewidth=2
            )

        ax.set_xlabel("runtime (s)")
        if i == 0:
            ax.set_ylabel("RMSE")

    fig.subplots_adjust(bottom=0.4, wspace=0.4)
    legend_ax: Axes = axes[0]
    legend_ax.legend(
        handles=legend_elements,
        ncols=3,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.35),
        **plotting.squashed_legend_params,
    )

    plotting.save_fig(f"pareto_{method}")
    plt.show()
    plt.close()
    return


def _print_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(
        ["grid size", "obs fraction", "early stopping threshold", "method"], sort=False
    ).aggregate({"duration": ["mean"], "rmse": ["mean"]})
    df["RMSE"] = df["rmse"].apply(lambda row: f"{row['mean']:.3f}", axis="columns")
    df["duration (seconds)"] = df["duration"].apply(
        lambda row: f"{row['mean']:.3f}", axis="columns"
    )
    df = df.drop(["duration", "rmse"], axis="columns")
    # print(df.to_latex())
    return df


def _run_repeat(
    repeat_i: int,
    grid_size: int,
    obs_fraction: float,
    early_stopping_threshold: float,
    method: Method,
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
    mean = method.run(prior, obs, obs_noise, early_stopping_threshold)
    duration = time() - start_time

    return {
        "duration": duration,
        "rmse": metrics.rmse(mean, jnp.array(ground_truth)).item(),
    }


if __name__ == "__main__":
    main()
