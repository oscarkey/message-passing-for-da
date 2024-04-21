"""Runs a grid search on simulated data to select lr and c for message passing."""
import functools
import itertools
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import default_rng

from damp import gp, ground_truth_cache, inla_bridge, message_passing
from damp.gp import Obs, Shape
from damp.message_passing import Config
from damp.metrics import rmse
from experiments import plotting


def main() -> None:
    plotting.configure_matplotlib()
    plt.figure(figsize=(plotting.HALF_WIDTH, 1.3))

    save_dir = Path("outputs/lr_c_convergence")
    save_dir.mkdir(parents=True, exist_ok=True)

    lrs = [0.6, 0.7, 0.8]
    cs = [-10, -2, -1, 1, 5, 10, 20]
    grid_sizes = [128, 256, 512]
    hyperparameters = [
        {"lr": lr, "c": c, "grid_size": s}
        for c, lr, s in itertools.product(cs, lrs, grid_sizes)
    ]

    y_top = 0.5

    inla_rmses = {}
    for grid_size in grid_sizes:
        output_path = save_dir / f"inla_{grid_size}.npy"
        if output_path.exists():
            inla_result = np.load(output_path, allow_pickle=True).item()
        else:
            inla_result = _run_inla(grid_size)
            np.save(output_path, inla_result)
        inla_rmses[grid_size] = inla_result["rmse"].item()

    results = []
    for i, hyps in enumerate(hyperparameters):
        output_path = save_dir / f"{_dict_to_str(hyps)}.npy"
        if output_path.exists():
            result = np.load(output_path, allow_pickle=True).item()
        else:
            result = _run_mp(hyps)
            np.save(output_path, result)

        mid_rmse = (
            result["rmses"][result["steps"] == 4000].item() / inla_rmses[grid_size]
        )
        results.append({"mid rmse": mid_rmse} | hyps | result)

    df = pd.DataFrame(results)

    to_print = df[["mid rmse", "lr", "c", "grid_size"]]
    to_print = to_print.pivot(index=["grid_size", "lr"], columns="c", values="mid rmse")
    to_print.columns.name = None
    to_print.index = to_print.index.map(
        lambda x: (f"${x[0]} \\times {x[0]}$", f"{x[1]:.1f}")
    )
    to_print = to_print.map(lambda x: f"{x:.2f}" if not np.isnan(x) else "-")
    with pd.option_context("display.max_rows", None):
        print(to_print)
        print(to_print.to_latex(float_format="%.2f"))

    plot_grid_size = 256
    to_plot = df[
        (df["grid_size"] == plot_grid_size) & (df["c"].isin([-10, -1, 1, 5, 10, 20]))
    ]
    to_plot = to_plot.loc[to_plot.groupby("c")["c"].idxmax()]

    for i, (_, row) in enumerate(to_plot.iterrows()):
        steps, rmses = row["steps"], row["rmses"]
        bad = np.isnan(rmses)
        c = row["c"]
        c_str = c if c < 100.0 else f"10^{round(math.log(c, 10))}"
        label = f"$c={c_str}$"
        color = f"C{i}"
        if np.any(bad):
            plt.scatter([-10], [-10], label=label, marker="x", color=color)
        else:
            plt.plot(steps[~bad], rmses[~bad], label=label, color=color)

    plt.axhline(inla_rmses[plot_grid_size], color="black")

    plt.legend(**plotting.squashed_legend_params, ncols=2)
    plt.xlim(left=0, right=4000)
    plt.ylim(bottom=inla_rmses[plot_grid_size] - 0.05, top=y_top)
    plt.xlabel("iterations", **plotting.squashed_label_params)
    plt.ylabel("RMSE", **plotting.squashed_label_params)

    plt.tight_layout(pad=0.2)
    plotting.save_fig("lr_c_convergence")
    plt.close()


def _dict_to_str(d: dict[str, Any]) -> str:
    return "_".join(f"{k}_{v}" for k, v in sorted(d.items()))


def _run_mp(hyps: dict[str, Any]) -> dict[str, ndarray]:
    prior, gt, obs_noise, obs = _get_prior(hyps["grid_size"])
    posterior = gp.get_posterior(prior, obs, obs_noise)
    config = Config.from_posterior(
        posterior,
        **{k: v for k, v in hyps.items() if k != "grid_size"},
    )
    initial_edges = message_passing.get_initial_edges(config.graph)

    history = message_passing.iterate_with_history(
        config, initial_edges, n_iterations=5000, save_every=50
    )

    steps = np.array([step for step, _, _ in history])
    rmses = np.stack(
        [
            rmse(marginals.mean.reshape(prior.interior_shape), gt)
            for _, _, marginals in history
        ]
    )
    return {"steps": steps, "rmses": rmses}


def _run_inla(grid_size: int) -> dict[str, ndarray]:
    prior, gt, _, obs = _get_prior(grid_size)
    mean, std = inla_bridge.run(prior, obs)
    return {"rmse": np.array(rmse(mean, gt))}


@functools.cache
def _get_prior(grid_size: int) -> tuple[gp.Prior, ndarray, float, Obs]:
    numpy_rng = default_rng(seed=1293123)
    prior = gp.get_prior(Shape(grid_size, grid_size))
    ground_truth = next(ground_truth_cache.load_or_gen(prior))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=round(prior.shape.flatten() * 0.05),
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )
    return prior, ground_truth, obs_noise, obs


if __name__ == "__main__":
    main()
