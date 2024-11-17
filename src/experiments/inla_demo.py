"""Demonstrates INLA on simulated data on a 2D rectangle."""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.random import default_rng

from damp import gp, ground_truth_cache, inla_bridge
from damp.gp import Shape


def main() -> None:
    numpy_rng = default_rng(seed=1120)

    prior = gp.get_prior(Shape(24, 16))
    ground_truth = next(ground_truth_cache.load_or_gen(prior))
    obs = gp.choose_observations(
        numpy_rng, n_obs=30, ground_truth=ground_truth, obs_noise=1e-3
    )

    pred_means, pred_stds = inla_bridge.run(prior, obs)

    fig, (gt_ax, mean_ax, std_ax) = plt.subplots(ncols=3, figsize=(10, 3))
    vmin = ground_truth.min()
    vmax = ground_truth.max()
    gt_ax.imshow(ground_truth.T, vmin=vmin, vmax=vmax)
    mean_ax.imshow(pred_means.T, vmin=vmin, vmax=vmax)
    std_ax.imshow(pred_stds.T)
    obs_xs, obs_ys = zip(*[(x - 1, y - 1) for (x, y), val in obs], strict=True)
    std_ax.scatter(obs_xs, obs_ys, color="red", s=1)

    gt_ax.set_title("ground truth", fontsize=10)
    mean_ax.set_title("predicted mean", fontsize=10)
    std_ax.set_title("predicted std", fontsize=10)
    for ax in (gt_ax, mean_ax, std_ax):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "inla_demo.png")
    plt.close()


if __name__ == "__main__":
    main()
