"""Demonstrates message passing (without multigrid) on simulated data."""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.random import default_rng

from damp import gp, ground_truth_cache, message_passing
from damp.gp import Shape
from damp.message_passing import Config


def main() -> None:
    numpy_rng = default_rng(seed=1124)

    prior = gp.get_prior(Shape(256, 128))
    ground_truth = next(ground_truth_cache.load_or_gen(prior, start_at=0))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=round(prior.shape.flatten() * 0.01),
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )
    posterior = gp.get_posterior(prior, obs, obs_noise)
    config = Config.from_posterior(posterior, c=-2.0, lr=0.7)
    initial_edges = message_passing.get_initial_edges(config.graph)

    edges, marginals = message_passing.iterate(
        config, initial_edges, n_iterations=50_000
    )

    fig, (gt_ax, mean_ax, std_ax) = plt.subplots(ncols=3, figsize=(8, 3))
    vmin = ground_truth.min()
    vmax = ground_truth.max()
    gt_ax.imshow(ground_truth.T, vmin=vmin, vmax=vmax)
    mean_ax.imshow(marginals.mean.reshape(prior.interior_shape).T, vmin=vmin, vmax=vmax)
    std_ax.imshow(marginals.std.reshape(prior.interior_shape).T)
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
    plt.savefig(plot_dir / "mp_demo.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
