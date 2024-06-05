"""Demonstrates message passing (without multigrid) on simulated data."""
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

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4, 2))

    edges, marginals = message_passing.iterate(
        config, initial_edges, n_iterations=50_000
    )

    vmin = ground_truth.min()
    vmax = ground_truth.max()
    axes[0].imshow(ground_truth.T, vmin=vmin, vmax=vmax)
    axes[1].imshow(marginals.mean.reshape(prior.interior_shape).T, vmin=vmin, vmax=vmax)
    axes[2].imshow(marginals.std.reshape(prior.interior_shape).T)
    obs_xs, obs_ys = zip(*[(x - 1, y - 1) for (x, y), val in obs], strict=True)
    axes[2].scatter(obs_xs, obs_ys, color="red", s=1)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("plots/mp_demo.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
