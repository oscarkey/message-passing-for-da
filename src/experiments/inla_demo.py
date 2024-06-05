"""Demonstrates INLA on simulated data on a 2D rectangle."""
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

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    vmin = ground_truth.min()
    vmax = ground_truth.max()
    axes[0].imshow(ground_truth.T, vmin=vmin, vmax=vmax)
    axes[1].imshow(pred_means.T, vmin=vmin, vmax=vmax)
    axes[2].imshow(pred_stds.T)
    obs_xs, obs_ys = zip(*[(x - 1, y - 1) for (x, y), val in obs], strict=True)
    axes[2].scatter(obs_xs, obs_ys, color="red", s=1)
    plt.savefig("plots/inla_demo.png")
    plt.close()


if __name__ == "__main__":
    main()
