"""Demonstrates message passing (with multigrid) on simulated data."""
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy.random import default_rng

from damp import gp, ground_truth_cache, metrics, multigrid
from damp.gp import Shape


def main() -> None:
    numpy_rng = default_rng(seed=1124)
    prior = gp.get_prior(Shape(256, 256))
    ground_truth = next(ground_truth_cache.load_or_gen(prior, start_at=0))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=round(prior.shape.flatten() * 0.05),
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )
    start_time = time.time()
    output = multigrid.run_rect(
        prior,
        obs,
        obs_noise,
        min_grid_size=32,
        c=-2.0,
        lr=0.7,
    )
    end_time = time.time()
    print("Total Runtime = ", (end_time - start_time))

    _, final_marginals = output[-1]
    print(f"RMSE = {metrics.rmse(final_marginals.mean, ground_truth).item()}")

    plot_path = Path("plots/multigrid")
    plot_path.mkdir(exist_ok=True, parents=True)
    for level_i, (level_shape, level_marginals) in enumerate(output):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))
        vmin = ground_truth.min()
        vmax = ground_truth.max()

        axes[0].imshow(ground_truth.T, vmin=vmin, vmax=vmax)
        axes[1].imshow(jnp.pad(level_marginals.mean.T, 1), vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth", fontsize=8)
        axes[1].set_title(
            f"Multigrid (level {level_i}: {level_shape.width}x{level_shape.height})",
            fontsize=8,
        )
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(plot_path / f"level_{level_i}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
