"""Demonstrates 3D-Var on simulated data on a 2D rectangle."""

from pathlib import Path
from time import time

import jax.random
import matplotlib.pyplot as plt
from numpy.random import default_rng

import damp.threedvar as threedvar
from damp import gp, ground_truth_cache
from damp.gp import Shape


def main() -> None:
    numpy_rng = default_rng(seed=1124)

    prior = gp.get_prior(Shape(128, 256))
    ground_truth = next(ground_truth_cache.load_or_gen(prior))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=round(prior.shape.flatten() * 0.01),
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )

    rng = jax.random.key(seed=123456)
    rng, rng_input = jax.random.split(rng)

    start = time()
    result = threedvar.run_optimizer(rng_input, prior, obs, obs_noise)
    end = time()
    print(f"Took {end-start:.2f}s")

    fig, (gt_ax, pred_ax) = plt.subplots(ncols=2, figsize=(8, 3))
    vmin = ground_truth.min()
    vmax = ground_truth.max()
    gt_ax.imshow(ground_truth, vmin=vmin, vmax=vmax)
    pred_ax.imshow(result, vmin=vmin, vmax=vmax)

    gt_ax.set_title("ground truth", fontsize=10)
    pred_ax.set_title("predicted mean", fontsize=10)
    for ax in (gt_ax, pred_ax):
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "3dvar_demo.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
