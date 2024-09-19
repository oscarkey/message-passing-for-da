"""Demonstrates 3D-Var on simulated data on a 2D rectangle."""
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

    n_rows = 2
    fig, axes = plt.subplots(n_rows, ncols=2, figsize=(4, n_rows * 1.5), squeeze=False)
    vmin = ground_truth.min()
    vmax = ground_truth.max()
    axes[0, 0].imshow(ground_truth, vmin=vmin, vmax=vmax)
    axes[-1, 0].imshow(result, vmin=vmin, vmax=vmax)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("plots/3dvar_demo.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
