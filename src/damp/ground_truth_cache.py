"""Simulates ground truth data from a GP prior, and caches the results."""
import itertools
from pathlib import Path
from time import time
from typing import Iterator

import numpy as np
from numpy import ndarray
from numpy.random import default_rng

from damp import gp

DIRECTORY = Path("outputs/ground_truths")
BASE_SEED = 901823


def load_or_gen(prior: gp.Prior, start_at: int = 0) -> Iterator[ndarray]:
    def gen() -> Iterator[ndarray]:
        for seed in range(BASE_SEED, BASE_SEED + 1000):
            try:
                gt = _load_one(prior, seed)
                print(f"Loaded ground truth with seed {seed}")
                yield gt
            except OSError:
                print("Run out of cached ground truths, generating one...")
                gt = _gen_and_save_one(prior, seed)
                yield gt

    return itertools.islice(gen(), start_at, None)


def _load_one(prior: gp.Prior, seed: int) -> ndarray:
    return np.load(DIRECTORY / f"{_get_name(prior, seed)}.npy")


def _gen_and_save_one(prior: gp.Prior, seed: int) -> ndarray:
    numpy_rng = default_rng(seed)
    start_time = time()
    gt = gp.sample_prior(numpy_rng, prior)
    duration = time() - start_time
    print(f"Generated grouth truth with seed {seed} in {duration:.2f} seconds")

    DIRECTORY.mkdir(exist_ok=True, parents=True)
    output_path = DIRECTORY / f"{_get_name(prior, seed)}.npy"
    np.save(output_path, gt)

    return gt


def _get_name(prior: gp.Prior, seed: int) -> str:
    return f"{prior.name}_seed_{seed}"
