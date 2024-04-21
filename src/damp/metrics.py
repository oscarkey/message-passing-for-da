from typing import Union

import jax
import jax.numpy as jnp
from numpy import ndarray

from damp.jax_utils import jit

Array = Union[jax.Array, ndarray]


@jit
def rmse(mean: Array, ground_truth: Array) -> Array:
    gt_interior = ground_truth[1:-1, 1:-1]
    assert mean.shape == gt_interior.shape
    return jnp.sqrt(jnp.mean((mean - gt_interior) ** 2))
