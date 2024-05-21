from collections.abc import Iterable
from math import ceil
from typing import Callable, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array, tree_util, vmap
from tqdm import tqdm

T = TypeVar("T")


def tree_concatenate(trees: Iterable[T]) -> T:
    """Concatenates the leaves of a list of pytrees to produce a single pytree."""
    leaves, treedefs = zip(*[tree_util.tree_flatten(tree) for tree in trees])
    grouped_leaves = zip(*leaves)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return cast(T, treedefs[0].unflatten(result_leaves))


def batch_vmap(
    f: Callable[[Array], T], xs: Array, batch_size: int, progress: bool = False
) -> T:
    """Equivalent to vmap(f)(xs), but vmaps only batch_size elements of xs at a time.

    This reduces memory usage.

    :param progress: If True, displays a progress bar.
    """
    n_batches = int(ceil(xs.shape[0] / batch_size))

    batch_results: list[T] = []
    progress_bar = tqdm(total=xs.shape[0]) if progress else None
    for batch_i in range(n_batches):
        batch_xs = xs[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_results.append(vmap(f)(batch_xs))
        if progress_bar is not None:
            progress_bar.update(len(batch_xs))

    return tree_concatenate(batch_results)


F = TypeVar("F", bound=Callable)


def with_jittable_assertions(f: F) -> F:
    """Wraps chex.with_jittable_assertions, passing through the type of the function."""
    return chex.with_jittable_assertions(f)  # type: ignore


def jit(f: F, *args, **kwargs) -> F:
    """Wraps jax.jit() but passes through the type of the function."""
    return jax.jit(f, *args, **kwargs)  # type: ignore
