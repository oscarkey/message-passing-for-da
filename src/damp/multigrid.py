from collections.abc import Iterator
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy import ndarray

from damp import gp, message_passing
from damp.gp import Obs, Prior, Shape
from damp.jax_utils import jit
from damp.message_passing import Config, Edges, Marginals

DEFAULT_MAX_ITERATIONS = 10_000
DEFAULT_STOPPING_THRESHOLD = 0.0001

ObsMatrix = ndarray


def run_rect(
    prior: Prior,
    obs: Obs,
    obs_noise: float,
    min_grid_size: int = 32,
    stopping_threshold: float = DEFAULT_STOPPING_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    **config_kwargs,
) -> list[tuple[Shape, Marginals]]:
    """Runs multi-grid message passing on a rectangular domain.

    Starts running message passing on a low-resolution grid, and doubles the resolution
    of the grid until it reaches the target shape. The coarse grid will have the same
    aspect ratio as the target grid, with its longest side having length min_grid_size.

    :param max_iterations: the maximum number of iterations at a particular level. Early
                           stopping is used so message passing may stop before this many
                           iterations.
    """
    target_shape = prior.shape
    # TODO: Ensure the level prior has the same parameters as the target prior.
    # At the moment this doesn't matter because the only parameter is the grid
    # shape.
    priors_except_target = [
        gp.get_prior(shape)
        for shape in _build_level_shapes(min_grid_size, target_shape)[:-1]
    ]
    all_level_priors = priors_except_target + [prior]
    return _run(
        all_level_priors,
        obs,
        obs_noise,
        stopping_threshold,
        max_iterations,
        **config_kwargs,
    )


def run_sphere(
    prior: Prior,
    obs: Obs,
    obs_noise: float,
    lat: ndarray,
    lon: ndarray,
    ratios: list[int],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    **config_kwargs,
) -> list[tuple[Shape, Marginals]]:
    """Runs multi-grid message passing on a spherical domain."""
    target_width, target_length = prior.shape
    levels = [
        Shape(int(target_width / level), int(target_length / level)) for level in ratios
    ]
    # TODO: Ensure the level prior has the same parameters as the target prior.
    # At the moment this doesn't matter because the only parameter is the grid
    # shape.
    priors_except_target = [
        gp.get_prior_sphere(shape, lon[::ratio], lat[::ratio])
        for shape, ratio in zip(levels[:-1], ratios[:-1], strict=True)
    ]
    all_level_priors = priors_except_target + [prior]
    return _run(all_level_priors, obs, obs_noise, max_iterations, **config_kwargs)


def _run(
    level_priors: list[Prior],
    obs: Obs,
    obs_noise: float,
    stopping_threshold: float,
    max_iterations: int,
    **config_kwargs,
) -> list[tuple[Shape, Marginals]]:
    level_configs = list(
        _get_level_configs(obs, obs_noise, level_priors, config_kwargs)
    )

    marginals = []
    edges = message_passing.get_initial_edges(level_configs[0].graph)

    for level_i, (prior, config) in enumerate(
        zip(level_priors, level_configs, strict=True)
    ):
        print(
            f"Running Message Passing for level {level_i} "
            f"({prior.shape.width} x {prior.shape.height})"
        )

        edges, level_marginals = message_passing.iterate(
            config,
            edges,
            n_iterations=max_iterations,
            early_stopping_threshold=stopping_threshold,
        )
        level_marginals = Marginals(
            mean=level_marginals.mean.reshape(prior.interior_shape),
            std=level_marginals.std.reshape(prior.interior_shape),
        )
        marginals.append(level_marginals)

        last_level = level_i == len(level_priors) - 1
        if not last_level:
            # Duplicate the edges to the resolution of the next level.
            edges = _expand_edges(
                edges,
                level_priors[level_i].interior_shape,
                level_priors[level_i + 1].interior_shape,
            )

    return [(prior.shape, m) for prior, m in zip(level_priors, marginals, strict=True)]


def _get_level_configs(
    obs: Obs,
    obs_noise: float,
    level_priors: list[Prior],
    config_kwargs: dict[str, Any],
) -> Iterator[Config]:
    target_shape = level_priors[-1].shape
    obs_grid = fill_obs_matrix(obs, level_priors[-1].shape)
    for level_prior in level_priors:
        level_obs = pull_obs_from_target(obs_grid, target_shape, level_prior.shape)
        posterior = gp.get_posterior(level_prior, level_obs, obs_noise)
        yield Config.from_posterior(posterior, **config_kwargs)


def _expand_edges(edges: Edges, previous_shape: Shape, next_shape: Shape) -> Edges:
    """Gets the initial edges for the next multigrid level."""
    return _expand_edges_jittable(
        a_grid=edges.a.reshape(
            previous_shape.width, previous_shape.height, edges.a.shape[1]
        ),
        b_grid=edges.b.reshape(
            previous_shape.width, previous_shape.height, edges.b.shape[1]
        ),
        width_ratio=next_shape.width // previous_shape.width,
        height_ratio=next_shape.height // previous_shape.height,
    )


@partial(jit, static_argnames=("width_ratio", "height_ratio"))
def _expand_edges_jittable(
    a_grid: Array, b_grid: Array, width_ratio: int, height_ratio: int
) -> Edges:
    a = width_ratio * height_ratio * a_grid
    a = jnp.repeat(a, width_ratio, axis=0)
    a = jnp.repeat(a, height_ratio, axis=1)
    a = jnp.pad(a, ((1, 1), (1, 1), (0, 0)))
    a = jnp.nan_to_num(a)
    a = a.reshape(-1, a_grid.shape[2])

    b = width_ratio * height_ratio * b_grid
    b = jnp.repeat(b, width_ratio, axis=0)
    b = jnp.repeat(b, height_ratio, axis=1)
    b = jnp.pad(b, ((1, 1), (1, 1), (0, 0)))
    b = jnp.nan_to_num(b)
    b = b.reshape(-1, b_grid.shape[2])

    return Edges(a=a, b=b)


def _build_level_shapes(min_size: int, target_shape: Shape) -> list[Shape]:
    assert target_shape.width % 2 == 0 and target_shape.height % 2 == 0
    assert min_size >= 2 and min_size % 2 == 0
    assert target_shape.width % min_size == 0 and target_shape.height % min_size == 0

    # We reduce the largest dimension to the minimum size, thus the other dimension
    # might end up smaller.
    if target_shape.width >= target_shape.height:
        min_shape = Shape(
            width=min_size,
            height=round(target_shape.height / target_shape.width * min_size),
        )
    else:
        min_shape = Shape(
            width=round(target_shape.width / target_shape.height * min_size),
            height=min_size,
        )

    levels = [min_shape]
    while levels[-1] != target_shape:
        levels.append(Shape(levels[-1].width * 2, levels[-1].height * 2))
    return levels


def fill_obs_matrix(obs: Obs, target_shape: Shape) -> ObsMatrix:
    """
    Fill a matrix with the observation values and locations.
    """
    obs_matrix = np.zeros(target_shape)
    for idx, val in obs:
        obs_matrix[idx[0], idx[1]] = val
    return obs_matrix


def pull_obs_from_target(
    obs_grid: ndarray, target_shape: Shape, level_shape: Shape
) -> Obs:
    """Selects observations from obs_grid that align with cells on the current grid.

    :param level_shape: the shape of the grid at the current level
    """
    assert target_shape.width % level_shape.width == 0
    assert target_shape.height % level_shape.height == 0
    width_ratio = target_shape.width // level_shape.width
    height_ratio = target_shape.height // level_shape.height

    collocated_points_mask = np.zeros(target_shape)
    # Select which points on the fine grid are also on the coarse grid.
    collocated_points_mask[::width_ratio, ::height_ratio] = 1
    # Select the observations at the collocated points.
    obs_on_grid = collocated_points_mask * obs_grid
    # Adjust the (x,y) coordinates of the remaining points on the target grid so they
    # are in the coordinate system of the coarse grid.
    regridded_observations = [
        ((xy[0] // width_ratio, xy[1] // height_ratio), obs_on_grid[xy[0], xy[1]])
        for xy in np.argwhere(obs_on_grid != 0)
    ]
    return regridded_observations
