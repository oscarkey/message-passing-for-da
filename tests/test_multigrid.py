import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import default_rng

from damp import gp, ground_truth_cache, message_passing, multigrid
from damp.gp import Shape
from damp.message_passing import Config


def test__fill_obs_matrix__obs_at_correct_positions_in_matrix() -> None:
    obs = [
        ((0, 0), 1),
        ((1, 1), 2),
        ((2, 1), -1),
        ((2, 2), 3),
    ]
    target_shape = Shape(3, 3)
    expected_result = np.array([[1, 0, 0], [0, 2, 0], [0, -1, 3]])

    result = multigrid.fill_obs_matrix(obs, target_shape)
    assert np.array_equal(result, expected_result)


def test__fill_obs_matrix__empty_obs__matrix_all_zeros() -> None:
    # Empty observation
    obs = []
    target_shape = Shape(3, 3)
    expected_result = np.zeros(target_shape)

    result = multigrid.fill_obs_matrix(obs, target_shape)
    assert np.array_equal(result, expected_result)


def test__fill_obs_matrix__out_of_bounds_index__raises() -> None:
    obs = [
        ((0, 0), 1),
        ((1, 1), 2),
        ((2, 1), -1),
        ((2, 2), 3),
        ((3, 1), 4),
    ]
    target_shape = Shape(3, 3)

    with pytest.raises(IndexError):
        multigrid.fill_obs_matrix(obs, target_shape)


def test__pull_obs_from_target__obs_have_correct_value() -> None:
    target_shape = Shape(8, 8)
    coarse_shape = Shape(2, 2)
    obs_grid = np.arange(1, target_shape.flatten() + 1)
    obs_grid = obs_grid.reshape(target_shape)
    obs = multigrid.pull_obs_from_target(obs_grid, target_shape, coarse_shape)
    assert len(obs) == coarse_shape.flatten()
    assert obs[0][1] == 1.0
    assert obs[1][1] == 5.0
    assert obs[2][1] == 33.0
    assert obs[3][1] == 37.0


def test__pull_obs_from_target__not_divisible__raises() -> None:
    # The target shape is not a multiple of the coarse shape.
    target_shape = Shape(3, 3)
    coarse_shape = Shape(2, 2)

    obs_grid = np.arange(1, target_shape.flatten() + 1)
    obs_grid = obs_grid.reshape(target_shape)

    with pytest.raises(AssertionError):
        multigrid.pull_obs_from_target(obs_grid, target_shape, coarse_shape)


def test__run__converges_to_similar_to_no_multigrid() -> None:
    numpy_rng = default_rng(seed=1124)
    prior = gp.get_prior(Shape(32, 32))
    ground_truth = next(ground_truth_cache.load_or_gen(prior, start_at=0))
    obs_noise = 1e-3
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=500,
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )
    posterior = gp.get_posterior(prior, obs, obs_noise)

    _, multigrid_marginals = multigrid.run_rect(
        prior,
        obs,
        obs_noise,
        min_grid_size=8,
        c=-2.0,
        lr=0.7,
    )[-1]

    config = Config.from_posterior(posterior, c=-2.0, lr=0.7)
    _, standard_marginals = message_passing.iterate(
        config,
        message_passing.get_initial_edges(config.graph),
        n_iterations=200,
        progress_bar=False,
    )

    error = jnp.abs(multigrid_marginals.mean - standard_marginals.mean.reshape(30, 30))
    assert jnp.all(error < ground_truth.max() * 0.01)
