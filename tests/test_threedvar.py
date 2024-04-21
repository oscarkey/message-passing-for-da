import jax.random
from numpy.random import default_rng

from damp import gp, ground_truth_cache, threedvar
from damp.gp import Shape


def test__run_optimizer__does_not_crash() -> None:
    prior = gp.get_prior(Shape(16, 16))
    ground_truth = next(ground_truth_cache.load_or_gen(prior))
    obs_noise = 1e-3
    numpy_rng = default_rng(seed=1124)
    obs = gp.choose_observations(
        numpy_rng,
        n_obs=60,
        ground_truth=ground_truth,
        obs_noise=obs_noise,
    )

    rng = jax.random.key(seed=23142834)
    rng, rng_input = jax.random.split(rng)

    mean = threedvar.run_optimizer(rng_input, prior, obs, obs_noise)

    assert mean.shape == prior.interior_shape
