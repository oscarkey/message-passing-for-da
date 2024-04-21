import functools

import jax.numpy as jnp
import jax.random
from jax import Array
from jax.experimental.sparse import BCSR, sparsify
from jaxopt import LBFGS

from damp import gp
from damp.jax_utils import jit


def run_optimizer(rng: Array, prior: gp.Prior, obs: gp.Obs, obs_noise: float) -> Array:
    x_init = 0.1 * jax.random.normal(rng, (prior.precision.shape[0],))
    prior_mean = jnp.zeros_like(x_init)
    # We use batched CSR format as this offers fast matrix-vector products.
    np_prior_precision = prior.precision.tocsr()
    prior_precision = BCSR(
        (
            np_prior_precision.data,
            np_prior_precision.indices,
            np_prior_precision.indptr,
        ),
        shape=np_prior_precision.shape,
    )
    obs_vals = jnp.array([val for _, val in obs])
    obs_idxs = jnp.array(
        [(y - 1) + (x - 1) * prior.interior_shape.height for (x, y), _ in obs]
    )

    opt = _create_optimizer()
    x_final, _ = opt.run(
        x_init, prior_mean, prior_precision, obs_vals, obs_idxs, obs_noise
    )
    return x_final.reshape(prior.interior_shape)


@functools.cache
def _create_optimizer() -> LBFGS:
    return LBFGS(fun=_objective)


@jit
@sparsify
def _objective(
    x: Array,
    prior_mean: Array,
    prior_precision: Array,
    obs_vals: Array,
    obs_idxs: Array,
    obs_noise: float,
) -> Array:
    term1 = ((obs_vals - x[obs_idxs]) ** 2).sum() / obs_noise
    term2 = (x - prior_mean).T @ (prior_precision @ (x - prior_mean))
    return term1 + term2
