import chex
import jax.numpy as jnp
import ref_impl
from numpy.random import default_rng

from damp import gp, message_passing
from damp.gp import Shape
from damp.graph import Graph
from damp.jax_utils import jit
from damp.message_passing import Config, Edges


def test__send_message__equal_to_reference_impl_over_entire_graph() -> None:
    config, edges, ref_graph = set_up_graph()

    send_message = jit(message_passing.send_message)
    set_edge = jit(config.graph.set_edge)

    for i in ref_graph.nodes:
        for j in ref_graph.neighbors(i):
            m = send_message(config, edges, i, j)
            edges = Edges(
                a=set_edge(edges.a, i, j, m.a), b=set_edge(edges.b, i, j, m.b)
            )

            ref_graph.send_message(i, j)
            expected = ref_graph.edges[(i, j)]["message"]

            assert jnp.allclose(m.a, expected.precision, atol=1e-9)
            assert jnp.allclose(m.b, expected.shift, atol=1e-9)


def test__send_message__i_outside_grid__returns_nan() -> None:
    config, initial_edges, _ = set_up_graph()
    send_message = message_passing.send_message

    message = send_message(config, initial_edges, i=to_idx(-1, 0), j=to_idx(0, 0))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)
    message = send_message(config, initial_edges, i=to_idx(0, -1), j=to_idx(0, 0))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)
    message = send_message(config, initial_edges, i=to_idx(10, 10), j=to_idx(9, 9))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)


def test__send_message__j_outside_grid__returns_nan() -> None:
    config, initial_edges, _ = set_up_graph()
    send_message = message_passing.send_message

    message = send_message(config, initial_edges, i=to_idx(0, 0), j=to_idx(-1, 0))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)
    message = send_message(config, initial_edges, i=to_idx(0, 0), j=to_idx(12, 0))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)


def test__send_message__i_j_not_neighbours__returns_nan() -> None:
    config, initial_edges, _ = set_up_graph()
    send_message = message_passing.send_message

    message = send_message(config, initial_edges, i=to_idx(0, 0), j=to_idx(0, 3))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)
    message = send_message(config, initial_edges, i=to_idx(0, 0), j=to_idx(2, 2))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)
    message = send_message(config, initial_edges, i=to_idx(4, 4), j=to_idx(6, 6))
    assert jnp.isnan(message.a)
    assert jnp.isnan(message.b)


def test__send_message__multiple_cs__messages_depends_on_c() -> None:
    config, initial_edges, _ = set_up_graph()
    send_message = message_passing.send_message

    c_graph = config.graph.duplicate_with_constant_weights(weight=jnp.array(1.0))
    c_graph.weights = c_graph.weights.at[45, 0].set(2.0)
    config = config.replace(c=c_graph)

    message_1 = send_message(config, initial_edges, i=to_idx(5, 5), j=to_idx(5, 6))
    message_2 = send_message(config, initial_edges, i=to_idx(5, 5), j=to_idx(6, 5))
    message_3 = send_message(config, initial_edges, i=to_idx(5, 5), j=to_idx(5, 3))

    assert jnp.allclose(message_1.b, message_2.b)
    assert jnp.allclose(message_1.a, message_2.a)
    assert not jnp.allclose(message_3.a, message_2.a)
    assert not jnp.allclose(message_3.a, message_2.a)


def test__send_all_messages_parallel__result_has_correct_shape() -> None:
    config, initial_edges, _ = set_up_graph()
    send_all_messages_parallel = message_passing.send_all_messages_parallel

    next_edges = send_all_messages_parallel(config, initial_edges)

    assert next_edges.a.shape == initial_edges.a.shape
    assert next_edges.b.shape == initial_edges.b.shape


def test__iterate__parallel__result_has_correct_shape() -> None:
    config, initial_edges, _ = set_up_graph()
    n_points = initial_edges.a.shape[0]

    _, marginals1 = message_passing.iterate(config, initial_edges, n_iterations=0)
    assert marginals1.mean.shape == (n_points,)
    assert marginals1.std.shape == (n_points,)

    _, marginals2 = message_passing.iterate(config, initial_edges, n_iterations=1)
    assert marginals2.mean.shape == (n_points,)
    assert marginals2.std.shape == (n_points,)


def test__iterate__multiple_cs__result_has_correct_shape() -> None:
    config, initial_edges, _ = set_up_graph()
    n_points = initial_edges.a.shape[0]

    c = config.graph.weights
    c = c.at[config.graph.connectivity != -1].set(1.0)
    c = c.at[1, 2].set(2.0)
    c = c.at[2, 1].set(2.0)
    config = config.replace(c=Graph(config.graph.connectivity, weights=c))

    _, marginals1 = message_passing.iterate(config, initial_edges, n_iterations=0)
    assert marginals1.mean.shape == (n_points,)
    assert marginals1.std.shape == (n_points,)

    _, marginals2 = message_passing.iterate(config, initial_edges, n_iterations=1)
    assert marginals2.mean.shape == (n_points,)
    assert marginals2.std.shape == (n_points,)


def set_up_graph(
    grid_size: int = 10,
) -> tuple[message_passing.Config, Edges, ref_impl.FactorGraphFromArray]:
    numpy_rng = default_rng(seed=1124)
    c = -1.0

    prior = gp.get_prior(Shape(grid_size, grid_size))
    ground_truth = gp.sample_prior(numpy_rng, prior)
    obs = gp.choose_observations(
        numpy_rng, n_obs=5, ground_truth=ground_truth, obs_noise=1e-3
    )
    posterior = gp.get_posterior(prior, obs)

    config = Config.from_posterior(posterior, c, lr=1.0)
    initial_edges = message_passing.get_initial_edges(config.graph)
    ref_factor_graph = ref_impl.FactorGraphFromArray(
        posterior.precision, posterior.shift, c=c
    )

    return config, initial_edges, ref_factor_graph


def to_idx(x: int, y: int) -> int:
    interior_size = 8
    return y * interior_size + x


def teardown_function() -> None:
    chex.block_until_chexify_assertions_complete()
