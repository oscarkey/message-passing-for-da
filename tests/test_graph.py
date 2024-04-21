import random
from collections import defaultdict

import jax.numpy as jnp
import networkx

from damp.graph import Graph


def test__init_edges__edges_have_correct_value() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.2)
    assert graph.get_edge(edges, to_idx(1, 1), to_idx(1, 0)) == 1.2


def test__get_edge__j_or_j_outside_of_graph__returns_nan() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)

    assert jnp.isnan(graph.get_edge(edges, i=0, j=-1))
    assert jnp.isnan(graph.get_edge(edges, i=24, j=25))
    assert jnp.isnan(graph.get_edge(edges, i=-1, j=0))
    assert jnp.isnan(graph.get_edge(edges, i=25, j=24))


def test__get_edge__i_j_not_connected__returns_nan() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)

    assert jnp.isnan(graph.get_edge(edges, i=to_idx(0, 0), j=to_idx(0, 2)))
    assert jnp.isnan(graph.get_edge(edges, i=to_idx(0, 0), j=to_idx(1, 1)))
    assert jnp.isnan(graph.get_edge(edges, i=to_idx(2, 2), j=to_idx(0, 4)))


def test__set_get_edge__returns_correct_value_for_every_edge() -> None:
    random.seed(29087)
    _, graph = create_test_graph()

    edges = graph.init_edges(value=0.0)
    values: dict[int, dict[int, float]] = defaultdict(lambda: {})
    for ix in range(5):
        for iy in range(5):
            for jx, jy in [
                (ix - 1, iy),
                (ix + 1, iy),
                (ix, iy - 1),
                (ix, iy + 1),
            ]:
                if ix < 0 or ix >= 5 or iy < 0 or iy >= 5:
                    continue
                if jx < 0 or jx >= 5 or jy < 0 or jy >= 5:
                    continue
                i = to_idx(ix, iy)
                j = to_idx(jx, jy)
                if i == 4 and j == 5:
                    pass
                val = random.uniform(-1, 1)
                values[i][j] = val
                edges = graph.set_edge(edges, i, j, jnp.array(val))

    for i, js in values.items():
        for j, expected in js.items():
            if i == 4 and j == 5:
                pass
            value = graph.get_edge(edges, i, j)
            assert value == expected, f"failed at {i}->{j}: {value:.3f} {expected:.3f}"


def test__get_weight__returns_correct_value() -> None:
    _, graph = create_test_graph(weights={((0, 0), (0, 1)): 3.0})
    assert graph.get_weight(to_idx(0, 0), to_idx(0, 1)) == 3.0
    assert graph.get_weight(to_idx(0, 1), to_idx(0, 0)) == 3.0
    assert graph.get_weight(to_idx(1, 1), to_idx(1, 2)) == 1.0


def test__sum_incoming_edges__set_values__returns_correct_value() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)
    edges = graph.set_edge(edges, to_idx(0, 1), to_idx(0, 0), jnp.array(3.0))
    edges = graph.set_edge(edges, to_idx(1, 0), to_idx(0, 0), jnp.array(-1.5))
    edges = graph.set_edge(edges, to_idx(2, 3), to_idx(2, 2), jnp.array(6.0))

    # (0, 0) has two incoming edges specified above, which sum to 1.5.
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(0, 0)), 1.5)
    # (2, 2) has one incoming edge specified above, plus three with the initial value
    # 1.0.
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(2, 2)), 9.0)


def test__sum_incoming_edges__just_initial_values__returns_correct_value() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)

    # Check nodes at the edges, and also in the middle.
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(0, 0)), 2.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(2, 0)), 3.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(4, 0)), 2.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(4, 2)), 3.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(4, 4)), 2.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(2, 4)), 3.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(0, 4)), 2.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(0, 2)), 3.0)
    assert jnp.allclose(graph.sum_incoming_edges(edges, to_idx(2, 2)), 4.0)


def test__sum_incoming_edges__nan_on_non_existant_edge__does_not_affect_sum() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)
    edges = edges.at[0, 3].set(jnp.nan)
    edges = graph.set_edge(edges, to_idx(0, 1), to_idx(0, 0), value=jnp.array(10.0))

    # Two edges are on the graph, one with default value 1.0 and one with value set to
    # 10.0. Hence the total is 11.0.
    assert graph.sum_incoming_edges(edges, to_idx(0, 0)) == 11.0


def test__sum_incoming_edges__nan_on_edge__sum_is_nan() -> None:
    _, graph = create_test_graph()
    edges = graph.init_edges(value=1.0)
    edges = graph.set_edge(edges, to_idx(0, 1), to_idx(0, 0), value=jnp.array(jnp.nan))

    assert jnp.isnan(graph.sum_incoming_edges(edges, to_idx(0, 0)))


def test__get_potential_neighbours__returns_correct_values() -> None:
    _, graph = create_test_graph()

    idxs = graph.get_neighour_indices(to_idx(0, 0))
    assert jnp.allclose(idxs, jnp.array([to_idx(1, 0), to_idx(0, 1), -1, -1]))


def create_test_graph(
    weights: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
) -> tuple[networkx.Graph, Graph]:
    ref_graph = networkx.grid_2d_graph(range(5), range(5))

    # networkx has x and y the other way round to us.
    weights = {((k[0][1], k[0][0]), (k[1][1], k[1][0])): v for k, v, in weights.items()}
    for nodes, weight in weights.items():
        ref_graph.add_edge(*nodes, weight=weight)

    our_graph = Graph.from_scipy_precision(
        networkx.to_scipy_sparse_array(ref_graph).astype(float)
    )
    return ref_graph, our_graph


def to_idx(x: int, y: int) -> int:
    return y * 5 + x
