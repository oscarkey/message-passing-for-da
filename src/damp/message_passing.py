"""Implements message passing.

This does not include multigrid, which is implemented on top of this in multigrid.py.
"""
from __future__ import annotations

from typing import Optional, Union

import chex
import jax.numpy as jnp
from jax import Array, vmap
from scipy.sparse import csr_matrix
from tqdm import tqdm

from damp.gp import Posterior
from damp.graph import Graph, Index
from damp.jax_utils import jit


@chex.dataclass(frozen=True)
class Edges:
    a: Array
    b: Array


@chex.dataclass(frozen=True)
class Marginals:
    mean: Array
    std: Array


@chex.dataclass(frozen=True)
class Config:
    graph: Graph
    c: Union[float, Graph]
    Gamma_diagonal: Array
    h: Array
    lr: float = 1.0

    def __post_init__(self) -> None:
        assert self.Gamma_diagonal.ndim == 1
        assert self.h.ndim == 1

        # This assertion is disabled for now because it was making things slow.
        # if isinstance(self.c, Stencil):
        #     chex.assert_trees_all_close(self.graph.mask, self.c.mask)
        if isinstance(self.c, Array):
            assert self.c.shape == (), "If c isn't a graph must be a scalar"

    @staticmethod
    def from_posterior(
        posterior: Posterior, c: Union[float, Graph], lr: float
    ) -> Config:
        """Creates a config for the graph represented by the given GP posterior."""
        # Remove the diagonal so we don't get self loops in the graph.
        precision_without_diagonal = csr_matrix(posterior.precision)
        precision_without_diagonal.setdiag(0.0)
        graph = Graph.from_scipy_precision(precision_without_diagonal)

        return Config(
            graph=graph,
            Gamma_diagonal=jnp.array(posterior.precision.diagonal()),
            h=jnp.array(posterior.shift),
            c=c,
            lr=lr,
        )


def get_initial_edges(graph: Graph) -> Edges:
    return Edges(a=graph.init_edges(value=0.0), b=graph.init_edges(value=1e-8))


def send_message(cfg: Config, edges: Edges, i: Index, j: Index) -> Edges:
    c_graph = _get_c_graph(cfg)

    # These equations are taken from the paper
    # "Message-Passing Algorithms for Quadratic Minimization"; Ruozzi 2013
    a_ji = cfg.graph.get_edge(edges.a, j, i)
    b_ji = cfg.graph.get_edge(edges.b, j, i)

    A_ij = c_graph.weighted_sum_incoming_edges(edges.a, i)
    A_ij += cfg.Gamma_diagonal[i]
    A_ij -= a_ji

    B_ij = c_graph.weighted_sum_incoming_edges(edges.b, i)
    B_ij += cfg.h[i]
    B_ij -= b_ji

    Gamma_ij_over_c_ij = cfg.graph.get_weight(i, j) / c_graph.get_weight(i, j)
    a_ij = -(Gamma_ij_over_c_ij**2) / A_ij
    b_ij = -(B_ij * Gamma_ij_over_c_ij) / A_ij

    return Edges(a=a_ij, b=b_ij)


def _get_c_graph(cfg: Config) -> Graph:
    # We may either have a single shared c, or a separate c for every edge in the
    # stencil. If the former, convert it into the latter for convenience.
    if isinstance(cfg.c, float) or (isinstance(cfg.c, Array) and cfg.c.shape == ()):
        return cfg.graph.duplicate_with_constant_weights(jnp.array(cfg.c))
    else:
        return cfg.c


@jit
def send_all_messages_parallel(c: Config, edges: Edges) -> Edges:
    def message_neighbours(j: Index) -> Edges:
        idxs = c.graph.get_neighour_indices(j)
        messages = vmap(lambda i: send_message(c, edges, i, j))(idxs)
        return Edges(a=messages.a, b=messages.b)

    new_edges = vmap(message_neighbours)(jnp.arange(0, edges.a.shape[0]))
    return Edges(
        a=(1.0 - c.lr) * edges.a + c.lr * new_edges.a,
        b=(1.0 - c.lr) * edges.b + c.lr * new_edges.b,
    )


@jit
def _extract_marginals(c: Config, edges: Edges) -> Marginals:
    return vmap(lambda i: _extract_marginal(c, edges, i))(
        jnp.arange(0, edges.a.shape[0])
    )


def _extract_marginal(cfg: Config, edges: Edges, i: Index) -> Marginals:
    c = _get_c_graph(cfg)
    precision = cfg.Gamma_diagonal[i] + c.weighted_sum_incoming_edges(edges.a, i)
    shift = cfg.h[i] + c.weighted_sum_incoming_edges(edges.b, i)

    return Marginals(mean=shift / precision, std=jnp.sqrt(1 / precision))


def iterate(
    c: Config,
    initial_edges: Edges,
    n_iterations: int,
    early_stopping_threshold: Optional[float] = None,
    progress_bar: bool = True,
) -> tuple[Edges, Marginals]:
    """Performs several iterations of message passing, starting at initial_edges.

    At each iteration, we send messages on every edge in the graph.

    :param n_iterations: the maximum number of iterations that will be performed. Fewer
                         may be performed if early_stopping_threshold is not None
    :param early_stopping_threshold: if not None, enables early stopping
                                     (see iterate_with_history() for how the criterion
                                      works)
    :returns: the edges and marginals after the final iteration
    """
    history = iterate_with_history(
        c,
        initial_edges,
        n_iterations,
        save_every=-1,
        early_stopping_threshold=early_stopping_threshold,
        progress_bar=progress_bar,
    )
    _, final_edges, final_nodes = history[-1]
    return final_edges, final_nodes


def iterate_with_history(
    c: Config,
    initial_edges: Edges,
    n_iterations: int,
    early_stopping_threshold: Optional[float] = None,
    save_every: int = 1,
    progress_bar: bool = True,
) -> list[tuple[int, Edges, Marginals]]:
    """Runs message passing, returning the edges/marginals every save_every iters.

    :returns: a list of tuples
              (iteration, edges at that iteration, marginals at that iteration)
    """
    edge_history = [(0, initial_edges)]
    edges = initial_edges
    initial_delta = None
    iterator = iter(tqdm(range(n_iterations))) if progress_bar else range(n_iterations)
    for i in iterator:
        new_edges = send_all_messages_parallel(c, edges)

        # We only check for early stopping every now and then to avoid slowing down the
        # process too much.
        if early_stopping_threshold is not None and i % 50 == 0:
            delta_magnitude = _get_delta_magnitude(edges, new_edges)
            if initial_delta is None:
                initial_delta = delta_magnitude
            should_early_stop = bool(
                delta_magnitude < (initial_delta * early_stopping_threshold)
            )
        else:
            should_early_stop = False
        last_iteration = should_early_stop or (i + 1 == n_iterations)

        if (save_every > 0 and (i + 1) % save_every == 0) or last_iteration:
            edge_history.append((i + 1, new_edges))

        edges = new_edges

        if should_early_stop:
            break

    result = [(i, e, _extract_marginals(c, e)) for i, e in edge_history]
    chex.block_until_chexify_assertions_complete()
    return result


@jit
def _get_delta_magnitude(previous: Edges, new: Edges) -> Array:
    # There are nans in the graph data structure which do not correspond to actual
    # edges. Thus ignore these when taking the mean.
    return jnp.maximum(
        jnp.nanmean(jnp.abs(previous.a - new.a)).mean(),
        jnp.nanmean(jnp.abs(previous.b - new.b)).mean(),
    )
