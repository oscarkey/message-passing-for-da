from __future__ import annotations

from abc import ABC
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import register_pytree_node_class
from scipy.sparse import csr_matrix, spmatrix

Scalar = Array
Index = Union[Array, int]

nan = jnp.array(jnp.nan)


@register_pytree_node_class
class Graph(ABC):
    """A directed symmetric graph where each edge can have a value and a weight.

    Each edge has the same weight in both directions, but can have a different value in
    each direction.

    It is optimized for specific operations:
       - graphs were most of the nodes have a similar degree
       - summing the values incoming into a node
    """

    @staticmethod
    def from_scipy_precision(precision: spmatrix) -> Graph:
        assert is_precision_symmetric(precision)

        p: csr_matrix = precision.tocsr()
        p.eliminate_zeros()

        n_nodes = p.shape[0]
        max_degree = np.unique(p.nonzero()[0], return_counts=True)[1].max()
        connectivity = np.full((n_nodes, max_degree), fill_value=-1)
        weights = np.full((n_nodes, max_degree), fill_value=0.0)
        for j_idx in range(n_nodes):
            row_start = p.indptr[j_idx]
            row_end = p.indptr[j_idx + 1]
            incoming_indices = p.indices[row_start:row_end]
            incoming_weights = p.data[row_start:row_end]
            connectivity[j_idx, 0 : len(incoming_indices)] = incoming_indices
            weights[j_idx, 0 : len(incoming_weights)] = incoming_weights

        return Graph(jnp.array(connectivity), jnp.array(weights))

    def __init__(self, connectivity: Array, weights: Array) -> None:
        """Creates a new instance.

        :param connectivity: [n x l] where n is the number of nodes and l is the maximum
                        degree of any node. Row j gives the indices of the nodes that
                        are connected to node j. If fewer than l nodes are connected to
                        j, then the rest of the row is padded with -1.
        :param weights: [n x l] where n is the number of nodes and l is the maximum
                        degree of any node. This gives the weights corresponding to the
                        edges defined in connectively, or is 0.0 at positions where no
                        edge exists.
        """
        assert connectivity.ndim == 2
        assert weights.shape == connectivity.shape
        self.connectivity = connectivity
        self.weights = weights

    def init_edges(self, value: float) -> Array:
        return jnp.full_like(self.weights, value)

    def get_neighour_indices(self, i: Index) -> Array:
        """Returns the indices of the nodes connected to i.

        The returned array is always of length l. If i has fewer neighbours than this
        the rest of the array is padded with -1.
        """
        return self.connectivity[i]

    def sum_incoming_edges(self, edges: Array, j: Index) -> Array:
        """Returns the sum of the values on all the edges coming into j."""
        return jnp.where(self.connectivity[j] != -1, edges[j], 0.0).sum()

    def weighted_sum_incoming_edges(self, edges: Array, j: Index) -> Array:
        """Returns the weighted sum of the values on all the edges coming into j.

        The edges are multiplied by their weight, before being summed.
        """
        return jnp.where(
            self.connectivity[j] != -1, edges[j] * self.weights[j], 0.0
        ).sum()

    def get_edge(self, edges: Array, i: Index, j: Index) -> Array:
        edge_exists = self._on_graph(i, j) & jnp.any(self.connectivity[j] == i)
        return lax.cond(
            edge_exists,
            lambda: jnp.where(self.connectivity[j] == i, edges[j], 0.0).sum(),
            lambda: nan,
        )

    def set_edge(self, edges: Array, i: Index, j: Index, value: Scalar) -> Array:
        new_edges = jnp.where(self.connectivity[j] == i, value, edges[j])
        return edges.at[j].set(new_edges)

    def get_weight(self, i: Index, j: Index) -> Scalar:
        return jnp.where(self.connectivity[j] == i, self.weights[j], 0.0).sum()

    def get_all_edges(self, edges: Array) -> Array:
        """Returns an Array containing all the edges that actually exist."""
        return edges.flatten()[self.connectivity.flatten() != -1]

    def has_bad_nans(self, edges: Array) -> bool:
        """Returns True if `edges` has nans at indices corresponding to actual edges."""
        return jnp.any(jnp.isnan(self.get_all_edges(edges))).item()

    def _on_graph(self, *idxs: Index) -> Array:
        on_graph = [(idx >= 0) & (idx < self.connectivity.shape[0]) for idx in idxs]
        return jnp.all(jnp.stack(on_graph))

    def duplicate_with_constant_weights(self, weight: Array) -> Graph:
        new_weights = jnp.where(self.weights != 0.0, weight, 0.0)
        return Graph(self.connectivity, new_weights)

    def tree_flatten(self):
        children = (self.connectivity, self.weights)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def is_precision_symmetric(p: spmatrix) -> bool:
    return p.shape[0] == p.shape[1] and (p != p.T).nnz == 0
