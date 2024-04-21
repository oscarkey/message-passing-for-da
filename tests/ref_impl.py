"""Reference implementation of the message passing algorithm using networkx."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
from tqdm import tqdm


@dataclass
class State:
    precision: float = 0.0
    shift: float = 1e-8


class FactorGraphFromArray(nx.DiGraph):
    def __init__(self, Gamma, h=None, c=None):
        # Store Gamma, h and c
        self.Gamma = Gamma
        self.h = np.zeros(len(Gamma)) if h is None else h
        self.c = 1.0 if c is None else c

        # Initialise factor graph from Gamma matrix
        super().__init__()
        try:
            G = nx.from_scipy_sparse_array(Gamma, create_using=nx.DiGraph)
        except:
            G = nx.from_numpy_array(Gamma, create_using=nx.DiGraph)
        G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
        self.add_nodes_from(G)
        self.add_edges_from(G.edges)

        # Initialise node states
        nx.set_node_attributes(self, State(), "state")

        # Initialise edge states
        nx.set_edge_attributes(self, State(), "message")

    def send_message(self, i, j):
        msg_ji = self.edges[(j, i)]["message"]

        a_ji = msg_ji.precision
        b_ji = msg_ji.shift

        # Compute variable-to-factor message
        A_ij = self.c * np.sum(
            [self.edges[(k, i)]["message"].precision for k in self.neighbors(i)]
        )
        A_ij += self.Gamma[i, i]
        A_ij -= a_ji

        B_ij = self.c * np.sum(
            [self.edges[(k, i)]["message"].shift for k in self.neighbors(i)]
        )
        B_ij += self.h[i]
        B_ij -= b_ji

        # Compute factor-to-variable message
        a_ij = -((self.Gamma[i, j] / self.c) ** 2) / A_ij
        b_ij = -(B_ij * self.Gamma[i, j]) / (self.c * A_ij)

        # Update message
        self.edges[(i, j)]["message"] = State(a_ij, b_ij)

    def update_state(self, i):
        # Update state at node i
        a = self.Gamma[i, i]
        b = self.h[i]
        for j in self.neighbors(i):
            a += self.c * self.edges[(j, i)]["message"].precision
            b += self.c * self.edges[(j, i)]["message"].shift
        self.nodes[i]["state"] = State(a, b)

    def send_all_messages(self, i):
        # Send messages to all neighbours of node i
        for j in self.neighbors(i):
            self.send_message(i, j)

    def iterate_message_passing(self, num_iter):
        for _ in tqdm(range(num_iter)):
            for i in self.nodes:
                self.send_all_messages(i)
            for i in self.nodes:
                self.update_state(i)

    @property
    def states(self):
        return dict([(x, self.nodes[x]["state"]) for x in self.nodes])

    @property
    def means(self):
        return np.array([s.shift / s.precision for s in self.states.values()])

    @property
    def stds(self):
        return np.array([np.sqrt(s.precision ** (-1)) for s in self.states.values()])
