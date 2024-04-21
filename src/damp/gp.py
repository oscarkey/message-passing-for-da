from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from findiff import PDE, BoundaryConditions, Coefficient, FinDiff, Identity
from numpy import ndarray
from numpy.random import Generator
from scipy.sparse import diags, spmatrix
from scipy.special import gamma

Obs = list[tuple[tuple[int, int], ndarray]]


class Shape(NamedTuple):
    width: int
    height: int

    def flatten(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Prior:
    d: int
    ls: float
    nu: float
    amp: float
    shape: Shape
    precision: spmatrix
    precision_decomposed: spmatrix
    grid_idxs: ndarray
    interior_idxs: ndarray

    @property
    def interior_shape(self) -> Shape:
        return Shape(self.shape.width - 2, self.shape.height - 2)

    @property
    def name(self) -> str:
        return f"size_{self.shape.width}_{self.shape.height}"


@dataclass(frozen=True)
class Posterior:
    shift: ndarray
    precision: spmatrix
    obs_noise: float
    obs_location_mask: spmatrix


def get_prior(shape: Shape) -> Prior:
    d = 2
    ls = 0.15
    nu = 1
    amp = 1.1
    x = np.linspace(0, 1, shape.width)
    y = np.linspace(0, 1, shape.height)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Set LHS
    diff_op = _kappa(nu, ls) ** 2 * Identity() - FinDiff(0, dx, 2) - FinDiff(1, dy, 2)

    # Construct matern-1 precision matrix
    mat = _operator_to_matrix(diff_op, shape)
    precision_decomposed = np.sqrt(dx * dy / (_q(d, nu, ls) * amp**2)) * mat
    precision = precision_decomposed.T @ precision_decomposed

    return Prior(
        d,
        ls,
        nu,
        amp,
        shape,
        precision,
        precision_decomposed,
        interior_idxs=_get_interior_indices(shape),
        grid_idxs=_get_domain_indices(shape),
    )


def get_prior_sphere(shape: Shape, lon: ndarray, lat: ndarray) -> Prior:
    d = 2
    ls = 0.2
    nu = 1
    amp = 1.9
    # Shifting the undefined region to the north pole.
    # lat now goes from 0 -> 180 deg
    lat = lat + 90
    phi = np.radians(lat)
    theta = np.radians(lon)
    dtheta, dphi = theta[1] - theta[0], phi[1] - phi[0]

    Theta, Phi = np.meshgrid(theta, phi)
    # Set LHS
    diff_op = (
        _kappa(nu, ls) ** 2 * Identity()
        - Coefficient(1 / np.tan(Phi)) * FinDiff(0, dphi)
        - FinDiff(0, dphi, 2)
        - Coefficient((1 / np.sin(Phi)) ** 2) * FinDiff(1, dtheta, 2)
    )
    # Construct matern-1 precision matrix
    mat = _operator_to_matrix(diff_op, shape)
    # Extract the interior Phi values
    # Used to scale the precision
    Phi_interior = Phi[1:-1, 1:-1]
    Phi_interior = Phi_interior.flatten()
    Phi_interior = diags(Phi_interior)

    precision_decomposed = (
        np.sqrt((np.sin(Phi_interior) * dtheta * dphi) / (_q(d, nu, ls) * amp**2))
        * mat
    )
    precision = precision_decomposed.T @ precision_decomposed
    return Prior(
        d,
        ls,
        nu,
        amp,
        shape,
        precision,
        precision_decomposed,
        interior_idxs=_get_interior_indices(shape),
        grid_idxs=_get_domain_indices(shape),
    )


def sample_prior(rng: Generator, prior: Prior) -> ndarray:
    x = np.linspace(0, 1, prior.shape.width)
    y = np.linspace(0, 1, prior.shape.height)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Set LHS
    kappa = _kappa(prior.nu, prior.ls)
    diff_op = kappa**2 * Identity() - FinDiff(0, dx, 2) - FinDiff(1, dy, 2)

    # Set RHS
    const = (dx * dy) ** (-0.5) * np.sqrt(_q(prior.d, prior.nu, prior.ls)) * prior.amp
    W = const * rng.normal(size=prior.shape)

    # Set boundary conditions (zero-Dirichlet)
    bc = BoundaryConditions(prior.shape)
    bc[0, :] = 0
    bc[-1, :] = 0
    bc[:, 0] = 0
    bc[:, -1] = 0

    # Solve PDE
    pde = PDE(diff_op, W, bc)
    return pde.solve()


def get_posterior(
    prior: Prior, observations: Obs, obs_noise: float = 1e-3
) -> Posterior:
    shape = prior.grid_idxs.shape

    N = np.prod(shape)
    mask = np.zeros(N)
    for idx, _ in observations:
        mask[prior.grid_idxs[idx]] = 1
    posterior_precision = prior.precision + obs_noise ** (-2) * diags(
        mask[prior.interior_idxs]
    )

    posterior_shift = np.zeros(np.prod(shape))
    for idx, observation in observations:
        posterior_shift[prior.grid_idxs[idx]] = observation / obs_noise**2
    posterior_shift = posterior_shift[prior.interior_idxs]

    return Posterior(
        posterior_shift,
        posterior_precision,
        obs_noise,
        obs_location_mask=mask[prior.interior_idxs],
    )


def _get_domain_indices(shape):
    siz = np.prod(shape)
    full_indices = np.array(list(range(siz))).reshape(shape)
    return full_indices


def _get_interior_indices(shape) -> ndarray:
    full_indices = _get_domain_indices(shape)
    interior_slice = tuple(slice(1, -1) for _ in range(len(shape)))
    interior_indices = full_indices[interior_slice].flatten()
    return interior_indices


def _operator_to_matrix(diff_op, shape):
    """
    Convert a findiff operator into a precision matrix
    """
    mat = diff_op.matrix(shape)
    interior_idxs = _get_interior_indices(shape)
    mat = mat[interior_idxs]
    mat = mat[:, interior_idxs]
    return mat


def _kappa(nu: float, ls: float) -> float:
    return np.sqrt(2 * nu) / ls


def _q(d: int, nu: float, ls: float) -> float:
    return (
        (4 * np.pi) ** (d / 2) * _kappa(nu, ls) ** (2 * nu) * gamma(nu + d / 2)
    ) / gamma(nu)


def choose_observations(
    rng: Generator, n_obs: int, ground_truth: ndarray, obs_noise: float
) -> Obs:
    x_idxs = np.arange(ground_truth.shape[0])
    y_idxs = np.arange(ground_truth.shape[1])
    X_idxs, Y_idxs = np.meshgrid(x_idxs[1:-1], y_idxs[1:-1], indexing="ij")
    all_idxs = np.stack([X_idxs.flatten(), Y_idxs.flatten()], axis=1)
    idxs = rng.choice(all_idxs, n_obs, replace=False)
    return [((x, y), ground_truth[(x, y)] + obs_noise * rng.normal()) for x, y in idxs]
