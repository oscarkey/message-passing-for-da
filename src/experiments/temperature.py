"""Reproduces the experiments on spherical temperature data.

Unfortunately these experiments depend on data from the Met Office's Unified Model which
we are unable to include in this repository. Thus, this code is for reference only.
"""
from pathlib import Path
from sys import argv
from time import time
from typing import Union

import cartopy.crs as ccrs
import cmocean as cmo
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jax import Array
from numpy import ndarray
from numpy.random import Generator, default_rng

import plotting
from damp import gp, inla_bridge, metrics, multigrid, threedvar
from damp.gp import Obs, Shape


class Ours:
    def run(
        self,
        prior: gp.Prior,
        obs: gp.Obs,
        obs_noise: float,
        lats: ndarray,
        lons: ndarray,
    ) -> Array:
        ratios = [4, 2, 1]
        output = multigrid.run_sphere(
            prior, obs, obs_noise, lats, lons, ratios, c=-2.0, lr=0.7
        )
        level, marginals = output[-1]
        return marginals.mean


class INLA:
    def run(
        self,
        prior: gp.Prior,
        obs: gp.Obs,
        obs_noise: float,
        lats: ndarray,
        lons: ndarray,
    ) -> Array:
        mean, std = inla_bridge.run(prior, obs)
        return jnp.array(mean)


class ThreeDVar:
    def run(
        self,
        prior: gp.Prior,
        obs: gp.Obs,
        obs_noise: float,
        lats: ndarray,
        lons: ndarray,
    ) -> Array:
        return threedvar.run_optimizer(
            jax.random.key(seed=123456), prior, obs, obs_noise
        )


Method = Union[Ours, INLA, ThreeDVar]


def main(method_name: str) -> None:
    if method_name == "mp":
        method: Method = Ours()
    elif method_name == "3dvar":
        method = ThreeDVar()
    elif method_name == "inla":
        method = INLA()
    else:
        raise ValueError(f"Unknown method '{method_name}'")

    numpy_rng = default_rng(seed=1124)

    data_dir = Path("data")
    # Depending on how the script is invoked, the data path can vary.
    if not data_dir.exists():
        data_dir = Path("../../data")
    ground_truth = np.load(data_dir / "global_temp/UM_temp.npy")
    lons = np.load(data_dir / "global_temp/UM_lon.npy")
    lats = np.load(data_dir / "global_temp/UM_lat.npy")
    era5 = np.load(data_dir / "global_temp/ERA5_temp.npy")
    sat_lats = np.load(data_dir / "global_temp/satellites_lat.npy")
    sat_lons = np.load(data_dir / "global_temp/satellites_lon.npy")

    # By setting ratio > 1 we can subsample when debugging.
    ratio = 1
    ground_truth = ground_truth[::ratio, ::ratio]
    era5 = era5[::ratio, ::ratio]
    lons = lons[::ratio]
    lats = lats[::ratio]

    # Zero mean the truth based on the climatology mean.
    zero_mean_gt = ground_truth - era5

    obs_noise = 1e-3
    obs = _get_observations(
        numpy_rng, zero_mean_gt, lats, lons, sat_lats, sat_lons, obs_noise
    )

    width, length = np.shape(ground_truth)
    prior = gp.get_prior_sphere(Shape(width, length), lons, lats)

    start_time = time()
    pred_mean = method.run(prior, obs, obs_noise, lats, lons)
    end_time = time()
    print(f"Total Runtime = {end_time - start_time:.1f}")
    rmse = metrics.rmse(pred_mean, zero_mean_gt).item()
    print(f"RMSE = {rmse:.2f}")

    _plot_mean(method_name, era5, pred_mean, lats, lons, sat_lats, sat_lons)
    _plot_error(method_name, zero_mean_gt, pred_mean, lats, lons, sat_lats, sat_lons)


def _get_observations(
    numpy_rng: Generator,
    zero_mean_ground_truth: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
    obs_noise: float,
) -> Obs:
    def get():
        for lat, lon in zip(sat_lats, sat_lons, strict=True):
            x = np.argmin(abs(lat - lats))
            y = np.argmin(abs(lon - lons))
            gt = zero_mean_ground_truth[
                np.argmin(abs(lat - lats)), np.argmin(abs(lon - lons))
            ]
            value = gt + obs_noise * numpy_rng.normal()
            yield (x, y), value

    return list(get())


def _plot_mean(
    method_name: str,
    era5: np.ndarray,
    pred_mean: Array,
    lats: np.ndarray,
    lons: np.ndarray,
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
) -> None:
    temp = xr.DataArray(
        pred_mean + era5[1:-1, 1:-1],
        dims=["latitude", "longitude"],
        coords={"latitude": lats[1:-1], "longitude": lons[1:-1]},
    )
    temp = temp.rename("Temperature (K)")

    fig, ax = plt.subplots(
        subplot_kw=dict(projection=ccrs.Orthographic(0, 10), facecolor="gray")
    )
    fg = temp.plot(
        transform=ccrs.PlateCarree(),
        cmap="cmo.thermal",
        vmin=260,
        vmax=300,
        ax=ax,
        add_colorbar=False,
    )

    # Add coastlines, gridlines, and labels
    ax.coastlines()
    # ax.gridlines(draw_labels=True)
    ax.set_title("")

    cb = plt.colorbar(fg, orientation="vertical")
    cb.set_label(label="Temperature (K)", size=20)
    cb.ax.tick_params(labelsize=15)

    # Overlay lat, lon coordinates (customize as needed)
    ax.plot(
        sat_lons,
        sat_lats,
        "ko",
        markersize=0.05,
        transform=ccrs.PlateCarree(),
        label="Satellite tracks",
    )
    ax.legend(markerscale=100, loc=(0.11, -0.05), fontsize=20)

    plt.tight_layout()
    plotting.save_fig(f"temperature_mean_{method_name}")


def _plot_error(
    method_name: str,
    zero_mean_gt: ndarray,
    pred_mean: Array,
    lats: np.ndarray,
    lons: np.ndarray,
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
) -> None:
    error = xr.DataArray(
        zero_mean_gt[1:-1, 1:-1] - pred_mean,
        dims=["latitude", "longitude"],
        coords={"latitude": lats[1:-1], "longitude": lons[1:-1]},
    )
    fig = plt.figure()
    fig.add_subplot(projection=ccrs.Robinson())
    ax = plt.gca()
    pcm = error.plot(
        transform=ccrs.PlateCarree(),
        cmap="cmo.balance",
        ax=ax,
        add_colorbar=False,
    )

    cbar_ax = fig.add_axes([0.25, 0.0, 0.5, 0.01])
    # Add coastlines, gridlines, and labels
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel(r"$L_{1}$ error (K)", fontsize=10, labelpad=-40)
    cbar_ax.xaxis.set_ticks_position("top")
    plt.tight_layout()
    plotting.save_fig(f"temperature_error_{method_name}")


if __name__ == "__main__":
    if len(argv) != 2:
        raise ValueError("Usage: python src/experiments/temperature.py [mp,3dvar,inla]")
    main(argv[1])
