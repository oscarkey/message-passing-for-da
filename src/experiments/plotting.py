"""Utilities for creating plots ready for the paper."""
from pathlib import Path

import matplotlib.pyplot as plt

# One mm in inches, to convert to inches as required by matplotlib.
mm = 0.0393701

# These are based on the Climate Informatics template.
FULL_WIDTH = 144 * mm
HALF_WIDTH = FULL_WIDTH * 0.49

squashed_legend_params = {
    "handlelength": 1.0,
    "handletextpad": 0.5,
    "labelspacing": 0.3,
    "borderaxespad": 0.2,
    "borderpad": 0.25,
    "columnspacing": 0.7,
}
squashed_label_params = {"labelpad": 1.5}


def configure_matplotlib() -> None:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
    plt.rc("font", family="serif")


def save_fig(name: str) -> None:
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / f"{name}.png", dpi=200)
    plt.savefig(plot_dir / f"{name}.pdf", transparent=True)
