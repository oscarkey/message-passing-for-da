"""Calls an external R installation to run R-INLA."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import rpy2.robjects as ro
from numpy import ndarray
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from damp import gp


def run(prior: gp.Prior, obs: gp.Obs) -> tuple[ndarray, ndarray]:
    r_Matrix = importr("Matrix")
    importr("INLA")

    with TemporaryDirectory(prefix="damp_r_communication_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        py_pp = prior.precision.tocoo()
        ro.globalenv["prior_precision"] = r_Matrix.sparseMatrix(
            i=ro.IntVector(py_pp.row + 1),
            j=ro.IntVector(py_pp.col + 1),
            x=ro.FloatVector(py_pp.data),
            dims=ro.IntVector(py_pp.shape),
        )

        data = ro.r["read.csv"](str(_write_data(prior, obs, temp_dir)))

        formula = ro.r(
            f'y ~ f(x, model = "generic0", Cmatrix = prior_precision, hyper = list(prec = list( initial = 1e-3, fixed = TRUE)))'
        )
        ro.globalenv["result"] = ro.r["inla"](formula, data=data)

        r_results = ro.r("result$summary.random$x")
        with (ro.default_converter + pandas2ri.converter).context():
            results = ro.conversion.get_conversion().rpy2py(r_results)

    pred_means = results["mean"].to_numpy().reshape(prior.interior_shape)
    pred_stds = results["sd"].to_numpy().reshape(prior.interior_shape)
    return pred_means, pred_stds


def _write_data(prior: gp.Prior, obs: gp.Obs, temp_dir: Path) -> Path:
    output_path = temp_dir / "data.csv"
    vals_by_idx = {_convert_xy_to_inla_idx(prior, xy): val for xy, val in obs}
    with open(output_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        # The INLA idx start at 1, so remember to set the range appropriately.
        # Oscar: is there a reason we don't just iterate over the dictionary?
        for idx in range(1, prior.precision.shape[0] + 1):
            if idx in vals_by_idx:
                writer.writerow([idx, vals_by_idx[idx]])
    return output_path


def _convert_xy_to_inla_idx(prior: gp.Prior, xy: tuple[int, int]) -> int:
    x, y = xy
    # x and y are in boundary coordinates -> convert to interior coordinates
    x = x - 1
    y = y - 1
    # Given a field Z of size [width x height], we store the point (x,y) at Z[x,y]
    # which is the xth column and yth row.
    zero_indexed_index = (x * prior.interior_shape.height) + y
    # Finally, add one to convert from Python zero-indexed arrays to R's one-indexed
    # arrays.
    return zero_indexed_index + 1
