"""Exercise the two stand-alone launchers against the installed package."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize(
    ("variable", "reference_file"),
    [("PRCP", "rainfall_predictand.nc"), ("TEMP", "temp_predictand.nc")],
)
def test_standalone_launcher_runs_pbc_and_resumes(tmp_path, variable, reference_file):
    project = Path(__file__).resolve().parents[1]
    launcher_name = f"Cli_{variable}_WA_scientific_v8_clean_numba_010926.py"
    launcher = shutil.copy(project / launcher_name, tmp_path)
    workdir = tmp_path / "forecast products"
    intermediate = workdir / "intermediate"
    intermediate.mkdir(parents=True)

    times = pd.to_datetime([f"{year}-07-01" for year in range(1990, 2012)])
    coordinates = {"T": times, "Y": [10.0], "X": [2.0]}
    signal = np.arange(len(times)) + (22.0 if variable == "TEMP" else 200.0)
    observations = xr.DataArray(signal[:, None, None], dims=("T", "Y", "X"),
                                coords=coordinates, name=variable)
    observations.isel(T=slice(None, -1)).to_netcdf(workdir / reference_file)
    probs = xr.full_like(observations, 1 / 3).expand_dims(probability=["PB", "PN", "PA"])
    probs.isel(T=slice(None, -1)).rename("hdcst_prob_example").to_netcdf(
        intermediate / "example_hindcasts.nc")
    probs.isel(T=[-1]).rename("fcst_prob_example").to_netcdf(
        intermediate / "example_forecasts.nc")

    command = [
        sys.executable, launcher, "--workdir", str(workdir),
        "--only", "pbc", "--pbc-no-persistence", "--pbc-min-samples", "3",
        "--season", "JAS", "--init-month", "04", "--forecast-year", "2011",
        "--obs-years", "1990", "2010", "--clim-years", "1990", "2010",
        "--hindcast-years", "1990", "2010", "--no-dask", "--ncores", "1",
    ]
    subprocess.run(command, cwd=tmp_path, check=True)
    checkpoint = workdir / "checkpoints" / "chk_pbc.json"
    outputs = json.loads(checkpoint.read_text(encoding="utf-8"))["outputs"]
    result_path = next(workdir / path for path in outputs if Path(path).name.startswith("PBC_example_"))
    with xr.open_dataset(result_path) as result:
        assert result.attrs["source_variable"] == variable
        assert np.isfinite(result.forecast_probability).all().item()
        np.testing.assert_allclose(result.forecast_probability.sum("probability"), 1.0)

    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "matching checkpoint exists" in second.stdout
