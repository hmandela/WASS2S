#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WAS_Hydro public hydrology interface.

Pure importable module: no CLI, no argparse, no main().

Scientific high-water regime detection lives in
``highwater_season_predictands_modeaware.py``. GloFAS/HydroBASINS matching,
bias correction, regionalization and independent evaluation live in
``glofas_highflow_bias_correction.py``. This module provides the WASS2S-facing
class and converts results to xarray and input-like DataFrame representations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
try:
    from .highflow_periods import ClimatologyPeriod, slice_series, slice_dataarray
    from .highflow_window_regionalization import GridWindowConfig, regionalize_highflow_windows
    from .highflow_topkriging import TopKrigingConfig
except ImportError:
    from highflow_periods import ClimatologyPeriod, slice_series, slice_dataarray
    from highflow_window_regionalization import GridWindowConfig, regionalize_highflow_windows
    from highflow_topkriging import TopKrigingConfig
# matplotlib and Cartopy are needed only by plot_country_subbasins.  They are
# imported lazily inside that method so that importing WAS_Hydro stays cheap and
# works on headless compute environments without a plotting stack.

# Sibling scientific/correction modules.  Relative import when installed as
# part of a package (wass2s), absolute import when the modules sit side by side
# in a flat directory or are run from a notebook.
try:
    from .highwater_season_predictands_modeaware import (
        HighWaterConfig,
        fit_station_highwater_regime,
        compute_annual_predictands,
        bootstrap_window_uncertainty,
    )
except ImportError:  # pragma: no cover - standalone / flat layout
    from highwater_season_predictands_modeaware import (
        HighWaterConfig,
        fit_station_highwater_regime,
        compute_annual_predictands,
        bootstrap_window_uncertainty,
    )

try:
    from .glofas_highflow_bias_correction import (
        GloFASPixelMatchConfig,
        GloFASHighFlowBiasConfig,
        GloFASCorrectionEvaluationConfig,
        prepare_glofas_hybas_matching,
        validate_glofas_auxiliary,
        extract_glofas_discharge_at_matches,
        select_highflow_windows,
        fit_gauged_highflow_corrections,
        apply_gauged_highflow_corrections,
        propagate_multiplicative_factors_network,
        propagate_multiplicative_factors,
        make_hybas_outlet_targets,
        match_hybas_outlets_to_glofas,
        extract_all_hybas_glofas_outlets,
        apply_propagated_highflow_factors,
        outlet_xarray_to_picture,
        evaluate_glofas_highflow_correction,
    )
except ImportError:  # pragma: no cover - standalone / flat layout
    from glofas_highflow_bias_correction import (
        GloFASPixelMatchConfig,
        GloFASHighFlowBiasConfig,
        GloFASCorrectionEvaluationConfig,
        prepare_glofas_hybas_matching,
        validate_glofas_auxiliary,
        extract_glofas_discharge_at_matches,
        select_highflow_windows,
        fit_gauged_highflow_corrections,
        apply_gauged_highflow_corrections,
        propagate_multiplicative_factors_network,
        propagate_multiplicative_factors,
        make_hybas_outlet_targets,
        match_hybas_outlets_to_glofas,
        extract_all_hybas_glofas_outlets,
        apply_propagated_highflow_factors,
        outlet_xarray_to_picture,
        evaluate_glofas_highflow_correction,
    )

# v5 aggregate-first construction. Daily v4 APIs remain available separately.
if __package__:
    from .highflow_predictand import PredictandConfig
    from .highflow_outlet_aggregation import AggregationConfig
else:
    from highflow_predictand import PredictandConfig
    from highflow_outlet_aggregation import AggregationConfig

# A library must not configure the root logger; that is the host application's
# decision.  Attach a NullHandler and let wass2s (or the user) set the level.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class WAS_Hydro:
    # ----------------------------------------------------------------------
    # Missing-data conventions used by the hydrological observation readers.
    # Blank/non-numeric cells are converted to NaN by pandas, while these
    # numeric sentinel codes are replaced explicitly.
    # ----------------------------------------------------------------------
    HYDRO_MISSING_VALUES: tuple[float, ...] = (-99.0, -999.0)

    # ======================================================================
    # Existing utilities
    # ======================================================================

    def adjust_duplicates(self,series: pd.Series,increment: float=0.00001)->pd.Series:
        """Unique indexing coordinates without colliding with existing values.

        The physical coordinates are retained separately as X_ORIG/Y_ORIG and
        restored in DataFrames. This sparse-grid convention is for WASS2S;
        a compact station dimension is preferable for very large networks.
        """
        if not np.isfinite(increment) or increment<=0:raise ValueError("increment must be positive")
        out=pd.to_numeric(series.copy(),errors="coerce").astype(float)
        reserved=set(out[np.isfinite(out)].tolist());used=set()
        for pos,value in enumerate(out.to_numpy()):
            if not np.isfinite(value):continue
            new=value
            if new in used:
                step=1
                while new in used or new in reserved:
                    new=value+increment*step;step+=1
            out.iloc[pos]=new;used.add(new)
        return out

    def _resolve_hydro_missing_values(
        self,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
    ) -> tuple[float, ...]:
        """Return the numeric missing-value codes used for discharge data."""
        if missing_values is None:
            return tuple(float(v) for v in self.HYDRO_MISSING_VALUES)
        return tuple(float(v) for v in missing_values)

    def _clean_discharge_values(
        self,
        values,
        *,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> np.ndarray:
        """Convert discharge values to float and apply hydrological missing-data QC.

        Missing observations may be blank/non-numeric, NaN, -99, -999, or any
        additional numeric sentinel supplied through ``missing_values``.
        Remaining negative discharge can optionally be treated as invalid.
        """
        # copy=True is required: under pandas >= 3 (copy-on-write) to_numpy may
        # return a read-only view, which the in-place masking below cannot edit.
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
            dtype=float, copy=True
        )
        sentinels = self._resolve_hydro_missing_values(missing_values)
        for code in sentinels:
            numeric[np.isclose(numeric, code, equal_nan=False)] = np.nan
        if negative_flow_as_nan:
            numeric[numeric < 0] = np.nan
        return numeric

    def subbasin_data_to_xarray(
        self,
        df: pd.DataFrame,
        *,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> xr.Dataset:
        """Convert the WASS2S outlet table to a sparse T,Y,X xarray Dataset.

        The first three rows are expected to contain longitude, latitude and
        elevation metadata, followed by YYYYmmdd daily observations. Blank
        values and the default sentinel codes -99/-999 are converted to NaN.
        """
        if "ID" not in df.columns:
            raise ValueError("Hydrological DataFrame must contain an 'ID' column.")
        if len(df) < 4:
            raise ValueError("Hydrological DataFrame is too short.")

        station_cols = [c for c in df.columns if c != "ID"]
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["SUBID", "LONG", "LAT", "ELEV"]

        for col in ["LONG", "LAT", "ELEV"]:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
            for code in self._resolve_hydro_missing_values(missing_values):
                metadata.loc[np.isclose(metadata[col], code, equal_nan=False), col] = np.nan

        metadata["LONG"] = self.adjust_duplicates(metadata["LONG"])
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])

        data_part = df.iloc[3:].rename(columns={"ID": "DATE"}).copy()
        data_long = data_part.melt(
            id_vars=["DATE"],
            var_name="SUBID",
            value_name="VALUE",
        )

        final_df = pd.merge(data_long, metadata, on="SUBID", how="left")
        final_df["DATE"] = pd.to_datetime(
            final_df["DATE"].astype(str).str.strip(),
            format="%Y%m%d",
            errors="coerce",
        )
        final_df["VALUE"] = self._clean_discharge_values(
            final_df["VALUE"],
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
        )
        final_df = final_df.dropna(subset=["DATE", "LAT", "LONG"])

        ds = (
            final_df[["DATE", "LAT", "LONG", "VALUE"]]
            .set_index(["DATE", "LAT", "LONG"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LONG": "X", "DATE": "T"})
        )

        # Preserve station identity/elevation as sparse Y/X coordinates.
        ys = ds["Y"].values
        xs = ds["X"].values
        subid = np.full((len(ys), len(xs)), "", dtype=object)
        elev2d = np.full((len(ys), len(xs)), np.nan, dtype=float)
        ymap = {float(v): i for i, v in enumerate(ys)}
        xmap = {float(v): i for i, v in enumerate(xs)}
        for row in metadata.itertuples(index=False):
            if not (np.isfinite(row.LAT) and np.isfinite(row.LONG)):
                continue
            if float(row.LAT) in ymap and float(row.LONG) in xmap:
                yi = ymap[float(row.LAT)]
                xi = xmap[float(row.LONG)]
                subid[yi, xi] = str(row.SUBID)
                elev2d[yi, xi] = float(row.ELEV) if pd.notna(row.ELEV) else np.nan
        ds = ds.assign_coords(SUBID=(("Y", "X"), subid), ELEV=(("Y", "X"), elev2d))
        return ds

    # HydroSHEDS standard (without lakes). The area rules below are configurable
    # cartographic defaults, NOT an official conversion from area to Pfafstetter level.
    HYDROBASINS_AREA_LEVEL_RULES = (
        (5_000, 12), (20_000, 11), (100_000, 10), (500_000, 9),
        (2_000_000, 8), (5_000_000, 7), (10_000_000, 6),
        (20_000_000, 5), (50_000_000, 4), (100_000_000, 3),
        (200_000_000, 2), (float("inf"), 1),
    )
    HYDROSHEDS_REGIONS = ("af", "ar", "as", "au", "eu", "gr", "na", "sa", "si")

    @staticmethod
    def _hydrosheds_bbox(extent_obs):
        """Validate [north, west, south, east]; crossing the dateline is explicit error."""
        try:
            values = np.asarray(extent_obs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("extent_obs must be [north, west, south, east].") from exc
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("extent_obs needs four finite coordinates [N, W, S, E].")
        north, west, south, east = values
        if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
            raise ValueError(
                "Require -90 <= south < north <= 90 and -180 <= west < east <= 180. "
                "Split dateline-crossing requests into two extents."
            )
        return box(west, south, east, north)

    @staticmethod
    def _hydrosheds_wgs84(geo_df):
        import warnings
        if not isinstance(geo_df, gpd.GeoDataFrame):
            raise TypeError("Expected a geopandas.GeoDataFrame.")
        gdf = geo_df.copy()
        if gdf.crs is None:
            warnings.warn("Missing CRS: assuming EPSG:4326, as in the original WAS_Hydro.",
                          UserWarning, stacklevel=2)
            return gdf.set_crs(4326)
        return gdf.to_crs(4326)

    def hydrosheds_area_km2(self, extent_obs):
        """Area of the latitude/longitude rectangle on the WGS84 ellipsoid, km².

        Integrates the ellipsoid surface element along meridians and parallels;
        thus even a large rectangle is not approximated by four geodesic edges.
        This is the area of the requested rectangle, not land or catchment area.
        """
        import math
        west, south, east, north = self._hydrosheds_bbox(extent_obs).bounds
        a, inv_f = 6378137.0, 298.257223563
        f = 1.0 / inv_f
        e2 = f * (2.0 - f)
        e = math.sqrt(e2)
        def primitive(latitude):
            u = math.sin(math.radians(latitude))
            return 0.5 * (u / (1.0 - e2 * u * u) + math.atanh(e * u) / e)
        return (a * a * (1.0 - e2) * math.radians(east - west)
                * (primitive(north) - primitive(south)) / 1e6)

    def choose_hydrobasins_level(self, extent_obs=None, *, area_km2=None,
                                area_level_rules=None):
        """Choose a level using configurable (maximum area in km², level) rules.

        area_km2 can explicitly supply the study-area size; otherwise use the
        area of extent_obs. Levels describe a hierarchy, not uniform basin sizes.
        To request an exact level pass level=1..12 to download_hydrosheds instead.
        """
        if area_km2 is None:
            area_km2 = self.hydrosheds_area_km2(extent_obs)
        if isinstance(area_km2, bool) or not np.isfinite(area_km2) or area_km2 <= 0:
            raise ValueError("area_km2 must be finite and strictly positive.")
        rules = (self.HYDROBASINS_AREA_LEVEL_RULES if area_level_rules is None
                 else tuple(area_level_rules))
        previous, previous_level, selected = 0.0, 12, None
        for maximum, candidate in rules:
            if (np.isnan(maximum) or maximum <= previous or
                    isinstance(candidate, bool) or not isinstance(candidate, (int, np.integer))
                    or not 1 <= candidate <= 12):
                raise ValueError("Area rules need increasing positive maxima and integer levels 1..12.")
            if selected is None and area_km2 <= maximum:
                selected = int(candidate)
            if candidate > previous_level:
                raise ValueError("Levels must stay constant or decrease as area increases.")
            previous = maximum
            previous_level = candidate
        if selected is None:
            raise ValueError("Area rules do not cover this area; add an (inf, level) final rule.")
        return selected

    @staticmethod
    def _hydrosheds_zip_layer(path):
        """Validate archive structure and select its sole shapefile, without extraction."""
        import zipfile
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            shapes = [n for n in names if n.lower().endswith(".shp")
                      and not n.startswith("__MACOSX/")]
            if len(shapes) != 1:
                raise ValueError(f"Expected one shapefile in {path}; found {len(shapes)}.")
            stem = shapes[0][:-4].lower()
            lower_names = {n.lower() for n in names}
            if not all(stem + suffix in lower_names for suffix in (".dbf", ".shx", ".prj")):
                raise ValueError(f"Incomplete shapefile in {path} (need SHP/SHX/DBF/PRJ).")
        return "zip://" + str(Path(path).resolve()) + "!" + shapes[0]

    def _download_hydrosheds_zip(self, url, destination, *, force_download=False,
                                 offline=False, timeout=120, retries=3):
        """Stream to a temporary file; retain an existing valid cache on failure."""
        import os
        import tempfile
        import time
        import urllib.error
        import urllib.request
        import zipfile
        destination = Path(destination)
        if offline and force_download:
            raise ValueError("offline=True is incompatible with force_download=True.")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
            raise ValueError("retries must be a positive integer.")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        if destination.exists() and not force_download:
            try:
                self._hydrosheds_zip_layer(destination)
                logger.info("HydroSHEDS cache: %s", destination)
                return destination
            except (OSError, ValueError, zipfile.BadZipFile):
                if offline:
                    raise ValueError(f"Invalid offline archive: {destination}")
        if offline:
            raise FileNotFoundError(f"Offline archive missing: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            temp_path = None
            try:
                logger.info("Downloading %s", url)
                request = urllib.request.Request(url, headers={"User-Agent": "WASS2S-WAS_Hydro/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    expected = response.headers.get("Content-Length")
                    count = 0
                    with tempfile.NamedTemporaryFile(dir=destination.parent,
                                                     suffix=".part", delete=False) as stream:
                        temp_path = Path(stream.name)
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            stream.write(chunk)
                            count += len(chunk)
                    if expected is not None and count != int(expected):
                        raise OSError(f"Incomplete download: {count} / {expected} bytes")
                self._hydrosheds_zip_layer(temp_path)
                with zipfile.ZipFile(temp_path) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f"ZIP checksum failed for {bad}")
                os.replace(temp_path, destination)
                return destination
            except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
                permanent = (isinstance(exc, urllib.error.HTTPError)
                             and exc.code in (400, 401, 403, 404))
                if permanent or attempt == retries - 1:
                    raise RuntimeError(
                        f"Cannot download HydroSHEDS archive {url}. "
                        f"Download it manually to {destination} and use offline=True if needed."
                    ) from exc
                logger.warning("HydroSHEDS download retry %s/%s: %s", attempt + 1, retries, exc)
                time.sleep(min(2 ** attempt, 8))
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

    @staticmethod
    def _hydrosheds_ids(gdf, field="HYBAS_ID"):
        """Normalize attribute case and keep integer HydroBASINS IDs exact."""
        match = [c for c in gdf.columns if c.upper() == field.upper()]
        if len(match) != 1:
            raise ValueError(f"Expected one {field} field; found {match}.")
        result = gdf.rename(columns={match[0]: field}).copy()
        values = pd.to_numeric(result[field], errors="raise")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Invalid non-integer or missing {field}.")
        result[field] = values.astype("int64")
        return result

    def download_hydrosheds(
        self, extent_obs, output_dir="hydrosheds", *, level="auto", regions="af",
        area_km2=None, area_level_rules=None, include_rivers=True,
        include_pour_points=True, basin_selection="intersects",
        river_min_upstream_area_km2=None, river_min_order=None,
        cache_dir=None, force_download=False, offline=False, timeout=120,
        retries=3, export=True,
    ):
        """Download standard HydroBASINS, HydroRIVERS and official Pour Points.

        Parameters
        ----------
        extent_obs : [north, west, south, east], EPSG:4326
        level : 'auto' or int in 1..12
            Auto uses the rectangle area (or explicit area_km2) and configurable
            HYDROBASINS_AREA_LEVEL_RULES. These are practical defaults, not an
            official area/level relationship or a requested mean subbasin size.
        regions : str or sequence
            Explicit HydroSHEDS regions: af, ar, as, au, eu, gr, na, sa, si.
            Default af is Africa. For a boundary-spanning study include all
            relevant regions; the function does not infer continents from a bbox.
        basin_selection : 'intersects', 'centroid', or 'within'
            Selection preserves entire basin polygons. Intersects is recommended
            for hydrology; clip_bbox(..., clip=True) is for display/export only.
        river_min_upstream_area_km2, river_min_order : optional
            Filter UPLAND_SKM / ORD_STRA. Rivers have one native resolution;
            they do not share the HydroBASINS level system. Entire intersecting
            river reaches are kept, so endpoints may extend outside the basin.
        force_download : bool
            False reuses source archives. Exported subsets have unique run paths.
        offline : bool
            Read only cached official ZIPs (also accepts manually downloaded ZIPs).

        Returns
        -------
        dict: basins, rivers, pour_points (WGS84 GeoDataFrames; optional layers
            are None), metadata, paths. Exports a GeoPackage and JSON manifest
            into a unique subdirectory of output_dir unless export=False.

        Notes
        -----
        Archives are continental for basins/rivers and global for Pour Points;
        this is download-then-subset, not a remote bbox API. Pour Points are
        selected by HYBAS_ID at the chosen level, including outside-bbox outlets
        and multiple outlets of coastal aggregates. They are never centroids.
        HydroRIVERS.HYBAS_L12 cannot be directly joined to a different basin level.
        Basins selected at a rectangle border do not constitute a full upstream
        watershed delineation. NEXT_DOWN can legitimately reference absent basins.
        Source/version links and selection details are retained in the manifest.
        Existing high-water/GloFAS scientific methods and their dependencies remain
        unchanged. See https://www.hydrosheds.org/products/hydrobasins and
        https://www.hydrosheds.org/products/hydrorivers for documentation/license.
        """
        import json
        import tempfile
        import warnings
        from datetime import datetime, timezone

        bbox_poly = self._hydrosheds_bbox(extent_obs)
        bbox_area = self.hydrosheds_area_km2(extent_obs)
        selection_area = bbox_area if area_km2 is None else area_km2
        if (isinstance(selection_area, bool) or not np.isfinite(selection_area)
                or selection_area <= 0):
            raise ValueError("area_km2 must be finite and strictly positive.")
        if area_level_rules is not None:
            area_level_rules = tuple(area_level_rules)
        if level == "auto":
            selected_level = self.choose_hydrobasins_level(
                extent_obs, area_km2=selection_area, area_level_rules=area_level_rules)
        elif isinstance(level, (int, np.integer)) and not isinstance(level, bool) and 1 <= level <= 12:
            selected_level = int(level)
        else:
            raise ValueError("level must be 'auto' or an integer from 1 to 12.")
        if basin_selection not in ("intersects", "centroid", "within"):
            raise ValueError("Invalid basin_selection; use intersects, centroid, or within.")
        if isinstance(regions, str):
            regions = (regions,)
        regions = tuple(dict.fromkeys(str(r).lower() for r in regions))
        if not regions or set(regions) - set(self.HYDROSHEDS_REGIONS):
            raise ValueError(f"regions must use codes {self.HYDROSHEDS_REGIONS}.")
        if river_min_upstream_area_km2 is not None and (
                not np.isfinite(river_min_upstream_area_km2) or river_min_upstream_area_km2 < 0):
            raise ValueError("river_min_upstream_area_km2 must be finite and nonnegative.")
        if river_min_order is not None and (isinstance(river_min_order, bool) or
                not isinstance(river_min_order, (int, np.integer)) or river_min_order < 1):
            raise ValueError("river_min_order must be a positive integer.")
        output_dir = Path(output_dir).expanduser().resolve()
        cache = (output_dir / "cache" if cache_dir is None
                 else Path(cache_dir).expanduser().resolve())
        sources = []
        def get_archive(product, filename, directory, region=None):
            url = f"https://data.hydrosheds.org/file/{directory}/{filename}"
            archive = self._download_hydrosheds_zip(
                url, cache / filename, force_download=force_download, offline=offline,
                timeout=timeout, retries=retries)
            sources.append({"product": product, "region": region, "url": url,
                            "archive": str(archive)})
            return self._hydrosheds_zip_layer(archive)

        logger.info("HydroSHEDS: rectangle %.1f km2, selection %.1f km2, level %02d, regions %s",
                    bbox_area, selection_area, selected_level, regions)
        frames = []
        for region in regions:
            uri = get_archive("basins", f"hybas_{region}_lev{selected_level:02d}_v1c.zip",
                              "HydroBASINS/standard", region)
            frame = gpd.read_file(uri, bbox=bbox_poly.bounds)
            frame = self.clip_bbox(extent_obs, frame, selection=basin_selection)
            frames.append(self._hydrosheds_ids(frame))
        basins = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                 geometry="geometry", crs=4326)
        basins = basins.drop_duplicates("HYBAS_ID").reset_index(drop=True)
        if basins.empty:
            raise ValueError("No selected basins: check extent_obs, regions and basin_selection.")
        # No geometry clipping here: preserve SUB_AREA, UP_AREA and network meaning.
        basin_union = (basins.geometry.union_all() if hasattr(basins.geometry, "union_all")
                       else basins.geometry.unary_union)
        rivers = None
        if include_rivers:
            river_frames = []
            for region in regions:
                uri = get_archive("rivers", f"HydroRIVERS_v10_{region}_shp.zip",
                                  "HydroRIVERS", region)
                frame = self._hydrosheds_wgs84(gpd.read_file(uri, bbox=basin_union.bounds))
                frame = frame.loc[frame.geometry.intersects(basin_union)].copy()
                for field, threshold in (("UPLAND_SKM", river_min_upstream_area_km2),
                                         ("ORD_STRA", river_min_order)):
                    if threshold is not None:
                        if field not in frame:
                            raise ValueError(f"HydroRIVERS archive lacks {field}.")
                        frame = frame.loc[pd.to_numeric(frame[field], errors="coerce") >= threshold]
                river_frames.append(self._hydrosheds_ids(frame, "HYRIV_ID"))
            rivers = gpd.GeoDataFrame(pd.concat(river_frames, ignore_index=True),
                                      geometry="geometry", crs=4326)
            rivers = rivers.drop_duplicates("HYRIV_ID").reset_index(drop=True)
        pour_points = None
        missing_point_ids = []
        if include_pour_points:
            uri = get_archive("pour_points", f"hybas_pour_lev{selected_level:02d}_v1_shp.zip",
                              "hydrobasins/pour_point")
            ids = basins["HYBAS_ID"].tolist()
            # Official source field spelling is Hybas_ID. OGR attribute filtering
            # keeps ALL outlets of selected basins, regardless of their location.
            # OGR limits SQL expression size. Small selections use an attribute
            # filter; large selections scan bounded chunks to avoid a huge IN
            # expression or loading the entire global point layer in Python.
            if len(ids) <= 250:
                where = '"Hybas_ID" IN (' + ",".join(str(int(i)) for i in ids) + ")"
                chunks = [self._hydrosheds_ids(self._hydrosheds_wgs84(
                    gpd.read_file(uri, where=where)))]
            else:
                chunks = []
                offset, chunk_size = 0, 100_000
                while True:
                    chunk = self._hydrosheds_ids(self._hydrosheds_wgs84(
                        gpd.read_file(uri, rows=slice(offset, offset + chunk_size))))
                    chunks.append(chunk.loc[chunk["HYBAS_ID"].isin(ids)].copy())
                    if len(chunk) < chunk_size:
                        break
                    offset += chunk_size
            points = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True),
                                      geometry="geometry", crs=4326)
            pour_points = points.loc[points["HYBAS_ID"].isin(ids)].copy().reset_index(drop=True)
            if not pour_points.empty and not pour_points.geom_type.isin(["Point", "MultiPoint"]).all():
                raise ValueError("Official Pour Points layer must contain point geometries.")
            missing_point_ids = sorted(set(ids) - set(pour_points["HYBAS_ID"]))
            if missing_point_ids:
                warnings.warn(f"No official Pour Point found for {len(missing_point_ids)} basin(s); "
                              "see metadata['missing_pour_point_ids']. No outlets were invented.",
                              UserWarning, stacklevel=2)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "extent_obs_NWSE": [float(v) for v in extent_obs],
            "bbox_area_km2": bbox_area, "level_selection_area_km2": float(selection_area),
            "level": selected_level, "level_mode": "auto" if level == "auto" else "explicit",
            "area_rule_note": "Configurable heuristic; Pfafstetter levels are not uniform areas.",
            "area_level_rules": [[None if np.isinf(a) else float(a), int(l)] for a, l in
                                 (self.HYDROBASINS_AREA_LEVEL_RULES if area_level_rules is None
                                  else area_level_rules)],
            "regions": list(regions), "basin_selection": basin_selection,
            "basin_geometry": "full", "river_geometry": "full intersecting reaches",
            "pour_point_selection": "HYBAS_ID; all matching official outlets, including outside bbox",
            "river_min_upstream_area_km2": (None if river_min_upstream_area_km2 is None
                                            else float(river_min_upstream_area_km2)),
            "river_min_order": None if river_min_order is None else int(river_min_order),
            "missing_pour_point_ids": [int(v) for v in missing_point_ids],
            "counts": {"basins": len(basins), "rivers": None if rivers is None else len(rivers),
                       "pour_points": None if pour_points is None else len(pour_points)},
            "sources": sources, "versions": {"basins": "v1c standard", "rivers": "v10",
                                               "pour_points": "v1 standard only"},
            "documentation": ["https://www.hydrosheds.org/products/hydrobasins",
                              "https://www.hydrosheds.org/products/hydrorivers"],
        }
        paths = {}
        if export:
            output_dir.mkdir(parents=True, exist_ok=True)
            subset_dir = Path(tempfile.mkdtemp(prefix=f"subset_lev{selected_level:02d}_", dir=output_dir))
            gpkg_path = subset_dir / "hydrosheds.gpkg"
            for layer_name, frame in (("basins", basins), ("rivers", rivers), ("pour_points", pour_points)):
                if frame is not None:
                    frame.to_file(gpkg_path, layer=layer_name, driver="GPKG", index=False)
            manifest_path = subset_dir / "metadata.json"
            manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            paths = {"geopackage": str(gpkg_path), "metadata": str(manifest_path)}
        return {"basins": basins, "rivers": rivers, "pour_points": pour_points,
                "metadata": metadata, "paths": paths}

    # Alias following the naming convention of other WASS2S download methods.
    WAS_Download_HydroSHEDS = download_hydrosheds

    def clip_bbox(self, extent_obs, geo_df, *, selection="intersects", clip=False):
        """Select features using ``[north, west, south, east]`` in degrees.

        By default, keep the **complete** geometry of every intersecting
        feature, including basins whose centroid lies outside the window.
        ``selection="centroid"`` reproduces the former geographic-centroid
        selection; ``selection="within"`` keeps only fully contained features.
        ``clip=True`` additionally cuts the selected geometries at the window
        boundary. Such cutting is for display/export: basin area and network
        attributes continue to describe the original HydroSHEDS feature.

        The result is a copy in EPSG:4326. Boundary-only intersections that
        reduce a polygon to a line (or a line to a point) are removed when
        clipping. An extent crossing the antimeridian must be split first.
        """
        from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon

        bbox_poly = self._hydrosheds_bbox(extent_obs)
        if selection not in {"intersects", "centroid", "within"}:
            raise ValueError("selection must be 'intersects', 'centroid', or 'within'.")
        if not isinstance(clip, (bool, np.bool_)):
            raise TypeError("clip must be a boolean.")
        gdf = self._hydrosheds_wgs84(geo_df)
        usable = gdf.geometry.notna() & ~gdf.geometry.is_empty
        gdf = gdf.loc[usable].copy()
        if gdf.empty:
            return gdf
        if selection == "centroid":
            # Raw Shapely intentionally preserves the former lon/lat centroid
            # criterion; this is not an equal-area physical centroid.
            selected = gdf.geometry.map(lambda geom: geom.centroid.within(bbox_poly))
        elif selection == "within":
            selected = gdf.geometry.within(bbox_poly)
        else:
            selected = gdf.geometry.intersects(bbox_poly)
        out = gdf.loc[selected].copy()
        if not clip or out.empty:
            return out

        def dimension(geom):
            kind = geom.geom_type
            if kind in {"Polygon", "MultiPolygon"}:
                return 2
            if kind in {"LineString", "LinearRing", "MultiLineString"}:
                return 1
            if kind in {"Point", "MultiPoint"}:
                return 0
            return max((dimension(part) for part in geom.geoms), default=-1)

        def matching_parts(geom, target_dim):
            if geom.is_empty:
                return []
            if geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
                return [part for child in geom.geoms
                        for part in matching_parts(child, target_dim)]
            return [geom] if dimension(geom) == target_dim else []

        def cut_preserving_dimension(geom):
            dim = dimension(geom)
            parts = matching_parts(geom.intersection(bbox_poly), dim)
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return {0: MultiPoint, 1: MultiLineString, 2: MultiPolygon}[dim](parts)

        # Assign positionally so duplicate input indices remain supported.
        geometry_name = out.geometry.name
        out[geometry_name] = gpd.GeoSeries(
            [cut_preserving_dimension(geom) for geom in out.geometry],
            index=out.index, crs=out.crs,
        )
        return out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()

    def plot_country_subbasins(
        self,
        geo_centroid_zone,
        name="Country Subbasins",
        *,
        rivers=None,
        pour_points=None,
        extent_obs=None,
        show=True,
        save_path=None,
        figsize=(9, 7),
        basemap=True,
        projection=None,
        river_min_upstream_area_km2=None,
        borders=True,
    ):
        """Plot HydroBASINS with optional HydroRIVERS and pour points.

        ``projection`` defaults to Cartopy Equal Earth; all supplied layers
        are transformed from EPSG:4326. ``extent_obs`` uses the WASS2S order
        ``[north, west, south, east]``. Without an explicit extent, the view
        includes all supplied features, including basin outlets outside the
        initial download window.

        With ``basemap=True``, land is pink, country borders are black (1 pt),
        and lakes are blue (Natural Earth 10m, drawn above basin fills).
        Set ``basemap=False`` for offline rendering without Natural Earth
        downloads. HydroRIVERS are displayed only when supplied as ``rivers``.
        ``river_min_upstream_area_km2`` filters their ``UPLAND_SKM`` attribute
        for display; it does not change the input or downloaded layer.
        Returns ``(figure, axes)``; ``show=False`` supports notebooks and HPC.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError as exc:  # pragma: no cover - optional dependencies
            raise ImportError(
                "plot_country_subbasins requires matplotlib and Cartopy."
            ) from exc

        def prepare_layer(frame, allowed, label):
            if frame is None:
                return None
            layer = self._hydrosheds_wgs84(frame)
            layer = layer.loc[layer.geometry.notna() & ~layer.geometry.is_empty].copy()
            wrong = ~layer.geometry.geom_type.isin(allowed)
            if wrong.any():
                kinds = ", ".join(sorted(layer.loc[wrong].geometry.geom_type.unique()))
                raise ValueError(f"{label} contains unsupported geometry types: {kinds}.")
            return layer

        basins = prepare_layer(geo_centroid_zone, {"Polygon", "MultiPolygon"}, "Basins")
        streams = prepare_layer(rivers, {"LineString", "MultiLineString"}, "Rivers")
        outlets = prepare_layer(pour_points, {"Point", "MultiPoint"}, "Pour points")
        if river_min_upstream_area_km2 is not None:
            threshold = float(river_min_upstream_area_km2)
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("river_min_upstream_area_km2 must be finite and nonnegative.")
            if streams is None:
                raise ValueError("A river area filter requires the rivers layer.")
            if "UPLAND_SKM" not in streams.columns:
                raise ValueError("The rivers layer must contain UPLAND_SKM for area filtering.")
            areas = pd.to_numeric(streams["UPLAND_SKM"], errors="coerce")
            streams = streams.loc[areas >= threshold].copy()

        active = [layer for layer in (basins, streams, outlets)
                  if layer is not None and not layer.empty]
        if extent_obs is not None:
            xmin, ymin, xmax, ymax = self._hydrosheds_bbox(extent_obs).bounds
        else:
            if not active:
                raise ValueError("No nonempty geometry to plot; supply a layer or extent_obs.")
            bounds = np.asarray([layer.total_bounds for layer in active], dtype=float)
            if not np.isfinite(bounds).all():
                raise ValueError("Layer bounds must contain finite geographic coordinates.")
            xmin, ymin = bounds[:, :2].min(axis=0)
            xmax, ymax = bounds[:, 2:].max(axis=0)
            pad_x = max((xmax - xmin) * 0.05, 0.05)
            pad_y = max((ymax - ymin) * 0.05, 0.05)
            xmin, xmax = max(-180., xmin - pad_x), min(180., xmax + pad_x)
            ymin, ymax = max(-90., ymin - pad_y), min(90., ymax + pad_y)

        data_crs = ccrs.PlateCarree()
        map_crs = ccrs.EqualEarth() if projection is None else projection
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": map_crs})
        if basemap:
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7d6e0", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e8f4fa", zorder=0)
            # Above basin fills (zorder=1), so lakes remain visible within basins.
            ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#b7dff2",
                           edgecolor="#0077b6", linewidth=0.4, zorder=2)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#000000",
                           linewidth=1.2, zorder=3)
            if borders:
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.0,
                               edgecolor="#000000", zorder=3)
        else:
            ax.set_facecolor("#f8f8f8")

        legend = []
        if basins is not None and not basins.empty:
            ax.add_geometries(basins.geometry, crs=data_crs, facecolor="#eeeeee",
                              edgecolor="#000000", linewidth=0.3, zorder=1)
            legend.append(Patch(facecolor="#eeeeee", edgecolor="#000000",
                                linewidth=0.3, label="HydroBASINS"))
        if streams is not None and not streams.empty:
            ax.add_geometries(streams.geometry, crs=data_crs, facecolor="none",
                              edgecolor="#0077b6", linewidth=0.8, zorder=4)
            legend.append(Line2D([0], [0], color="#0077b6", linewidth=0.8, label="HydroRIVERS"))
        if outlets is not None and not outlets.empty:
            points = [point for geom in outlets.geometry
                      for point in (geom.geoms if geom.geom_type == "MultiPoint" else [geom])]
            ax.scatter([point.x for point in points], [point.y for point in points],
                       transform=data_crs, s=24, marker="o", facecolor="#d94801",
                       edgecolor="white", linewidth=0.5, zorder=5)
            legend.append(Line2D([0], [0], linestyle="", marker="o", markerfacecolor="#d94801",
                                 markeredgecolor="white", markersize=6, label="Pour points"))
        ax.set_extent([xmin, xmax, ymin, ymax], crs=data_crs)
        gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.3,
                          color="gray", alpha=0.5, linestyle=":")
        gl.right_labels = False
        gl.top_labels = False
        if legend:
            ax.legend(handles=legend, loc="lower left", framealpha=0.95)
        ax.set_title(name)
        fig.tight_layout()
        if save_path is not None:
            path = Path(save_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax


    # ======================================================================
    # Input adapters for the high-water calculation
    # ======================================================================

    def _read_hydro_observations(
        self,
        df: pd.DataFrame,
        *,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Read a WASS2S hydrological-observation DataFrame station by station.

        Expected layout
        ---------------
        ``ID`` plus one column per outlet/station. The first three rows contain
        longitude, latitude and elevation metadata; all subsequent rows are
        daily observations identified by ``YYYYmmdd``.

        Missing data
        ------------
        Blank/non-numeric cells become NaN automatically. Numeric sentinel
        values are also converted to NaN; by default these are -99 and -999.
        Remaining negative discharge is considered invalid when
        ``negative_flow_as_nan=True``.

        Returns
        -------
        dict
            ``station_name -> {series, X, Y, X_ORIG, Y_ORIG, ELEV}``.

        Notes
        -----
        ``X``/``Y`` are unique indexing coordinates for sparse T,Y,X xarray
        output. ``X_ORIG``/``Y_ORIG`` preserve the physical station coordinates
        and are used when exporting back to the input-like DataFrame format.
        """
        if "ID" not in df.columns:
            raise ValueError("Hydrological DataFrame must contain an 'ID' column.")
        if len(df) < 4:
            raise ValueError(
                "Hydrological DataFrame must contain three metadata rows and "
                "at least one observation row."
            )

        station_cols = [c for c in df.columns if c != "ID"]
        if not station_cols:
            raise ValueError("No station/outlet columns were found.")

        sentinels = self._resolve_hydro_missing_values(missing_values)

        # Original physical metadata.
        lon_raw = pd.to_numeric(df.iloc[0][station_cols], errors="coerce")
        lat_raw = pd.to_numeric(df.iloc[1][station_cols], errors="coerce")
        elev = pd.to_numeric(df.iloc[2][station_cols], errors="coerce")
        for code in sentinels:
            lon_raw = lon_raw.mask(np.isclose(lon_raw, code, equal_nan=False))
            lat_raw = lat_raw.mask(np.isclose(lat_raw, code, equal_nan=False))
            elev = elev.mask(np.isclose(elev, code, equal_nan=False))

        # Only the internal xarray coordinates are nudged when duplicates occur.
        lon = self.adjust_duplicates(lon_raw)
        lat = self.adjust_duplicates(lat_raw)

        raw_dates = df.iloc[3:]["ID"].astype(str).str.strip()
        dates = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
        valid_date = dates.notna().to_numpy()
        dates = pd.DatetimeIndex(dates[valid_date])

        stations: dict[str, dict[str, Any]] = {}
        for name in station_cols:
            values = self._clean_discharge_values(
                df.iloc[3:][name],
                missing_values=sentinels,
                negative_flow_as_nan=negative_flow_as_nan,
            )
            values = values[valid_date]
            s = pd.Series(values, index=dates, name=str(name), dtype=float).sort_index()

            # Duplicate timestamps are averaged rather than arbitrarily dropped.
            if s.index.has_duplicates:
                s = s.groupby(level=0).mean()

            stations[str(name)] = {
                "series": s,
                "X": float(lon[name]) if pd.notna(lon[name]) else np.nan,
                "Y": float(lat[name]) if pd.notna(lat[name]) else np.nan,
                "X_ORIG": float(lon_raw[name]) if pd.notna(lon_raw[name]) else np.nan,
                "Y_ORIG": float(lat_raw[name]) if pd.notna(lat_raw[name]) else np.nan,
                "ELEV": float(elev[name]) if pd.notna(elev[name]) else np.nan,
            }
        return stations

    def _xarray_to_station_series(
        self,
        data: xr.DataArray | xr.Dataset,
        variable: str = "Observation",
        *,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Extract point/outlet series from a T,Y,X DataArray or Dataset.

        Only Y/X locations containing at least one finite value are retained.
        This is intended for outlet observations represented on a sparse T,Y,X
        grid, as produced by ``subbasin_data_to_xarray``.
        """
        if isinstance(data, xr.Dataset):
            if variable not in data:
                raise KeyError(
                    f"Variable {variable!r} not found. Available: {list(data.data_vars)}"
                )
            da = data[variable]
        else:
            da = data

        if "T" not in da.dims:
            raise ValueError("Input xarray must contain temporal dimension 'T'.")

        stations: dict[str, dict[str, Any]] = {}

        # Preferred WASS2S point-on-grid representation.
        if "Y" in da.dims and "X" in da.dims:
            stacked = da.stack(_point=("Y", "X"))
            valid = stacked.notnull().any("T")
            stacked = stacked.sel(_point=valid)

            # Preserve station identity/elevation when the sparse xarray carries
            # the SUBID/ELEV coordinates generated by this class.
            subid_coord = None
            elev_coord = None
            if "SUBID" in da.coords and set(da["SUBID"].dims) == {"Y", "X"}:
                subid_coord = da["SUBID"]
            elif isinstance(data, xr.Dataset) and "SUBID" in data.coords:
                subid_coord = data["SUBID"]
            if "ELEV" in da.coords and set(da["ELEV"].dims) == {"Y", "X"}:
                elev_coord = da["ELEV"]
            elif isinstance(data, xr.Dataset) and "ELEV" in data.coords:
                elev_coord = data["ELEV"]

            for j in range(stacked.sizes.get("_point", 0)):
                one = stacked.isel(_point=j)
                y = float(one["Y"].values)
                x = float(one["X"].values)

                name = f"Y{y:.5f}_X{x:.5f}"
                elev = np.nan
                if subid_coord is not None:
                    raw_name = subid_coord.sel(Y=y, X=x).item()
                    if raw_name is not None and str(raw_name).strip():
                        name = str(raw_name)
                if elev_coord is not None:
                    try:
                        elev = float(elev_coord.sel(Y=y, X=x).item())
                    except (TypeError, ValueError):
                        elev = np.nan

                values = self._clean_discharge_values(
                    np.asarray(one.values),
                    missing_values=missing_values,
                    negative_flow_as_nan=negative_flow_as_nan,
                )
                times = pd.DatetimeIndex(pd.to_datetime(one["T"].values))
                s = pd.Series(values, index=times, name=name).sort_index()
                stations[name] = {
                    "series": s,
                    "X": x,
                    "Y": y,
                    "X_ORIG": x,
                    "Y_ORIG": y,
                    "ELEV": elev,
                }
            return stations

        # Also support a compact T,station / T,SUBID representation.
        other_dims = [d for d in da.dims if d != "T"]
        if len(other_dims) != 1:
            raise ValueError(
                "Unsupported xarray geometry. Expected T,Y,X or T,<station_dim>."
            )

        sdim = other_dims[0]
        for j in range(da.sizes[sdim]):
            one = da.isel({sdim: j})
            label = str(one[sdim].values)
            times = pd.DatetimeIndex(pd.to_datetime(one["T"].values))
            values = self._clean_discharge_values(
                np.asarray(one.values),
                missing_values=missing_values,
                negative_flow_as_nan=negative_flow_as_nan,
            )
            s = pd.Series(values, index=times, name=label)
            stations[label] = {
                "series": s,
                "X": np.nan,
                "Y": np.nan,
                "X_ORIG": np.nan,
                "Y_ORIG": np.nan,
                "ELEV": np.nan,
            }
        return stations

    def _prepare_highflow_observations(
        self,
        data: pd.DataFrame | xr.DataArray | xr.Dataset,
        variable: str = "Observation",
        *,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Normalize supported hydrological observation formats to station series."""
        if isinstance(data, pd.DataFrame):
            return self._read_hydro_observations(
                data,
                missing_values=missing_values,
                negative_flow_as_nan=negative_flow_as_nan,
            )
        if isinstance(data, (xr.DataArray, xr.Dataset)):
            return self._xarray_to_station_series(
                data,
                variable=variable,
                missing_values=missing_values,
                negative_flow_as_nan=negative_flow_as_nan,
            )
        raise TypeError(
            "data must be a hydrological pandas.DataFrame or xarray DataArray/Dataset"
        )

    # ======================================================================
    # Public high-flow API
    # ======================================================================

    def WAS_HighFlow_Predictands(
        self,
        data: pd.DataFrame | xr.DataArray | xr.Dataset,
        *,
        variable: str = "Observation",
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
        config: Optional[HighWaterConfig] = None,
        mode_specific: Optional[bool] = None,
        periods: Optional[ClimatologyPeriod] = None,
        reference_start: Optional[str] = None,
        reference_end: Optional[str] = None,
        n_bootstrap: Optional[int] = None,
        output_format: str = "both",
        return_diagnostics: bool = True,
    ):
        """Build station-specific seasonal high-water forecast predictands.

        Parameters
        ----------
        data : DataFrame or xarray
            Either your original WASS2S hydrological table or a T,Y,X xarray.
        variable : str
            Used only when ``data`` is an xarray.Dataset.
        missing_values : sequence of numbers, optional
            Numeric sentinel values representing missing discharge. Defaults to
            ``(-99, -999)``. Blank/non-numeric cells are always converted to NaN.
        negative_flow_as_nan : bool, default True
            Convert remaining negative discharge values to NaN after sentinel
            replacement. Recommended for ordinary river discharge.
        config : HighWaterConfig, optional
            Full scientific configuration. If omitted, robust defaults are used.
        mode_specific : bool or None, default None
            None
                Keep config.mode_specific_outputs unchanged.
            False
                Preserve season-only behaviour: HW1, HW2, ...
            True
                Additionally construct MODE1, MODE2, ... .  Multiple modes may
                belong to one continuous HW season and are then split at the
                intermodal climatological valley.
        reference_start, reference_end : str, optional
            Restrict the climatological reference period, e.g. 1991-01-01 and
            2020-12-31.
        n_bootstrap : int, optional
            Override the config bootstrap count. Use 0 for fast development runs.
        output_format : {'both', 'xarray', 'dataframe'}
            ``both`` returns a dictionary containing both representations.
            ``xarray`` returns only the xarray Dataset.
            ``dataframe`` returns a dictionary ``target_name -> input-like DataFrame``.
        return_diagnostics : bool
            Add the scientific diagnostic tables to the returned object.

        Returns
        -------
        dict, xr.Dataset, or dict[str, pandas.DataFrame]
            With ``output_format='both'`` (recommended), the result is::

                {
                    'xarray': <xr.Dataset>,
                    'dataframe': {
                        'HW1_Q_MEAN': <input-like DataFrame>,
                        'HW1_Q_MAX': <input-like DataFrame>,
                        ...
                    },
                    'diagnostics': {...},   # when requested
                }

            Each input-like DataFrame mimics the observation input: rows LON, LAT,
            DAILY/ELEV, followed by one YYYY0101 row per hydrological year.

        Notes
        -----
        Important target names include:
            HW1_Q_MEAN, HW1_Q_MAX, HW2_Q_MEAN, ...
            Q_HW_MEAN_MAIN, Q_HW_MAX_MAIN
            Q_HW_MEAN_ALL,  Q_HW_MAX_ALL

        When ``mode_specific=True``:
            MODE1_Q_MEAN, MODE1_Q_MAX, MODE2_Q_MEAN, ...
            Q_MODE_MEAN_MAIN, Q_MODE_MAX_MAIN
            Q_MODE_MEAN_ALL,  Q_MODE_MAX_ALL
        """
        cfg = config if config is not None else HighWaterConfig()
        # dataclasses.replace avoids modifying a configuration object supplied by
        # the caller, which is useful when several forecast experiments share it.
        if mode_specific is not None:
            cfg = replace(cfg, mode_specific_outputs=bool(mode_specific))
        if periods is not None:
            periods.validate()
            if reference_start is not None or reference_end is not None:
                raise ValueError("Use periods or reference_start/end, not both")
            reference_start,reference_end=periods.climatology
        if n_bootstrap is not None:
            cfg = replace(cfg, n_bootstrap=int(n_bootstrap))

        stations = self._prepare_highflow_observations(
            data,
            variable=variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
        )
        if not stations:
            raise ValueError("No valid station/outlet time series found in input.")

        annual_tables: dict[str, pd.DataFrame] = {}
        regimes: dict[str, dict[str, Any]] = {}
        station_rows: list[dict[str, Any]] = []
        season_rows: list[dict[str, Any]] = []
        mode_rows: list[dict[str, Any]] = []

        for station, meta in stations.items():
            series = meta["series"]
            logger.info("High-water regime: %s", station)

            try:
                if periods is not None:periods.validate(pd.DatetimeIndex(series.dropna().index))
                regime = fit_station_highwater_regime(
                    series,
                    cfg,
                    reference_start=reference_start,
                    reference_end=reference_end,
                )
                application_series=slice_series(series,*periods.application) if periods is not None else series
                annual = compute_annual_predictands(application_series, regime, cfg)

                # Optional block-bootstrap confidence intervals on the detected
                # boundaries. The reference period is already applied by
                # fit_station_highwater_regime; bootstrap_window_uncertainty
                # takes only (series, regime, config) and RETURNS the annotated
                # windows, so the result must be written back into the regime,
                # otherwise the CI columns never reach highwater_seasons.
                if cfg.n_bootstrap > 0:
                    regime["windows"] = bootstrap_window_uncertainty(
                        series,
                        regime,
                        cfg,
                    )

                regimes[station] = regime
                annual_tables[station] = annual

                station_rows.append(
                    {
                        "station": station,
                        "X": meta["X"],
                        "Y": meta["Y"],
                        "ELEV": meta["ELEV"],
                        "status": regime.get("regime_status","OK"),
                        "modality": regime["modality"],
                        "n_modes": regime["n_modes"],
                        "n_highwater_seasons": regime["n_highwater_seasons"],
                        "n_mode_windows": regime.get("n_mode_windows", 0),
                        "hydrological_anchor": regime["anchor_month_day"],
                        "annual_mean_q": regime["annual_mean_q"],
                        "flow_center_of_mass_doy": regime.get("flow_center_of_mass_doy", np.nan),
                        "circular_concentration": regime.get("circular_concentration", np.nan),
                        "bootstrap_enabled": bool(cfg.n_bootstrap > 0),
                    }
                )

                for w in regime["windows"]:
                    row = {"station": station, "X": meta["X"], "Y": meta["Y"]}
                    row.update({k: v for k, v in w.items() if k != "mode_positions"})
                    season_rows.append(row)

                if cfg.mode_specific_outputs:
                    for m in regime.get("mode_windows", []):
                        row = {"station": station, "X": meta["X"], "Y": meta["Y"]}
                        row.update(m)
                        mode_rows.append(row)

            except Exception as exc:
                # Regional datasets often contain a few unusable outlets.  Do not
                # abort the complete domain; record the failure transparently.
                logger.exception("High-water fit failed for %s: %s", station, exc)
                station_rows.append(
                    {
                        "station": station,
                        "X": meta["X"],
                        "Y": meta["Y"],
                        "ELEV": meta["ELEV"],
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )

        if not annual_tables:
            raise RuntimeError("High-water regime fitting failed for every station.")

        ds = self._annual_highflow_tables_to_xarray(annual_tables, stations)
        picture_tables = self._annual_highflow_tables_to_picture(annual_tables, stations)

        diagnostics = {
            "station_summary": pd.DataFrame(station_rows),
            "highwater_seasons": pd.DataFrame(season_rows),
            "highflow_modes": pd.DataFrame(mode_rows),
            "annual_long": self._annual_tables_to_long(annual_tables, stations),
            "regimes": regimes,
            "config": cfg,
            "periods": periods.describe() if periods is not None else None,
        }
        return self._format_highflow_outputs(
            ds,
            picture_tables,
            diagnostics=diagnostics,
            output_format=output_format,
            return_diagnostics=return_diagnostics,
        )

    def _annual_highflow_tables_to_xarray(
        self,
        annual_tables: dict[str, pd.DataFrame],
        stations: dict[str, dict[str, Any]],
    ) -> xr.Dataset:
        """Convert station annual tables to a WASS2S-style T,Y,X Dataset."""
        # Keep only actual forecast targets; diagnostics such as N_VALID and
        # PEAK_DATE remain in the long DataFrame returned separately.
        target_cols = self._highflow_target_columns(annual_tables)

        # Retain only hydrological years for which at least one forecast target
        # is actually available somewhere in the domain. This avoids exporting
        # empty leading/trailing years created by incomplete hydrological years.
        valid_years: set[int] = set()
        for tab in annual_tables.values():
            present = [c for c in target_cols if c in tab.columns]
            if not present:
                continue
            mask = tab[present].notna().any(axis=1)
            valid_years.update(
                tab.loc[mask, "hydro_year"].dropna().astype(int).tolist()
            )
        all_years = sorted(valid_years)
        T = pd.to_datetime([f"{y:04d}-01-01" for y in all_years])

        # If geographic coordinates exist, preserve the requested T,Y,X layout.
        usable_geo = [
            s for s in stations
            if np.isfinite(stations[s].get("X", np.nan))
            and np.isfinite(stations[s].get("Y", np.nan))
        ]

        if usable_geo:
            ys = np.array(sorted({float(stations[s]["Y"]) for s in usable_geo}))
            xs = np.array(sorted({float(stations[s]["X"]) for s in usable_geo}))
            ymap = {float(v): i for i, v in enumerate(ys)}
            xmap = {float(v): i for i, v in enumerate(xs)}
            tmap = {int(y): i for i, y in enumerate(all_years)}

            data_vars = {}
            for var in target_cols:
                arr = np.full((len(T), len(ys), len(xs)), np.nan, dtype=float)
                for station, tab in annual_tables.items():
                    if station not in usable_geo or var not in tab:
                        continue
                    yi = ymap[float(stations[station]["Y"])]
                    xi = xmap[float(stations[station]["X"])]
                    for row in tab[["hydro_year", var]].itertuples(index=False):
                        if pd.isna(row[1]):
                            continue
                        ti = tmap[int(row[0])]
                        arr[ti, yi, xi] = float(row[1])
                data_vars[var] = (("T", "Y", "X"), arr)

            # Preserve station identity on the sparse Y/X outlet grid.  Cells
            # that do not correspond to an observed outlet remain empty.
            subid = np.full((len(ys), len(xs)), "", dtype=object)
            elev2d = np.full((len(ys), len(xs)), np.nan, dtype=float)
            for station in usable_geo:
                yi = ymap[float(stations[station]["Y"])]
                xi = xmap[float(stations[station]["X"])]
                subid[yi, xi] = station
                elev2d[yi, xi] = float(stations[station].get("ELEV", np.nan))

            ds = xr.Dataset(data_vars=data_vars, coords={"T": T, "Y": ys, "X": xs})
            ds = ds.assign_coords(
                SUBID=(("Y", "X"), subid),
                ELEV=(("Y", "X"), elev2d),
            )
            ds.attrs.update(
                {
                    "description": "Annual high-water discharge predictands",
                    "T_semantics": "T labels hydrological year; station-specific anchor is in diagnostics",
                    "units": "same discharge units as observations",
                }
            )
            return ds

        # Fallback compact station dimension if coordinates are unavailable.
        names = list(stations)
        tmap = {int(y): i for i, y in enumerate(all_years)}
        smap = {s: i for i, s in enumerate(names)}
        data_vars = {}
        for var in target_cols:
            arr = np.full((len(T), len(names)), np.nan, dtype=float)
            for station, tab in annual_tables.items():
                if var not in tab:
                    continue
                si = smap[station]
                for row in tab[["hydro_year", var]].itertuples(index=False):
                    if pd.isna(row[1]):
                        continue
                    arr[tmap[int(row[0])], si] = float(row[1])
            data_vars[var] = (("T", "station"), arr)

        return xr.Dataset(data_vars=data_vars, coords={"T": T, "station": names})

    def _annual_highflow_tables_to_picture(
        self,
        annual_tables: dict[str, pd.DataFrame],
        stations: dict[str, dict[str, Any]],
    ) -> dict[str, pd.DataFrame]:
        """Return every forecast target in the original picture-table geometry.

        A picture table stores one variable for all outlets, therefore all-target
        output is naturally a dictionary ``variable_name -> DataFrame``.  Station
        column order follows the input order.  Failed/unavailable stations remain
        present and contain NaN values, which makes round-tripping predictable.

        The annual row label is ``YYYY0101`` where YYYY is the hydrological-year
        label.  For a cross-calendar-year window this is only the WASS2S annual
        coordinate; the true start/end month-days remain in diagnostics.
        """
        target_cols = self._highflow_target_columns(annual_tables)
        station_names = list(stations)

        out: dict[str, pd.DataFrame] = {}
        for var in target_cols:
            # A picture table is one variable, so its time axis can be filtered
            # specifically to years with at least one valid station value.
            valid_years: set[int] = set()
            for tab in annual_tables.values():
                if var not in tab.columns:
                    continue
                mask = tab[var].notna()
                valid_years.update(
                    tab.loc[mask, "hydro_year"].dropna().astype(int).tolist()
                )
            all_years = sorted(valid_years)

            rows: list[dict[str, Any]] = []

            lon_row = {"ID": "LON"}
            lat_row = {"ID": "LAT"}
            elev_row = {"ID": "DAILY/ELEV"}
            for station in station_names:
                meta = stations[station]
                lon_row[station] = meta.get("X_ORIG", meta.get("X", np.nan))
                lat_row[station] = meta.get("Y_ORIG", meta.get("Y", np.nan))
                elev_row[station] = meta.get("ELEV", np.nan)
            rows.extend([lon_row, lat_row, elev_row])

            # Build quick year->value mappings once per station.
            lookup: dict[str, dict[int, float]] = {}
            for station in station_names:
                tab = annual_tables.get(station)
                if tab is None or var not in tab.columns:
                    lookup[station] = {}
                    continue
                tmp = tab[["hydro_year", var]].copy()
                tmp = tmp.dropna(subset=["hydro_year"])
                lookup[station] = {
                    int(y): (float(v) if pd.notna(v) else np.nan)
                    for y, v in tmp.itertuples(index=False, name=None)
                }

            for year in all_years:
                row = {"ID": f"{year:04d}0101"}
                for station in station_names:
                    row[station] = lookup[station].get(year, np.nan)
                rows.append(row)

            picture = pd.DataFrame(rows, columns=["ID", *station_names])
            picture.attrs.update(
                {
                    "variable": var,
                    "description": "Annual high-flow discharge predictand",
                    "time_semantics": "YYYY0101 labels hydrological year, not necessarily the physical season start",
                    "units": "same discharge units as observations",
                }
            )
            out[var] = picture
        return out

    def _highflow_target_columns(
        self,
        annual_tables: dict[str, pd.DataFrame],
    ) -> list[str]:
        """Return only annual discharge target columns, excluding diagnostics."""
        return sorted(
            {
                c
                for tab in annual_tables.values()
                for c in tab.columns
                if (
                    c.endswith("_Q_MEAN")
                    or c.endswith("_Q_MAX")
                    or c in {
                        "Q_HW_MEAN_MAIN",
                        "Q_HW_MAX_MAIN",
                        "Q_HW_MEAN_ALL",
                        "Q_HW_MAX_ALL",
                        "Q_MODE_MEAN_MAIN",
                        "Q_MODE_MAX_MAIN",
                        "Q_MODE_MEAN_ALL",
                        "Q_MODE_MAX_ALL",
                    }
                )
            }
        )

    def _format_highflow_outputs(
        self,
        xarray_obj,
        dataframe_obj,
        *,
        diagnostics: Optional[dict[str, Any]] = None,
        output_format: str = "both",
        return_diagnostics: bool = False,
    ):
        """Centralize output-format handling for the public high-flow API."""
        fmt = str(output_format).lower().strip()
        if fmt not in {"both", "xarray", "dataframe"}:
            raise ValueError("output_format must be 'both', 'xarray', or 'dataframe'.")

        if fmt == "both":
            result = {"xarray": xarray_obj, "dataframe": dataframe_obj}
            if return_diagnostics:
                result["diagnostics"] = diagnostics
            return result

        obj = xarray_obj if fmt == "xarray" else dataframe_obj
        if return_diagnostics:
            return obj, diagnostics
        return obj

    def _annual_tables_to_long(
        self,
        annual_tables: dict[str, pd.DataFrame],
        stations: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Concatenate complete annual station diagnostics into a long table."""
        blocks = []
        for station, tab in annual_tables.items():
            tmp = tab.copy()
            tmp.insert(0, "station", station)
            tmp.insert(1, "X", stations[station].get("X", np.nan))
            tmp.insert(2, "Y", stations[station].get("Y", np.nan))
            tmp.insert(3, "ELEV", stations[station].get("ELEV", np.nan))
            blocks.append(tmp)
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    def _resolve_highflow_varname(
        self,
        *,
        statistic: str = "mean",
        selector: str | int = "main",
        kind: str = "season",
    ) -> str:
        """Resolve public selector arguments to the internal target variable."""
        statistic = str(statistic).lower().strip()
        kind = str(kind).lower().strip()
        if statistic not in {"mean", "max"}:
            raise ValueError("statistic must be 'mean' or 'max'.")
        if kind not in {"season", "mode"}:
            raise ValueError("kind must be 'season' or 'mode'.")

        stat = statistic.upper()
        if isinstance(selector, str):
            sel = selector.lower().strip()
            if sel in {"main", "all"}:
                prefix = "HW" if kind == "season" else "MODE"
                return f"Q_{prefix}_{stat}_{sel.upper()}"
            if sel.startswith("hw") and kind == "season":
                return f"HW{int(sel[2:])}_Q_{stat}"
            if sel.startswith("mode") and kind == "mode":
                return f"MODE{int(sel[4:])}_Q_{stat}"
            try:
                n = int(sel)
            except ValueError as exc:
                raise ValueError(
                    "selector must be 'main', 'all', or a positive integer."
                ) from exc
        else:
            n = int(selector)

        if n < 1:
            raise ValueError("Integer selector must be >= 1.")
        prefix = "HW" if kind == "season" else "MODE"
        return f"{prefix}{n}_Q_{stat}"

    def WAS_Select_HighFlow_Target(
        self,
        targets: xr.Dataset,
        *,
        statistic: str = "mean",
        selector: str | int = "main",
        kind: str = "season",
    ) -> xr.DataArray:
        """Select one target from an already-computed high-flow xarray Dataset."""
        varname = self._resolve_highflow_varname(
            statistic=statistic, selector=selector, kind=kind
        )
        if varname not in targets:
            raise KeyError(
                f"Target {varname!r} was not generated. "
                f"Available targets: {list(targets.data_vars)}"
            )

        out = targets[varname].copy()
        out.name = varname
        out.attrs.update(
            {
                "long_name": varname.replace("_", " "),
                "statistic": str(statistic).lower().strip(),
                "window_kind": str(kind).lower().strip(),
                "selector": str(selector),
                "units": targets.attrs.get(
                    "units", "same discharge units as observations"
                ),
            }
        )
        return out

    def WAS_HighFlow_Target(
        self,
        data: pd.DataFrame | xr.DataArray | xr.Dataset,
        *,
        statistic: str = "mean",
        selector: str | int = "main",
        kind: str = "season",
        mode_specific: Optional[bool] = None,
        variable: str = "Observation",
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
        config: Optional[HighWaterConfig] = None,
        periods: Optional[ClimatologyPeriod] = None,
        reference_start: Optional[str] = None,
        reference_end: Optional[str] = None,
        n_bootstrap: Optional[int] = None,
        output_format: str = "both",
        return_diagnostics: bool = False,
    ):
        """Return one selected high-flow predictand in xarray/DataFrame formats.

        Parameters
        ----------
        statistic : {'mean', 'max'}
            ``mean`` = seasonal mean discharge.
            ``max``  = seasonal maximum discharge.
        selector : {'main', 'all', 1, 2, 3, ...}
            * ``main`` : climatologically dominant season/mode.
            * ``all``  : all windows combined.
            * integer  : chronological HW or MODE number.
        kind : {'season', 'mode'}
            ``season`` selects HW1/HW2/... .
            ``mode`` selects MODE1/MODE2/... and automatically activates the
            optional mode-aware behaviour unless explicitly disabled.

        Examples
        --------
        Seasonal mean of the dominant contiguous high-water season::

            q = hydro.WAS_HighFlow_Target(df, statistic='mean', selector='main')

        Maximum of the second high-water season::

            q = hydro.WAS_HighFlow_Target(df, statistic='max', selector=2)

        Mean discharge of chronological second mode in a bimodal regime::

            q = hydro.WAS_HighFlow_Target(
                df, statistic='mean', selector=2, kind='mode'
            )
        """
        statistic = str(statistic).lower().strip()
        kind = str(kind).lower().strip()
        if statistic not in {"mean", "max"}:
            raise ValueError("statistic must be 'mean' or 'max'.")
        if kind not in {"season", "mode"}:
            raise ValueError("kind must be 'season' or 'mode'.")

        if mode_specific is None:
            mode_specific = kind == "mode"
        if kind == "mode" and not mode_specific:
            raise ValueError("kind='mode' requires mode_specific=True.")

        all_outputs = self.WAS_HighFlow_Predictands(
            data,
            variable=variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
            config=config,
            mode_specific=mode_specific,
            periods=periods,
            reference_start=reference_start,
            reference_end=reference_end,
            n_bootstrap=n_bootstrap,
            output_format="both",
            return_diagnostics=True,
        )
        ds = all_outputs["xarray"]
        picture_tables = all_outputs["dataframe"]
        diagnostics = all_outputs["diagnostics"]

        varname = self._resolve_highflow_varname(
            statistic=statistic, selector=selector, kind=kind
        )
        out_xr = self.WAS_Select_HighFlow_Target(
            ds, statistic=statistic, selector=selector, kind=kind
        )
        if varname not in picture_tables:
            raise KeyError(
                f"Picture target {varname!r} was not generated. "
                f"Available: {list(picture_tables)}"
            )
        out_df = picture_tables[varname].copy()

        return self._format_highflow_outputs(
            out_xr,
            out_df,
            diagnostics=diagnostics,
            output_format=output_format,
            return_diagnostics=return_diagnostics,
        )

    def high_flow(self, *args, **kwargs):
        """Short WASS2S alias of :meth:`WAS_HighFlow_Target`.

        Examples
        --------
        Return the dominant high-water seasonal mean in both formats::

            result = hydro.high_flow(df, statistic="mean", selector="main")
            q_xr = result["xarray"]
            q_df = result["dataframe"]

        Return the second climatological mode maximum::

            result = hydro.high_flow(
                df, statistic="max", selector=2, kind="mode"
            )
        """
        return self.WAS_HighFlow_Target(*args, **kwargs)

    # ======================================================================
    # GloFAS / HydroBASINS matching, correction and evaluation
    # ======================================================================

    def _station_metadata_frame(
        self,
        stations: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Build the station metadata table used by HydroBASINS matching."""
        rows = []
        for station, meta in stations.items():
            rows.append({
                "station": str(station),
                # Original physical coordinates are preferred for spatial matching.
                "X": float(meta.get("X_ORIG", meta.get("X", np.nan))),
                "Y": float(meta.get("Y_ORIG", meta.get("Y", np.nan))),
                "ELEV": float(meta.get("ELEV", np.nan)) if pd.notna(meta.get("ELEV", np.nan)) else np.nan,
            })
        return pd.DataFrame(rows)

    def _daily_station_wide_to_xarray(
        self,
        wide: pd.DataFrame,
        stations: dict[str, dict[str, Any]],
        *,
        name: str,
    ) -> xr.DataArray:
        """Convert date×station daily discharge to sparse WASS2S T,Y,X."""
        idx = pd.DatetimeIndex(wide.index)
        usable = [
            s for s in wide.columns
            if s in stations
            and np.isfinite(stations[s].get("X", np.nan))
            and np.isfinite(stations[s].get("Y", np.nan))
        ]
        if not usable:
            return xr.DataArray(
                wide.to_numpy(dtype=float),
                coords={"T": idx, "station": wide.columns.astype(str)},
                dims=("T", "station"),
                name=name,
            )
        ys = np.array(sorted({float(stations[s]["Y"]) for s in usable}))
        xs = np.array(sorted({float(stations[s]["X"]) for s in usable}))
        ymap = {float(v): i for i, v in enumerate(ys)}
        xmap = {float(v): i for i, v in enumerate(xs)}
        arr = np.full((len(idx), len(ys), len(xs)), np.nan, dtype=float)
        subid = np.full((len(ys), len(xs)), "", dtype=object)
        elev = np.full((len(ys), len(xs)), np.nan, dtype=float)
        for s in usable:
            yi = ymap[float(stations[s]["Y"])]
            xi = xmap[float(stations[s]["X"])]
            arr[:, yi, xi] = pd.to_numeric(wide[s], errors="coerce").to_numpy(dtype=float)
            subid[yi, xi] = str(s)
            elev[yi, xi] = float(stations[s].get("ELEV", np.nan))
        da = xr.DataArray(arr, coords={"T": idx, "Y": ys, "X": xs}, dims=("T", "Y", "X"), name=name)
        da = da.assign_coords(SUBID=(("Y", "X"), subid), ELEV=(("Y", "X"), elev))
        da.attrs["units"] = "same discharge units as input GloFAS/observations"
        return da

    def _daily_station_wide_to_dataframe(
        self,
        wide: pd.DataFrame,
        stations: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Export daily station discharge to the same ID/LON/LAT/ELEV layout as input."""
        names = [str(c) for c in wide.columns]
        rows: list[list[Any]] = []
        rows.append(["LON"] + [stations[s].get("X_ORIG", stations[s].get("X", np.nan)) for s in names])
        rows.append(["LAT"] + [stations[s].get("Y_ORIG", stations[s].get("Y", np.nan)) for s in names])
        rows.append(["DAILY/ELEV"] + [stations[s].get("ELEV", np.nan) for s in names])
        vals = wide[names].to_numpy(dtype=float)
        for i, date in enumerate(pd.DatetimeIndex(wide.index)):
            rows.append([pd.Timestamp(date).strftime("%Y%m%d")] + vals[i, :].tolist())
        return pd.DataFrame(rows, columns=["ID"] + names)

    def _prepare_glofas_highflow_context(
        self,
        observations: pd.DataFrame | xr.DataArray | xr.Dataset,
        glofas_discharge: xr.DataArray | xr.Dataset,
        glofas_uparea: xr.DataArray | xr.Dataset,
        hybas9: gpd.GeoDataFrame,
        *,
        hybas_outlets: Optional[gpd.GeoDataFrame] = None,
        highflow_config: Optional[HighWaterConfig] = None,
        match_config: Optional[GloFASPixelMatchConfig] = None,
        bias_config: Optional[GloFASHighFlowBiasConfig] = None,
        observation_variable: str = "Observation",
        glofas_variable: Optional[str] = None,
        uparea_variable: Optional[str] = None,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
        periods: Optional[ClimatologyPeriod] = None,
        station_attributes: Optional[pd.DataFrame] = None,
        reference_start: Optional[str] = None,
        reference_end: Optional[str] = None,
        n_bootstrap: int = 0,
    ) -> dict[str, Any]:
        """Prepare observations, HW/MODE definitions, HYBAS mapping and matched GloFAS."""
        auxiliary_audit=validate_glofas_auxiliary(glofas_discharge,glofas_uparea,
            discharge_variable=glofas_variable,uparea_variable=uparea_variable)
        bcfg = bias_config or GloFASHighFlowBiasConfig()
        need_modes = bcfg.correction_scope == "mode"
        hcfg = highflow_config or HighWaterConfig()
        hcfg = replace(hcfg, mode_specific_outputs=need_modes or hcfg.mode_specific_outputs, n_bootstrap=int(n_bootstrap))

        stations = self._prepare_highflow_observations(
            observations,
            variable=observation_variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
        )
        obs_series = {s: meta["series"] for s, meta in stations.items()}
        station_meta = self._station_metadata_frame(stations)
        if station_attributes is not None:
            if "station" not in station_attributes or station_attributes.station.duplicated().any():
                raise ValueError("station_attributes requires unique station identifiers")
            attrs=station_attributes.copy();attrs["station"]=attrs.station.astype(str)
            cols=[c for c in attrs if c=="station" or c not in station_meta]
            station_meta=station_meta.merge(attrs[cols],on="station",how="left")

        hf = self.WAS_HighFlow_Predictands(
            observations,
            variable=observation_variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
            config=hcfg,
            mode_specific=hcfg.mode_specific_outputs,
            periods=periods,
            reference_start=reference_start,
            reference_end=reference_end,
            n_bootstrap=n_bootstrap,
            output_format="both",
            return_diagnostics=True,
        )
        diagnostics = hf["diagnostics"]
        windows = select_highflow_windows(
            diagnostics,
            scope=bcfg.correction_scope,
            selector=bcfg.selector,
        )

        mapping = prepare_glofas_hybas_matching(
            station_meta,
            hybas9,
            glofas_uparea,
            hybas_outlets=hybas_outlets,
            uparea_variable=uparea_variable,
            match_config=match_config,
        )
        mapping["auxiliary_audit"]=auxiliary_audit
        qwide, qx = extract_glofas_discharge_at_matches(
            glofas_discharge,
            mapping["glofas_matches"],
            variable=glofas_variable,
            valid_status=(match_config or GloFASPixelMatchConfig()).accepted_match_statuses,
        )
        # Include actual matched coordinates for duplicate-pixel CV exclusions.
        matched=mapping["glofas_matches"]
        mapping["station_hybas"]=mapping["station_hybas"].merge(
            matched[[c for c in ("station","glofas_lon","glofas_lat") if c in matched]],on="station",how="left")
        return {
            "stations": stations,
            "observations": obs_series,
            "highflow": hf,
            "diagnostics": diagnostics,
            "windows": windows,
            "mapping": mapping,
            "glofas_wide": qwide,
            "glofas_outlet_xarray": qx,
            "highflow_config": hcfg,
            "match_config": match_config or GloFASPixelMatchConfig(),
            "bias_config": bcfg,
        }

    def WAS_Match_GloFAS_Pixels(
        self,
        observations: pd.DataFrame | xr.DataArray | xr.Dataset,
        hybas9: gpd.GeoDataFrame,
        glofas_uparea: xr.DataArray | xr.Dataset,
        *,
        hybas_outlets: Optional[gpd.GeoDataFrame] = None,
        match_config: Optional[GloFASPixelMatchConfig] = None,
        station_attributes: Optional[pd.DataFrame] = None,
        observation_variable: str = "Observation",
        uparea_variable: Optional[str] = None,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
    ) -> dict[str, Any]:
        """Audit station -> HYBAS L9 -> hydrologically plausible GloFAS pixel mapping."""
        stations = self._prepare_highflow_observations(
            observations,
            variable=observation_variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
        )
        metadata=self._station_metadata_frame(stations)
        if station_attributes is not None:
            if "station" not in station_attributes or station_attributes.station.duplicated().any():
                raise ValueError("station_attributes requires unique station identifiers")
            attrs=station_attributes.copy();attrs["station"]=attrs.station.astype(str)
            metadata=metadata.merge(attrs[[c for c in attrs if c=="station" or c not in metadata]],on="station",how="left")
        return prepare_glofas_hybas_matching(
            metadata,
            hybas9,
            glofas_uparea,
            hybas_outlets=hybas_outlets,
            uparea_variable=uparea_variable,
            match_config=match_config,
        )

    def correct_glofas_highflow(
        self,
        observations: pd.DataFrame | xr.DataArray | xr.Dataset,
        glofas_discharge: xr.DataArray | xr.Dataset,
        glofas_uparea: xr.DataArray | xr.Dataset,
        hybas9: gpd.GeoDataFrame,
        *,
        hybas_outlets: Optional[gpd.GeoDataFrame] = None,
        highflow_config: Optional[HighWaterConfig] = None,
        match_config: Optional[GloFASPixelMatchConfig] = None,
        bias_config: Optional[GloFASHighFlowBiasConfig] = None,
        observation_variable: str = "Observation",
        glofas_variable: Optional[str] = None,
        uparea_variable: Optional[str] = None,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
        periods: Optional[ClimatologyPeriod] = None,
        station_attributes: Optional[pd.DataFrame] = None,
        reference_start: Optional[str] = None,
        reference_end: Optional[str] = None,
        calibration_start: Optional[str] = None,
        calibration_end: Optional[str] = None,
        n_bootstrap: int = 0,
        regionalize_all_hybas: bool = True,
    ) -> dict[str, Any]:
        """Correct matched GloFAS daily discharge on observed high-flow windows.

        The observation defines the HW/MODE calendar. The correction is fitted on
        daily paired data, then optionally regionalized to ungauged HYBAS L9
        outlets. Scalar-factor regionalization is available for mean_ratio and
        log_ratio. EQM is retained at gauged stations only.
        """
        ctx = self._prepare_glofas_highflow_context(
            observations, glofas_discharge, glofas_uparea, hybas9,
            hybas_outlets=hybas_outlets,
            highflow_config=highflow_config,
            match_config=match_config,
            bias_config=bias_config,
            observation_variable=observation_variable,
            glofas_variable=glofas_variable,
            uparea_variable=uparea_variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
            periods=periods,
            station_attributes=station_attributes,
            reference_start=reference_start,
            reference_end=reference_end,
            n_bootstrap=n_bootstrap,
        )
        bcfg = ctx["bias_config"]
        if periods is not None:
            if calibration_start is not None or calibration_end is not None:
                raise ValueError("Use periods or calibration_start/end, not both")
            calibration_start,calibration_end=periods.calibration
        fits, models = fit_gauged_highflow_corrections(
            ctx["observations"], ctx["glofas_wide"], ctx["windows"], config=bcfg,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
        )
        if periods is not None:
            ctx["glofas_wide"]=slice_series(ctx["glofas_wide"],*periods.application)
        corrected_wide = apply_gauged_highflow_corrections(
            ctx["glofas_wide"], models, outside_highflow=bcfg.outside_highflow
        )
        raw_xr = self._daily_station_wide_to_xarray(ctx["glofas_wide"], ctx["stations"], name="GloFAS_raw")
        cor_xr = self._daily_station_wide_to_xarray(corrected_wide, ctx["stations"], name="GloFAS_highflow_corrected")
        raw_df = self._daily_station_wide_to_dataframe(ctx["glofas_wide"], ctx["stations"])
        cor_df = self._daily_station_wide_to_dataframe(corrected_wide, ctx["stations"])

        # Attach HYBAS_ID to fitted gauged factors for network propagation.
        station_hybas = ctx["mapping"]["station_hybas"]
        merge_cols = [c for c in ["station", "HYBAS_ID"] if c in station_hybas.columns]
        fits_hybas = fits.merge(station_hybas[merge_cols].drop_duplicates("station"), on="station", how="left") if not fits.empty else fits

        regional = None
        if regionalize_all_hybas and bcfg.propagate_to_ungauged:
            if bcfg.method == "eqm":
                regional = {
                    "status": "NOT_REGIONALIZED",
                    "reason": "EQM is distributional; scalar-factor propagation is intentionally disabled.",
                }
            else:
                okfits = fits_hybas[fits_hybas.get("fit_status", "") == "OK"].copy()
                if okfits.empty:
                    raise ValueError("No fitted gauged factor is available for regionalization; inspect gauged_fit or disable regionalize_all_hybas")
                factors = propagate_multiplicative_factors(hybas9, okfits, station_hybas,
                    hybas_outlets=hybas_outlets, config=bcfg)
                if hybas_outlets is None:
                    regional = {
                        "status": "FACTORS_ONLY",
                        "factors": factors,
                        "reason": "hybas_outlets is required to extract every L9 GloFAS outlet.",
                    }
                else:
                    targets = make_hybas_outlet_targets(hybas9, hybas_outlets)
                    all_matches, all_candidates = match_hybas_outlets_to_glofas(
                        targets, glofas_uparea, variable=uparea_variable, config=ctx["match_config"]
                    )
                    qall = extract_all_hybas_glofas_outlets(
                        glofas_discharge, all_matches, variable=glofas_variable,
                        valid_status=ctx["match_config"].accepted_match_statuses
                    )
                    if periods is not None:qall=slice_dataarray(qall,*periods.application)
                    qcorr, audit = apply_propagated_highflow_factors(
                        qall, factors, ctx["windows"], outside_highflow=bcfg.outside_highflow
                    )
                    regional = {
                        "status": "OK",
                        "factors": factors,
                        "mapping": all_matches,
                        "mapping_candidates": all_candidates,
                        "application_audit": audit,
                        "raw": {
                            "xarray": qall,
                            "dataframe": outlet_xarray_to_picture(qall, targets),
                        },
                        "corrected": {
                            "xarray": qcorr,
                            "dataframe": outlet_xarray_to_picture(qcorr, targets),
                        },
                    }

        return {
            "periods": periods.describe() if periods is not None else None,
            "mass_balance": "Independent outlet correction; network mass balance is not enforced",
            "mapping": ctx["mapping"],
            "highflow_diagnostics": ctx["diagnostics"],
            "windows": ctx["windows"],
            "gauged_fit": fits_hybas,
            "gauged_models": models,
            "gauged": {
                "raw": {"xarray": raw_xr, "dataframe": raw_df},
                "corrected": {"xarray": cor_xr, "dataframe": cor_df},
            },
            "regional": regional,
        }

    def evaluate_glofas_highflow_correction(
        self,
        observations: pd.DataFrame | xr.DataArray | xr.Dataset,
        glofas_discharge: xr.DataArray | xr.Dataset,
        glofas_uparea: xr.DataArray | xr.Dataset,
        hybas9: gpd.GeoDataFrame,
        *,
        hybas_outlets: Optional[gpd.GeoDataFrame] = None,
        highflow_config: Optional[HighWaterConfig] = None,
        match_config: Optional[GloFASPixelMatchConfig] = None,
        bias_config: Optional[GloFASHighFlowBiasConfig] = None,
        evaluation_config: Optional[GloFASCorrectionEvaluationConfig] = None,
        observation_variable: str = "Observation",
        glofas_variable: Optional[str] = None,
        uparea_variable: Optional[str] = None,
        missing_values: Optional[tuple[float, ...] | list[float]] = None,
        negative_flow_as_nan: bool = True,
        periods: Optional[ClimatologyPeriod] = None,
        station_attributes: Optional[pd.DataFrame] = None,
        reference_start: Optional[str] = None,
        reference_end: Optional[str] = None,
        n_bootstrap: int = 0,
    ) -> dict[str, Any]:
        """Evaluate GloFAS high-flow correction with independent CV.

        Evaluation levels
        -----------------
        temporal:
            Leave-one-hydrological-year-out at a known station.
        spatial:
            Leave-one-station-out; the withheld station is treated as ungauged
            and reconstructed by network, IDW-log or hybrid regionalization.
        spatiotemporal:
            Optional strict combination: station and hydrological year withheld.

        Returns daily scores, Q_HW_MEAN/Q_HW_MAX scores, peak timing errors,
        factor-transfer diagnostics, pixel-match quality and a method ranking.
        """
        ctx = self._prepare_glofas_highflow_context(
            observations, glofas_discharge, glofas_uparea, hybas9,
            hybas_outlets=hybas_outlets,
            highflow_config=highflow_config,
            match_config=match_config,
            bias_config=bias_config,
            observation_variable=observation_variable,
            glofas_variable=glofas_variable,
            uparea_variable=uparea_variable,
            missing_values=missing_values,
            negative_flow_as_nan=negative_flow_as_nan,
            periods=periods,
            station_attributes=station_attributes,
            reference_start=reference_start,
            reference_end=reference_end,
            n_bootstrap=n_bootstrap,
        )
        evaluation = evaluate_glofas_highflow_correction(
            ctx["observations"],
            ctx["glofas_wide"],
            ctx["windows"],
            ctx["diagnostics"]["station_summary"],
            ctx["mapping"]["station_hybas"],
            hybas9,
            pixel_matches=ctx["mapping"]["glofas_matches"],
            bias_config=ctx["bias_config"],
            evaluation_config=evaluation_config,
            highflow_config=ctx["highflow_config"],
            periods=periods,
        )
        evaluation["mapping"] = ctx["mapping"]
        evaluation["windows"] = ctx["windows"]
        evaluation["highflow_diagnostics"] = ctx["diagnostics"]
        return evaluation


    def regionalize_highflow_windows(self, *args, **kwargs):
        """Gridded calendar correction; see GridWindowConfig and explicit supports."""
        return regionalize_highflow_windows(*args, **kwargs)

    def build_highflow_predictand(self, *, glofas_discharge, glofas_uparea=None,
            basins=None, outlets=None, hydrosheds=None, hybas_level=None,
            observations=None, observation_variable="Observation", station_attributes=None,
            variable=None, config=None, aggregation_config=None, highflow_config=None,
            match_config=None, periods=None, output_format="both", years=None):
        """v5: windows -> annual aggregates -> bias correction of aggregates.

        PredictandConfig(reference_source="glofas") is a complete no-observation
        path. HYBAS level is inherited from hydrosheds or the supplied layers.
        See highflow_predictand.build_highflow_predictand for all controls.
        """
        if __package__:
            from .highflow_predictand import build_highflow_predictand
        else:
            from highflow_predictand import build_highflow_predictand
        return build_highflow_predictand(self,glofas_discharge=glofas_discharge,
            glofas_uparea=glofas_uparea,basins=basins,outlets=outlets,hydrosheds=hydrosheds,
            hybas_level=hybas_level,observations=observations,observation_variable=observation_variable,
            station_attributes=station_attributes,variable=variable,config=config,
            aggregation_config=aggregation_config,highflow_config=highflow_config,
            match_config=match_config,periods=periods,output_format=output_format,years=years)

    def aggregate_highflow_field(self, data, *, reference=None, outlets=None,
            windows=None, variable=None, config=None, kind=None, selector=None,
            years=None, output_format="both", **kwargs):
        """Same extraction/aggregation for any T,Y,X input (optional members).

        Pass reference=<result of build_highflow_predictand> to reuse exactly
        its HYBAS catalogue and calendars. No bias correction is applied.
        Mapping is recomputed for this source unless explicitly supplied.
        """
        if __package__:
            from .highflow_predictand import aggregate_with_reference
            from .highflow_outlet_aggregation import aggregate_field,format_aggregation
        else:
            from highflow_predictand import aggregate_with_reference
            from highflow_outlet_aggregation import aggregate_field,format_aggregation
        if reference is not None:
            if outlets is not None or windows is not None:
                raise ValueError("Pass reference OR outlets/windows")
            return aggregate_with_reference(data,reference,variable=variable,config=config,
                kind=kind,selector=selector,years=years,output_format=output_format,**kwargs)
        if outlets is None or windows is None:raise ValueError("outlets and windows required")
        result=aggregate_field(data,outlets,windows,variable=variable,config=config,
            kind=kind or "season",selector="main" if selector is None else selector,
            years=years,**kwargs)
        return format_aggregation(result,output_format=output_format)

    def evaluate_highflow_predictand(self, reference, *, config=None):
        """Evaluate seasonal-aggregate correction on held-out years.

        This is conditional on the reference windows. In no-observation mode,
        returns UNAVAILABLE_WITHOUT_OBSERVATIONAL_PAIRS, not artificial scores.
        """
        if __package__:
            from .highflow_seasonal_evaluation import evaluate_seasonal_correction
        else:
            from highflow_seasonal_evaluation import evaluate_seasonal_correction
        return evaluate_seasonal_correction(reference,config=config)


# Optional monthly DAILY bias correction, maintained as a v5.1.0 extension.
if __package__:
    from .highflow_monthly_bias import MonthlyBiasConfig
else:
    from highflow_monthly_bias import MonthlyBiasConfig

__all__ = [
    'WAS_Hydro', 'HighWaterConfig', 'ClimatologyPeriod', 'GridWindowConfig', 'TopKrigingConfig', 'GloFASPixelMatchConfig',
    'GloFASHighFlowBiasConfig', 'GloFASCorrectionEvaluationConfig', 'PredictandConfig', 'AggregationConfig', 'MonthlyBiasConfig'
]
