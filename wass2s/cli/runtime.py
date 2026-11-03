"""Shared scientific contracts for the PRCP and TEMP operational CLIs.

No data are downloaded by this module. Hindcast probabilities carry their
declared climatological event definition through NetCDF; mixtures never re-fit it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

from .probability import probability_calibration_mode

REVISION = "2026-09-28.1-outer-default-linear-ml-kharin-bma"
DIAGNOSTIC_COORDS = ("fold_threshold_lower", "fold_threshold_upper",
                     "fold_reference_probability")
EVENT_DEFINITION = "PB: y<=lower; PN: lower<y<=upper; PA: y>upper"
_ACTIVE = {}
_STAGE_START = {}


def _plain(array):
    return array.drop_vars([k for k in DIAGNOSTIC_COORDS if k in array.coords])


def _target(array):
    array = _plain(array)
    if "M" in array.dims:
        if array.sizes["M"] != 1:
            raise ValueError("An observation/deterministic target cannot have multiple M members")
        array = array.squeeze("M", drop=True)
    if set(array.dims) != {"T", "Y", "X"}:
        raise ValueError(f"Expected target dimensions T,Y,X; received {array.dims}")
    return array.transpose("T", "Y", "X")


def _check_probability(prob):
    if list(prob.probability.values) != ["PB", "PN", "PA"]:
        raise ValueError("Probabilities must have ordered PB, PN, PA categories")
    finite = np.isfinite(prob)
    partial = finite.any("probability") & ~finite.all("probability")
    bad = np.isinf(prob).any() | ((prob < 0) | (prob > 1)).any() | partial.any()
    bad = bad | (finite.all("probability") &
                 (abs(prob.sum("probability", skipna=False) - 1) > 1e-6)).any()
    if bool(bad.compute()):
        raise ValueError("Invalid probabilities: require a simplex or three missing values")


def _attach(prob, q, ref, kind):
    prob = _plain(prob)
    prob, q, ref = xr.align(prob, _plain(q), _plain(ref), join="exact")
    if "T" not in q.dims:
        q = q.expand_dims(T=prob["T"])
    if "T" not in ref.dims:
        ref = ref.expand_dims(T=prob["T"])
    out = prob.assign_coords({
        DIAGNOSTIC_COORDS[0]: q.isel(quantile=0, drop=True),
        DIAGNOSTIC_COORDS[1]: q.isel(quantile=1, drop=True),
        DIAGNOSTIC_COORDS[2]: ref,
    })
    _check_probability(out)
    out.attrs.update(event_definition=EVENT_DEFINITION,
                     probability_contract=REVISION, probability_scope=kind)
    return out


def attach_cv_diagnostics(prob, cv):
    """Attach the event thresholds and reference frequencies used for prediction."""
    diagnostics = getattr(cv, "cross_validation_diagnostics_", None)
    if diagnostics is None or not {"thresholds", "reference_probabilities"}.issubset(diagnostics):
        raise ValueError("Cross-validator did not return probability event diagnostics")
    return _attach(prob, diagnostics.thresholds,
                   diagnostics.reference_probabilities, "hindcast")


def probability_diagnostics(prob):
    missing = [name for name in DIAGNOSTIC_COORDS if name not in prob.coords]
    if missing:
        raise ValueError(f"Missing probability event provenance: {missing}. Re-run the producing stage.")
    if (prob.attrs.get("event_definition") != EVENT_DEFINITION or
        prob.attrs.get("probability_contract") != REVISION or
        prob.attrs.get("probability_scope") not in {"hindcast", "operational"}):
        raise ValueError("Missing or unsupported probability event convention/contract/scope")
    lo, hi, ref = (prob.coords[name] for name in DIAGNOSTIC_COORDS)
    lo, hi, ref = (_plain(v) for v in (lo, hi, ref))
    q = xr.concat([lo, hi], dim=xr.IndexVariable("quantile", [.33, .67]))
    return q, ref


def probability_mixture(arrays, label="equal probability mixture", require_diagnostics=True):
    """Fixed linear pool; all declared members must be valid at a given case.

    Missing members yield a missing pooled forecast, never different implicit
    weights by year. Event thresholds and reference frequencies must agree.
    """
    values = list(arrays.values()) if isinstance(arrays, dict) else list(arrays)
    if not values:
        raise ValueError("Cannot pool an empty model roster")
    first = values[0]
    if require_diagnostics:
        q, ref = probability_diagnostics(first)
    for item in values:
        _check_probability(item)
        xr.align(_plain(first), _plain(item), join="exact")
        if require_diagnostics:
            other_q, other_ref = probability_diagnostics(item)
            for attribute in ("event_definition", "probability_scope", "probability_contract"):
                if item.attrs.get(attribute) != first.attrs.get(attribute):
                    raise ValueError(f"{label}: incompatible {attribute}")
            try:
                xr.testing.assert_allclose(q, other_q, rtol=1e-10, atol=1e-10)
                xr.testing.assert_allclose(ref, other_ref, rtol=1e-10, atol=1e-10)
            except AssertionError as exc:
                raise ValueError(f"{label}: incompatible probability events/reference; "
                                 "recompute members with the same declared event reference") from exc
    stack = xr.concat([_plain(v) for v in values], dim="pool_member", join="exact",
                      coords="minimal", compat="equals")
    result = stack.mean("pool_member", skipna=False)
    if require_diagnostics:
        result = _attach(result, q, ref, first.attrs.get("probability_scope", "hindcast"))
    result.attrs.update(pooling="fixed equal linear pool; complete declared roster",
                        pool_size=len(values), model=label)
    return result


def deterministic_mixture(arrays):
    values = list(arrays.values()) if isinstance(arrays, dict) else list(arrays)
    if not values:
        raise ValueError("Empty deterministic roster")
    return xr.concat([_plain(v) for v in values], dim="pool_member", join="exact",
                     coords="minimal", compat="equals").mean("pool_member", skipna=False)


def mean_calibrated_probabilities(fields, *, calibration, forecast=False):
    """Pool CCA probabilities without recalibrating the complete hindcast.

    The resulting predictive distribution is an equal mixture of member
    distributions, not the fitted distribution of the deterministic mean.
    Hindcasts retain the exact event/reference diagnostics of their members.
    Missing members yield missing probabilities, preserving the fixed roster.
    ``calibration`` labels the component procedure; it does not certify any
    upstream choice of GCMs/domains as independent of the verification data.
    """
    calibration = str(calibration).strip().lower()
    if calibration == "inner":
        calibration = "nested"
    if calibration not in {"nested", "outer"}:
        raise ValueError("CCA probability calibration must be 'nested' or 'outer'")
    values = list(fields.values()) if isinstance(fields, dict) else list(fields)
    if not values:
        raise ValueError("Cannot pool an empty CCA probability roster")
    provenance = [all(key in value.coords for key in DIAGNOSTIC_COORDS)
                  for value in values]
    if forecast and any(provenance) and not all(provenance):
        raise ValueError("CCA forecast members have inconsistent event provenance")
    if not forecast:
        expected = int(calibration == "nested")
        for value in values:
            actual = value.attrs.get("calibration_strictly_nested")
            declared = value.attrs.get("probability_calibration")
            if declared == "inner":
                declared = "nested"
            # Inner residual refits with reused outer-selected hyperparameters
            # are a documented conditional approximation, not outer residuals.
            mismatched = (declared != calibration if declared is not None else
                          actual is not None and int(actual) != expected)
            if mismatched:
                raise ValueError("CCA member calibration differs from the requested mode; "
                                 "recompute the producing stage")
    result = probability_mixture(
        values, label="CCA equal mixture of calibrated member probabilities",
        require_diagnostics=(not forecast or all(provenance)))
    # xarray may inherit the first member's attributes during the mean.
    # The component procedure must not be promoted to a claim about all
    # upstream model selection in the pooled product.
    result.attrs.pop("calibration_strictly_nested", None)
    result.attrs.update(
        component_probability_calibration=calibration,
        probability_pooling="equal mixture of member distributions",
        probability_schema=2,
        calibration=("operational member probabilities from historical hindcasts"
                     if forecast else
                     "equal pool of training-only nested member probabilities"
                     if calibration == "nested" else
                     "equal pool of approximate outer-residual member probabilities"),
    )
    if not forecast:
        result.attrs["component_calibration_strictly_nested"] = int(
            calibration == "nested" and all(
                bool(value.attrs.get("calibration_strictly_nested", True)) for value in values))
    return result


def fold_probability_scores(obs, prob):
    from wass2s.was_crossfit import score_fold_probabilities
    q, ref = probability_diagnostics(prob)
    return score_fold_probabilities(_target(obs), _plain(prob), q, ref)


def probability_score_kwargs(prob):
    """Use the exact events and reference frequencies saved with probabilities."""
    if all(name in prob.coords for name in DIAGNOSTIC_COORDS):
        q, ref = probability_diagnostics(prob)
        return {"thresholds": q, "reference_probabilities": ref}
    return {}


def configured_cross_validator(args, observations):
    """Apply the declared event convention while preserving each CLI fold count."""
    from wass2s.was_cross_validate import WAS_Cross_Validator
    class DiagnosticCrossValidator(WAS_Cross_Validator):
        def cross_validate(self, *values, **options):
            outputs = super().cross_validate(*values, **options)
            diagnostics = getattr(self, "cross_validation_diagnostics_", None)
            if diagnostics is not None and {"thresholds", "reference_probabilities"}.issubset(diagnostics):
                outputs = tuple(attach_cv_diagnostics(value, self)
                                if isinstance(value, xr.DataArray) and "probability" in value.dims
                                else value for value in outputs)
            return outputs

    return DiagnosticCrossValidator(
        n_splits=observations.sizes["T"], nb_omit=2,
        tercile_reference=getattr(args, "tercile_reference", "climatology"))


def climatology_sample(observations, start, end):
    """Declared reference-period observations, independent of the fitting folds."""
    return _target(observations).sel(T=slice(str(start), str(end)))


def cv_probability_kwargs(args, variable=None):
    variable = variable or getattr(args, "variable", "PRCP")
    result = {"strict_oos_probability": True,
              "probability_calibration": probability_calibration_mode(args),
              "probability_inner_folds": int(getattr(args, "probability_inner_folds", None) or getattr(args, "svml_probability_folds", 4)),
              "probability_inner_retune": bool(getattr(args, "probability_inner_retune", True)),
              "tercile_reference": getattr(args, "tercile_reference", "climatology")}
    if args.dist_method == "bestfit":
        result["probability_distribution_map"] = (
            {"norm": 1, "lognorm": 2, "gamma": 4} if variable == "PRCP"
            else {"norm": 1, "t_dist": 6})
    return result


def operational_probabilities(obs, hdet, fdet, args, model=None):
    """Calibrate only an unseen future using past OOF residuals.

    This is deliberately never used to dress a complete historical OOF mean:
    doing so would let its own verification observation enter calibration.
    """
    from wass2s.was_crossfit import CrossFitConfig
    from wass2s.was_cross_validate import _training_probabilities, _climatology_options
    obs, hdet = xr.align(_target(obs), _target(hdet), join="exact")
    fdet = _target(fdet)
    if "T" not in fdet.dims:
        raise ValueError("Operational prediction must retain its actual target T coordinate")
    if set(obs["T"].dt.year.values) & set(fdet["T"].dt.year.values):
        raise ValueError("Operational target year occurs in training observations")
    if int(fdet["T"].dt.year.min()) <= int(obs["T"].dt.year.max()):
        raise ValueError("Operational forecast must follow its training history")
    if model is None:
        if args.dist_method != "nonparam":
            raise ValueError("Parametric operational calibration requires the fitted model")
        model = SimpleNamespace(dist_method="nonparam")
    cfg = CrossFitConfig(tercile_reference=getattr(args, "tercile_reference", "climatology"),
                         climatology_start=args.clim_years[0],
                         climatology_end=args.clim_years[1])
    options = _climatology_options(cfg, obs, args.clim_years[0], args.clim_years[1],
                                   cv_probability_kwargs(args))
    prob, q, ref, _ = _training_probabilities(
        model, fdet, obs - hdet, obs, cfg, options)
    out = _attach(prob, q, ref, "operational")
    out.attrs["calibration"] = "past out-of-fold residuals; future target excluded"
    return out


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def available_cpus():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", affinity))
    return max(1, min(affinity, allocation))


def source_hindcast_years(args):
    """CDS initialization interval; args.hindcast_years identifies valid targets."""
    return getattr(args, "hindcast_initialization_years", args.hindcast_years)


def normalize_hindcast_calendar(args):
    """Keep archive initialization years separate from actual target years.

    For a December initialization and offsets 1,2,3, a 1993--2016 archive
    verifies JFM 1994--2017. No observation or forecast is positionally shifted.
    """
    init = int(args.init_month)
    leads = [int(v) for v in args.lead_time]
    if not 1 <= init <= 12 or len(leads) != 3:
        raise ValueError("Require a valid initialization month and three lead offsets")
    shift = (init - 1 + leads[1]) // 12
    original = list(source_hindcast_years(args))
    args.hindcast_initialization_years = original
    args.hindcast_years = [int(v) + shift for v in original]
    args.target_year_offset = shift
    return args


def register_run(workdir, config):
    root = Path(workdir).absolute()
    path = root / "scientific_run_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("fingerprint") != config["fingerprint"]:
            requested = config.get("arguments", {})
            if requested.get("only") == ["consolidate"]:
                # A new second-stage experiment consumes immutable, registered
                # upstream predictions. It must not rewrite their configuration
                # or pretend that they were regenerated by the new sources.
                ignored = {"only", "redo", "force", "reuse_obs", "print_plan", "workdir",
                           "logo", "country_code", "ncores", "threads_per_worker", "no_dask",
                           "numba_threads", "verification_min_cells", "svml_n_jobs",
                           "svml_optuna_n_jobs", "svml_cv_n_jobs", "svml_tuning_model_n_jobs",
                           "cluster_n_jobs",
                           "ngr_prcp_family"}
                original = previous.get("arguments", {})
                keys = set(original) | set(requested)
                differences = [key for key in keys
                               if key not in ignored and not key.startswith("consolidation_")
                               and original.get(key) != requested.get(key)]
                if previous.get("variable") != config.get("variable") or differences:
                    raise ValueError("Consolidation-only reuse requires the same upstream settings; "
                                     f"changed: {sorted(differences)}. Reuse the original launcher settings.")
                config["source_products_fingerprint"] = previous["fingerprint"]
                config["source_products_sources"] = previous.get("sources", {})
                atomic_json(root / "scientific_consolidation_config.json", config)
                _ACTIVE[str(root)] = config
                return
            raise ValueError("This workdir contains a different scientific configuration/source revision. "
                             "Use --only consolidate with the original upstream settings to compare "
                             "registered hindcasts, or choose a new --workdir for a new upstream run.")
    elif any((root / "intermediate").glob("*_hindcasts.nc")):
        raise ValueError("Unversioned hindcasts found in workdir; use a new --workdir")
    atomic_json(path, config)
    _ACTIVE[str(root)] = config


def _artifact_files(root):
    files = list((root / "intermediate").rglob("*.nc")) + list(root.glob("*predictand*.nc"))
    files += list(root.glob("best_distribution_params.nc"))
    files += list((root / "intermediate").glob("*.pkl"))
    files += list((root / "forecasts").glob("*.nc"))
    files += list((root / "scores").glob("*.nc"))
    files += list((root / "scores").glob("*.json"))
    for suffix in ("*.pkl", "*.json", "*.nc"):
        files += list((root / "models").rglob(suffix))
    files += list((root / "kharin").rglob("*predictand*.nc"))
    files += list((root / "kharin").glob("design.json"))
    return sorted(set(files))


def begin_stage(workdir, key):
    root = Path(workdir).absolute()
    files = _artifact_files(root)
    _STAGE_START[str(root)] = {str(p.relative_to(root)): _sha(p) for p in files}


def _checkpoint_path(checkdir, key):
    if not key or any(c in key for c in "/\\"):
        raise ValueError("Invalid checkpoint key")
    return Path(checkdir) / f"scientific_{key}.json"


def checkpoint_done(checkdir, key):
    root = Path(checkdir).parent.absolute()
    config = _ACTIVE.get(str(root))
    path = _checkpoint_path(checkdir, key)
    if config is None or not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("fingerprint") != config["fingerprint"] or not record.get("outputs"):
        return False
    if key == "consolidate":
        current = {str(p.relative_to(root)) for p in (root / "intermediate").glob("*.nc")
                   if not p.name.startswith("consensus_")}
        prior = {p for p in record.get("inputs", {}) if p.startswith("intermediate/")}
        if current != prior:
            return False
    return all((root / p).is_file() and _sha(root / p) == digest
               for group in ("outputs", "inputs")
               for p, digest in record.get(group, {}).items())


def checkpoint_mark(checkdir, key, meta):
    root = Path(checkdir).parent.absolute()
    config = _ACTIVE.get(str(root))
    if config is None:
        raise RuntimeError("register_run must precede stage execution")
    before = _STAGE_START.get(str(root), {})
    files = _artifact_files(root)
    outputs = {}
    for path in files:
        name, digest = str(path.relative_to(root)), _sha(path)
        if before.get(name) != digest:
            outputs[name] = digest
    previous = _checkpoint_path(checkdir, key)
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("fingerprint") == config["fingerprint"]:
            outputs = {**old.get("outputs", {}), **outputs}
    inputs = {}
    if key != "obs":
        for path in root.glob("*predictand*.nc"):
            inputs[str(path.relative_to(root))] = _sha(path)
        mask_path = root / "scores" / "target_mask.nc"
        if mask_path.exists():
            inputs[str(mask_path.relative_to(root))] = _sha(mask_path)
        if key == "consolidate":
            for path in (root / "intermediate").glob("*.nc"):
                if not path.name.startswith("consensus_"):
                    inputs[str(path.relative_to(root))] = _sha(path)
    atomic_json(previous, {"fingerprint": config["fingerprint"], "stage": key,
                           "outputs": outputs, "inputs": inputs, "metadata": meta})
