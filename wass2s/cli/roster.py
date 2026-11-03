"""Resolve operational GCM availability before screening or fitting models.

Availability is a run-level choice, never a per-year or per-fold substitution.
Only absent archives are ignored; corrupt data, invalid units and programming
errors must still raise in the existing readers and validators.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path


def canonical_center(value):
    return "".join(c for c in str(value).split(".")[0].lower() if c.isalnum())


def resolve_archives(archives, roster, label, *, policy="available", report_path=None):
    """Return requested centres present in ALL archives, trimming maps in place.

    Keys tolerate downloader spelling differences (ECMWF_51 / ecmwf51).
    An absent entry, None/empty path or nonexistent file counts as unavailable.
    Duplicate canonical keys are errors, not an availability condition.
    No archive is opened or deleted here. In strict mode no map is changed.
    """
    if policy not in {"available", "strict"}:
        raise ValueError(f"Unknown GCM availability policy: {policy}")
    requested = list(roster)
    normalized = [canonical_center(c) for c in requested]
    if not requested or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label}: requested GCM roster must be nonempty and unique")
    if not archives:
        raise ValueError(f"{label}: no archives supplied for availability checks")
    indices = {}
    for name, values in archives.items():
        index = {}
        for key, value in values.items():
            center = canonical_center(key)
            if center in index:
                raise ValueError(f"{label}/{name}: ambiguous duplicate GCM {center}")
            index[center] = (key, value)
        indices[name] = index
    effective, excluded = [], []
    for center, norm in zip(requested, normalized):
        missing = []
        for name, index in indices.items():
            entry = index.get(norm)
            if entry is None or entry[1] is None:
                missing.append(name)
            elif isinstance(entry[1], (str, Path)) and not Path(entry[1]).is_file():
                missing.append(name)
        if missing:
            excluded.append({"model": center, "missing_archives": missing})
            logging.warning("[%s] GCM ignored: %s; unavailable: %s", label, center, ", ".join(missing))
        else:
            effective.append(center)
    fingerprint = hashlib.sha256(json.dumps(effective, separators=(",", ":")).encode()).hexdigest()
    report = {"label": label, "policy": policy, "requested_models": requested,
              "effective_models": effective, "excluded_models": excluded,
              "roster_fingerprint": fingerprint,
              "scope": "same available centres for all hindcast years, CV folds and forecast"}
    if report_path is not None:
        from .runtime import atomic_json
        atomic_json(Path(report_path), report)
    if policy == "strict" and excluded:
        raise RuntimeError(f"{label}: declared GCM archives unavailable: {excluded}")
    if not effective:
        raise RuntimeError(f"{label}: no GCM has all required hindcast and forecast archives; see availability report")
    keep = {canonical_center(c) for c in effective}
    for values in archives.values():
        for key in list(values):
            if canonical_center(key) not in keep:
                del values[key]
    logging.info("[%s] available GCM roster (%d/%d): %s", label, len(effective), len(requested), effective)
    return effective


def reconcile_download_roster(hind, future, roster, label, *, policy="available", report_path=None):
    return resolve_archives({"hindcast": hind, "forecast": future}, roster, label,
                            policy=policy, report_path=report_path)
