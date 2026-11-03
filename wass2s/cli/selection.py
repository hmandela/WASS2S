"""Target-independent GCM selection shared by operational CLI stages.

A skill ranking computed from the complete verification period must not choose
the inputs of a cross-validation on that same period. The supported alternative
here keeps the user's declared, available roster fixed for every validation fold
and the operational forecast. ``requested_centres`` and ``available_inputs``
must be called first: this module neither discovers archives nor assesses skill.

To reduce predictor count, declare ``--gcm-models`` or ``--drop-gcm`` before
examining verification scores. Global MAE/Pearson scores remain diagnostics.
"""
from __future__ import annotations

from collections.abc import Iterable

from .roster import canonical_center


GCM_SELECTION_PROCEDURE = "fixed_available_roster_v1"


def fixed_gcm_roster(centres: Iterable[str]) -> list[str]:
    """Keep every available declared GCM once, in its original stable order.

    The spelling of the first occurrence is preserved for the downloader and
    ``process_datasets_for_mme``. Alias spellings of one centre/variable are
    deduplicated; different variables from one centre remain distinct inputs.
    No observation, score, threshold or target-dependent top-N enters this API.
    """
    if isinstance(centres, (str, bytes)):
        raise TypeError("A GCM roster must be an iterable of model IDs, not one string.")
    result, seen = [], set()
    for value in centres:
        if not isinstance(value, str):
            raise TypeError("Every GCM ID must be a string.")
        model = value.strip()
        center = canonical_center(model)
        if not center:
            raise ValueError("A GCM ID must contain a nonempty centre name.")
        variable = model.partition(".")[2].upper()
        key = center, variable
        if key not in seen:
            seen.add(key)
            result.append(model)
    if not result:
        raise ValueError("No available GCM in the declared fixed roster.")
    return result


def fixed_gcm_selection_metadata(centres: Iterable[str]) -> dict:
    """Describe the actual selector in saved products and CNN provenance."""
    return {
        "best_models": fixed_gcm_roster(centres),  # retain the old JSON key
        "selection": GCM_SELECTION_PROCEDURE,
        "uses_target_scores": False,
        "validation_scope": (
            "The same declared available GCMs are used in every validation fold "
            "and in the forecast; MAE/Pearson diagnostics do not select inputs."
        ),
    }
