"""One probability-residual calibration policy for every CLI model family.

This module deliberately has no scientific dependencies: parsers and shell
entry points can validate aliases before importing expensive model libraries.
"""
from __future__ import annotations

import argparse


class ProbabilityCalibrationAction(argparse.Action):
    """Reject contradictory repetitions before argparse loses their history."""

    def __call__(self, parser, namespace, values, option_string=None):
        values = "nested" if values == "inner" else values
        previous = getattr(namespace, self.dest, None)
        if previous is not None and previous != values:
            raise argparse.ArgumentError(
                self, f"Conflicting probability calibration values: {previous!r} and {values!r}")
        setattr(namespace, self.dest, values)


def probability_calibration_mode(args):
    """Return the common mode, rejecting contradictory explicit aliases.

    ``outer_oof`` is accepted only by the historical CNN option. Unspecified
    values are ``None``; the operational default is ``outer``.
    ``inner`` is a public alias for the existing research mode ``nested``.
    """
    choices = []
    for attribute in (
        "probability_calibration",
        "cca_probability_calibration",
        "unet_probability_residuals",
    ):
        value = getattr(args, attribute, None)
        if value is None:
            continue
        mode = str(value).strip().lower()
        if attribute == "unet_probability_residuals" and mode == "outer_oof":
            mode = "outer"
        if mode == "inner":
            mode = "nested"
        allowed = ("nested", "outer")
        if mode not in allowed:
            flag = "--" + attribute.replace("_", "-")
            raise ValueError(f"{flag}: invalid probability calibration {value!r}")
        choices.append((attribute, mode))
    if len({mode for _, mode in choices}) > 1:
        detail = ", ".join(
            f"--{attribute.replace('_', '-')}={mode}"
            for attribute, mode in choices
        )
        raise ValueError(f"Conflicting probability calibration options: {detail}")
    return choices[0][1] if choices else "outer"


def resolve_probability_calibration(args):
    """Resolve once after argument parsing and synchronize legacy attributes."""
    mode = probability_calibration_mode(args)
    folds = getattr(args, "probability_inner_folds", None)
    if folds is not None:
        if int(folds) < 2:
            raise ValueError("--probability-inner-folds must be at least 2")
        # A common override also reaches legacy/CNN stage-specific readers.
        args.svml_probability_folds = int(folds)
        args.consolidation_prob_folds = int(folds)
    args.probability_calibration = mode
    args.cca_probability_calibration = mode
    args.unet_probability_residuals = "outer_oof" if mode == "outer" else "nested"
    return args
