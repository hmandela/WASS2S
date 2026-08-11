# Changelog

All notable changes to this project will be documented in this file.

## Unreleased — Deprecation lifecycle for superseded classes

### Deprecated
- `WAS_Analog__` (the archived predecessor of `WAS_Analog`) now emits a `FutureWarning` pointing to `WAS_Analog` on instantiation, and will be removed in **v0.6.0**. Its implementation moved from `wass2s/was_analog.py` to `wass2s/_deprecated.py`, where it's now a thin subclass of `WAS_Analog` instead of a ~1,700-line parallel copy — bug fixes to `WAS_Analog` now apply to it automatically. Import path (`from wass2s import WAS_Analog__`) is unchanged.

### Removed
- Deleted `WAS_Analog.download_reanalysis_` (trailing underscore) — an unused, superseded draft of `download_reanalysis` with no callers anywhere in the codebase.

### Notes for contributors
- New pattern for deprecating a public class/function: move it to `wass2s/_deprecated.py`, make it a thin subclass/wrapper of its replacement, and call `wass2s._lifecycle.warn_deprecated(old, new, removed_in=...)` from its entry point. Keep re-exporting it from `wass2s/__init__.py` so the public import path doesn't change.

## Unreleased — Environment management: conda → pixi

### Added
- Environments are now managed with [pixi](https://pixi.sh) via `pyproject.toml`, covering Linux, Windows, **and macOS** (macOS support is new — the previous conda setup only shipped Linux/Windows env files).
- `pixi.lock` provides fully reproducible, cross-platform dependency resolution.

### Changed
- README installation instructions now lead with `pixi install` / `pixi shell` as the recommended setup path.
- Dropped the `defaults` conda channel from the environment spec (unused, and gated behind Anaconda's Terms of Service).

### Unchanged / still supported
- `pip install wass2s` remains available for a lightweight install.
- The original `WAS_S2S_linux.yml` / `WAS_S2S_windows.yml` conda files are kept for users on the legacy workflow.

### Notes for contributors
- Package versions and platform-specific rules (e.g. `xgboost`/`lightgbm`/`xskillscore` via PyPI on Windows, via conda elsewhere) now live in `[tool.pixi.*]` tables in `pyproject.toml` instead of two separate YAML files — update dependencies there going forward.

## v0.4.10.1 — Initial release

**wass2s** is a Python toolkit for seasonal (and subseasonal) climate forecasting in West Africa and the Sahel, built to support the WMO's guidelines for objective, operational forecasting.

### Core capabilities
- **Data acquisition** (`was_download.py`): download NMME, C3S, and reanalysis products (AgERA5, ERSST, TAMSAT, CHIRPS) via CDS API and IRI Data Library.
- **Predictand preparation** (`was_compute_predictand.py`, `was_merge_predictand.py`, `was_transformdata.py`): build and merge observational target datasets, transform/regrid as needed.
- **Forecast methods**:
  - Statistical: CCA (`was_cca.py`, `was_cca_2.py`), linear models (`was_linear_models.py`), PCR (`was_pcr.py`), EOF-based analysis (`was_eof.py`), analog methods (`was_analog.py`).
  - Machine learning (`was_machine_learning.py`).
  - Multi-model ensembling (`was_mme.py`).
- **Bias correction** (`was_bias_correction.py`).
- **Cross-validation & verification** (`was_cross_validate.py`, `was_verification.py`) for objective skill assessment.
- **Seasonal analysis workflows** (`was_seasonal_analysis.py`) and agro-climatic indices (`ceac_agro.py`).

### Install
- `pip install wass2s`, or `conda env create -f WAS_S2S_linux.yml` / `WAS_S2S_windows.yml`.

### Known limitations
- Windows/Linux conda environments only (no macOS support yet).
- No formal changelog prior to this entry — history is available via `git log` but not curated per-release.
